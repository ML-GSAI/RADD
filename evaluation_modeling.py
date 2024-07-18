import argparse
import datetime
import math
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import utils
from load_model import load_model
from losses import get_loss_fn
import torch.nn as nn

class WrapperDDP(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
def _run(rank, args):
    if rank == 0:
        logger = utils.get_logger(os.path.join(args.work_dir, f"zero_shot_on_{args.valid_dataset}_{args.loss_type}_logs"))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    mprint("================================")

    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    mprint("================================")

    args_dict = vars(args)
    for arg_name, arg_value in args_dict.items():
        mprint(f"{arg_name}: {arg_value}")

    model, noise = load_model(args.model_path, device)
    token_dim = model.config.tokens + 1
    model = WrapperDDP(model, device_ids=[rank], static_graph=True)

    noise = DDP(noise, device_ids=[rank], static_graph=True)

    dataloader = data.get_valid_dataloaders(args)

    eval_iter = iter(dataloader)

    loss_fn = get_loss_fn(noise,token_dim, train=False, loss_type= args.loss_type)

    total_loss = 0

    with torch.no_grad():
        batch_num = 0

        for batch in eval_iter:
            if args.valid_dataset != "text8":
                batch = batch['input_ids'].to(device)
            else:
                batch = batch.to(device)

            cur_loss = 0
            for _ in range(args.monte_carlo_timesteps):
                loss = loss_fn(model, batch).mean()
                cur_loss += loss
            cur_loss /= args.monte_carlo_timesteps

            total_loss += cur_loss
            batch_num += 1

    dist.all_reduce(total_loss)
    total_loss /= args.ngpus

    total_loss /= batch_num
    
    ppl = math.exp(total_loss / args.length)

    mprint("================================")
    mprint(f"Evaluation PPL: {ppl}")
    mprint("================================\n\n")


def run(rank, args, port):
    def setup(rank, world_size, port):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

        # Initialize the process group
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
        )


    def cleanup():
        dist.destroy_process_group()


    try:
        setup(rank, args.ngpus, port)
        _run(rank, args)
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="Evaluation On RADD models")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default="data")
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--model_path", type=str, default='JingyangOu/radd-lambda-dce')
    parser.add_argument("--monte_carlo_timesteps", type=int, default=1024)
    parser.add_argument("--ngpus", type=int, default=4)
    parser.add_argument("--valid_dataset", type=str, default="ptb")
    parser.add_argument("--work_dir", type=str, default="./logs/radd-lambda-dce")
    parser.add_argument("--loss_type", type=str, default="lambda_DCE")

    args = parser.parse_args()
    if args.loss_type == "ar_forward" or args.loss_type == "ar_backward" or args.loss_type == "ar_random":
        assert args.monte_carlo_timesteps == 1

    os.makedirs(args.work_dir, exist_ok=True)
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(args.work_dir, f"zero_shot_on_{args.valid_dataset}_{args.loss_type}_logs"))

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run, args=(args, port), nprocs=args.ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()