import os
import os.path
from itertools import chain

import torch
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import math
import data
import losses
import noise_lib
import utils
from model import RADD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
from sampling import DiffusionSampler
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(cfg):
    try:
        dist.init_process_group(backend="nccl")
        _run(local_rank=int(os.environ["LOCAL_RANK"]), gloab_rank=int(os.environ["RANK"]),
             world_size=int(os.environ["WORLD_SIZE"]), cfg=cfg)
    finally:
        cleanup()


def _run(local_rank, gloab_rank, world_size, cfg):
    torch.cuda.set_device(local_rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if gloab_rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))
        wandb.init(dir=os.path.abspath(work_dir), project='sedd', config=OmegaConf.to_container(cfg, resolve=True),
                   name=cfg.wandb_name, job_type='train')

    # logging
    if gloab_rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))

    def mprint(msg):
        if gloab_rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
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

    
    # build score model 
    radd_model = RADD(cfg).to(device)
                    
    radd_model = DDP(radd_model, device_ids=[local_rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in radd_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        radd_model.parameters(), decay=cfg.training.ema)
    mprint(radd_model)
    mprint(f"EMA: {ema}")

    token_dim = cfg.tokens + 1


    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[local_rank], static_graph=True)

    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(radd_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.cuda.amp.GradScaler()
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=radd_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(cfg.gpt_dir) 
    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)

    train_step_fn = losses.get_step_fn(noise, token_dim, True, optimize_fn, cfg.training.accum,cfg.training.loss_type)
    eval_step_fn = losses.get_step_fn(noise, token_dim, False, optimize_fn, cfg.training.accum,cfg.training.loss_type)


    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampler = DiffusionSampler(cfg.sampling.predictor,radd_model,noise,sampling_shape,token_dim, strategy = 'direct',device = device)


    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    def log_and_wandb(matric, step, name="train_loss"):
        dist.all_reduce(matric)
        matric /= world_size

        mprint(f"step: %d, {name}: %.5e" % (step, matric.item()))
        if gloab_rank == 0:
            wandb.log({name: matric.item()}, step=step)

    while state['step'] < num_train_steps + 1:
        step = state['step']


        if cfg.data.train != "text8":
            batch = next(train_iter)['input_ids'].to(device)
        else:
            batch = next(train_iter).to(device)
        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                log_and_wandb(loss, step, 'train_loss')

            if step % cfg.training.snapshot_freq_for_preemption == 0 and gloab_rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                if cfg.data.valid != "text8":
                    eval_batch = next(eval_iter)['input_ids'].to(device)
                else:
                    eval_batch = next(eval_iter).to(device)
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                if gloab_rank == 0:
                    wandb.log({'eval_loss': eval_loss.item()}, step=step)

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                if gloab_rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(radd_model.parameters())
                    ema.copy_to(radd_model.parameters())
                    sample = sampler.sample(cfg.sampling.steps)
                    ema.restore(radd_model.parameters())

                    sentences = tokenizer.batch_decode(sample)

                    file_name = os.path.join(this_sample_dir, f"sample_{gloab_rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained(cfg.gpt_dir).to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")
                            if gloab_rank == 0:
                                wandb.log({'GPT2 perplexity': total_perplexity}, step=step)

                    dist.barrier()
