"""Training and evaluation"""

import hydra
import os
import run_train_ddp
import utils
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict




@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    ngpus = int(os.environ["WORLD_SIZE"])
    gloab_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["WANDB_MODE"] = "offline"
    if "load_dir" in cfg:
        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        work_dir = hydra_cfg.run.dir
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(work_dir).split('&&')[0]

	# Run the training pipeline
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    logger.info(f"Num of GPUs:{ngpus}, global rank: {gloab_rank}, local rank: {local_rank}")

    try:
        run_train_ddp.run_multiprocess(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()