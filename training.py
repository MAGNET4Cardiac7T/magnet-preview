import torch

from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf

    

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg: DictConfig):

    train_dl = instantiate(cfg.dataset)
    
    model = instantiate(cfg.model)

    trainer = instantiate(cfg.trainer)

    # resume training if possible
    trainer.fit(model, train_dl, ckpt_path=cfg.resume_from_checkpoint)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()