import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
from itertools import product
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.magnet.dataset.dataset import MagnetSimulationDataset
from src.magnet.models.unet import UNet
from typing import List, Any, Dict

from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from lightning_utilities.core.rank_zero import rank_zero_only

import wandb


def gather_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Gather predictions from a list of dictionaries into a single dictionary.

    Args:
        predictions (List[Dict[str, Any]]): List of dictionaries containing predictions.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing all predictions.
    """
    out = {}
    for key in predictions[0].keys():
        out[key] = np.concatenate([p[key] for p in predictions], axis=0)
    return out


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add all the fields to queue
    for field in cfg:
        queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)
    

@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3.2")
def main(cfg: DictConfig):
    
    print_config_tree(cfg, save_to_file=False)

    train_dl = instantiate(cfg.dataset)
    
    model = instantiate(cfg.model)

    trainer = instantiate(cfg.trainer)
    # resume training if possible
    predictions = trainer.predict(model, train_dl, ckpt_path="last")

    predictions = gather_predictions(predictions)
    
    #efield
    mse_full_efield = predictions['mse_full'][:,0:6]
    mse_subject_efield = predictions['mse_subject'][:,0:6]
    mse_space_efield = predictions['mse_space'][:,0:6]
    mean_full_efield = np.mean(mse_full_efield)
    mean_subject_efield = np.mean(mse_subject_efield)
    mean_space_efield = np.mean(mse_space_efield)

    #hfield
    mse_full_hfield = predictions['mse_full'][:,6:12]
    mse_subject_hfield = predictions['mse_subject'][:,6:12]
    mse_space_hfield = predictions['mse_space'][:,6:12]
    mean_full_hfield = np.mean(mse_full_hfield)
    mean_subject_hfield = np.mean(mse_subject_hfield)
    mean_space_hfield = np.mean(mse_space_hfield)
    mean = {'mean_full_efield': mean_full_efield, 'mean_subject_efield': mean_subject_efield, 'mean_space_efield': mean_space_efield}
    #mse = [mse_full, mse_subject, mse_space]
    wandb.log({'mean_full_efield': mean_full_efield, 'mean_subject_efield': mean_subject_efield, 'mean_space_efield': mean_space_efield,
               'mean_full_hfield': mean_full_hfield, 'mean_subject_hfield': mean_subject_hfield, 'mean_space_hfield': mean_space_hfield})
    from src.magnet.plots import plot_all_predictions

    pred_efield = predictions['pred'][:,0:6]*300
    pred_hfield = predictions['pred'][:,6:12]*1
    target_efield = predictions['target'][:,0:6]*300
    target_hfield = predictions['target'][:,6:12]*1

    plot_all_predictions(predictions= pred_efield, targets=target_efield, path=f"figures/predictions/{cfg.experiment_name}/efield")
    plot_all_predictions(predictions=pred_hfield, targets=target_hfield, path=f"figures/predictions/{cfg.experiment_name}/hfield")
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()