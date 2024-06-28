import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

def plot_one_prediction(E_abs_true: np.ndarray, E_abs_pred: np.ndarray, slice_num: int=50, axis: int = 2, fig_save_path: str = "predictions.png"):
    if axis == 0:
        pred_slice = E_abs_pred[slice_num,:,:]
        true_slice = E_abs_true[slice_num,:,:]
    elif axis == 1:
        pred_slice = E_abs_pred[:,slice_num,:]
        true_slice = E_abs_true[:,slice_num,:]
    else:
        pred_slice = E_abs_pred[:,:,slice_num]
        true_slice = E_abs_true[:,:,slice_num]
    error_slice = true_slice - pred_slice
    error_slice_abs = np.sqrt(error_slice**2+1e-3)
    #error_slice = np.ones_like(error_slice)

    # generate graphic
    fig, ax = plt.subplots(1,3, dpi=300, figsize=(10,4))
    c1 = ax[0].imshow(pred_slice, norm=LogNorm())
    fig.colorbar(c1,fraction=0.046, pad=0.04, label="E field (||E||²)")
    ax[0].set_title("Prediction")
    c2 = ax[1].imshow(true_slice, norm=LogNorm())
    fig.colorbar(c2,fraction=0.046, pad=0.04, label="E field (||E||²)")
    ax[1].set_title("Ground truth")
    c3 = ax[2].imshow(error_slice)
    fig.colorbar(c3,fraction=0.046, pad=0.04, label="Prediction error")
    ax[2].set_title("Error")
    fig.tight_layout(pad=0.5)
    fig.savefig(fig_save_path,bbox_inches="tight")
    plt.close()


def plot_all_predictions(predictions, targets, path: str = "figures/varia"):
    if os.path.exists(path) and not os.path.isdir(path):
        raise NotADirectoryError(f"The provided path {path} is not a directory!")
    
    if not os.path.exists(path):
        os.makedirs(path)

    
    for i, (pred, true) in enumerate(zip(predictions, targets)):

        E_abs_true = np.sum(true**2, axis=0)
        E_abs_pred = np.sum(pred**2, axis=0)
        figname = os.path.join(path, f"{i}.png")
        plot_one_prediction(E_abs_true, E_abs_pred, fig_save_path=figname)