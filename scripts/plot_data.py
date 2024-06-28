import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os

def plot_input_vs_field(E_abs: np.ndarray, input: np.ndarray, slice_num: int=50, axis: int = 2, fig_save_path: str = "predictions.png"):
    if axis == 0:
        field_slice = E_abs[slice_num,:,:]
        input_slice = input[slice_num,:,:]
    elif axis == 1:
        field_slice = E_abs[:,slice_num,:]
        input_slice = input[:,slice_num,:]
    else:
        field_slice = E_abs[:,:,slice_num]
        input_slice = input[:,:,slice_num]

    # generate graphic
    fig, ax = plt.subplots(1,2, dpi=300, figsize=(7,4))
    c1 = ax[0].imshow(field_slice, norm=LogNorm())
    fig.colorbar(c1,fraction=0.046, pad=0.04, label="E field (||E||²)")
    ax[0].set_title("Field")
    c2 = ax[1].imshow(input_slice)
    fig.colorbar(c2,fraction=0.046, pad=0.04, label="E field (||E||²)")
    ax[1].set_title("input")
    fig.tight_layout(pad=0.5)
    fig.savefig(fig_save_path,bbox_inches="tight")
    plt.show()
    plt.close()

def plot_input(input: np.ndarray, slice_num: int=50, axis: int = 2, fig_save_path: str = "predictions.png"):
    if axis == 0:
        input_slice = input[slice_num,:,:]
    elif axis == 1:
        input_slice = input[:,slice_num,:]
    else:
        input_slice = input[:,:,slice_num]

    # generate graphic
    fig = plt.figure(dpi=300, figsize=(7,4))
    c2 = plt.imshow(input_slice)
    plt.title("input")
    fig.tight_layout(pad=0.5)
    fig.savefig(fig_save_path,bbox_inches="tight")
    plt.show()
    plt.close()

if __name__ == "__main__":
    input = np.load("data/processed/train/Sphere_r_6_shifted_x_40_y_0_z_0.input")
    field = np.load("data/processed/train/Sphere_r_12cm_pred.efield")
    #input = np.load("data/processed/train/Sphere_r_12cm.input")
    #field = np.load("data/processed/train/Sphere_r_12cm.efield")

    Exyz=np.sum(field,4)  # combined fields Ex, Ey, Ez 
    field_abs=np.abs(np.sum(Exyz*Exyz,3))  # Ex^2+Ey^2+Ez^2  
    input_shape = input[3]

    #plot_input_vs_field(field_abs, input_shape, fig_save_path="figures/data/shift_1.png")
    plot_input(input_shape, fig_save_path="figures/data/shift_1.png")