import os
import sys
sys.path.append(os.getcwd())
from src.magnet.preprocessing.ascii import preprocess_simulation
import tqdm
import numpy as np

train_data = [
    # "Ellipse_1",
    # "Ellipse_2",
    # "Ellipse_3",
    # "Ellipse_4",
    # "Ellipse_5",
    # "Ellipse_6",
    # "Ellipse_7",
    # "Ellipse_8",
    # "Ellipse_9",
    # "Ellipse_10",
    # "Ellipse_11",
    # "Ellipse_12",
    "Cylinder_d_20cm",
    # "Sphere_r_6cm_shifted_x_-20_y_0_z_0",
    # "Sphere_r_6cm_shifted_x_-40_y_0_z_0",
    # "Sphere_r_6cm_shifted_x_0_y_-20_z_0",
    # "Sphere_r_6cm_shifted_x_0_y_20_z_0",
    # "Sphere_r_6cm_shifted_x_20_y_0_z_0",
    # "Sphere_r_6cm_shifted_x_40_y_0_z_0",
    # "Sphere_r_6cm",
    # "Sphere_r_10cm",
    # "Sphere_r_12cm",
    # "Sphere_with_y_direction_oriented_copper_rod",
    # "Sphere_r_10cm_epsilon_10_cond_0,79",
    # "Sphere_r_10cm_epsilon_20_cond_0,79",
    # "Sphere_r_10cm_epsilon_90_cond_0,79"
]

import h5py
for simulation in tqdm.tqdm(train_data):
    sample = preprocess_simulation(simulation)
    sample['input'] = sample['input'].astype(np.float32)
    sample['subject'] = sample['subject'].astype(np.float32)
    sample['efield'] = sample['efield'].astype(np.complex64)
    sample['hfield'] = sample['hfield'].astype(np.complex64)



    with h5py.File(f"data/processed_h5/{simulation}.h5", 'w') as f:
        f.create_dataset("input", data=sample["input"])
        f.create_dataset("subject", data=sample["subject"])
        f.create_dataset("efield", data=sample["efield"])
        f.create_dataset("hfield", data=sample["hfield"])
