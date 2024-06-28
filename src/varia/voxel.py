import trimesh
from trimesh.voxel.creation import local_voxelize
from trimesh.util import concatenate

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd


PATH = "data/dipoles/simple/"
DIMS = (121, 76, 96)

# load a mesh
mesh = trimesh.load_mesh('data/raw/Sphere_r_6cm/input/sphere_r_6cm.stl')

# load dipoles
materials = pd.read_csv(os.path.join(PATH, 'raw', "materials.txt"))
dipole_files = materials['file']

#mask = [0, 1, 1, 0, 1, 1, 0, 1]
mask = [1]*8
dipoles = [trimesh.load_mesh(os.path.join(PATH, 'raw', f)) for idx, f in enumerate(dipole_files) if mask[idx]]


merged_mesh = trimesh.util.concatenate(dipoles)
merged_voxels = local_voxelize(merged_mesh, (0,0,0), 4, 100)
dipoles_voxels = [local_voxelize(d, (0,0,0), 4, 100) for d in dipoles]


# color_cube = np.zeros_like(merged_voxels.matrix, dtype=int)
# color_cube = np.stack([color_cube]*4, axis=-1)
# colors = [[201,58,81,255], [103,160,113, 255], [249,132,45,255], [93,205,189, 255], [250,205,98,255]]
# for d, c in zip(dipoles_voxels, colors):
#     coil_color = (np.zeros_like(color_cube)+np.array(c).reshape((1,1,1,4)))*np.expand_dims(d.matrix, -1)
#     color_cube += coil_color

# colored_voxel = merged_voxels.as_boxes(colors=color_cube)

colored_voxel = merged_voxels.as_boxes()

merged_mesh.show()

s = trimesh.Scene()
s.add_geometry(colored_voxel)
s.show()
