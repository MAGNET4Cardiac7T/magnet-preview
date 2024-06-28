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

materials = pd.read_csv(os.path.join(PATH, 'raw', "materials.txt"))
dipole_files = materials['file']

meshes = [trimesh.load_mesh(os.path.join(PATH, 'raw', f)) for f in dipole_files]
voxelized_meshes = [local_voxelize(mesh, (0,0,0), 4, 100) for mesh in meshes]


processed_path = os.path.join(PATH, 'processed')
if not os.path.exists(processed_path):
    os.makedirs(processed_path)




for i, voxel_mesh in enumerate(voxelized_meshes):
    shape = voxel_mesh.matrix
    # crop shape
    low_x = 100-DIMS[0]//2
    high_x = low_x+DIMS[0]

    low_y = 100-DIMS[1]//2
    high_y = low_y+DIMS[1]

    low_z = 100-DIMS[2]//2
    high_z = low_z+DIMS[2]

    shape = shape[low_x:high_x,low_y:high_y,low_z:high_z]

    # save to output path
    filename = dipole_files[i][:-4]+'.npy'
    out_path = os.path.join(processed_path, filename)

    np.save(out_path, shape)

    # update processed materials
    materials.loc[i, 'file'] = filename

# save new materials file
materials.to_csv(os.path.join(processed_path, "materials.txt"), index=False)