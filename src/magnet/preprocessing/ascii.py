import trimesh
from trimesh.voxel.creation import local_voxelize

from typing import List, Dict

import pandas as pd
import numpy as np

import fnmatch
import os
import io
import json



def crop_voxel_mesh(voxel_mesh, dims=(121, 76, 96)):
    # crop shape
    low_x = 100-dims[0]//2
    high_x = low_x+dims[0]

    low_y = 100-dims[1]//2
    high_y = low_y+dims[1]

    low_z = 100-dims[2]//2
    high_z = low_z+dims[2]

    return voxel_mesh[low_x:high_x,low_y:high_y,low_z:high_z]*1.0

def read_field_from_one_dipole(fname: str, fixed_order: List[str] = ['x', 'y', 'z', 'ExRe', 'EyRe', 'EzRe', 'ExIm', 'EyIm', 'EzIm']):
    header = np.loadtxt(fname, max_rows=1, dtype=str)
    header = list(header[::2])
    

    data=np.loadtxt(fname,skiprows=2)
    # reorder the columns
    order = [header.index(col) for col in fixed_order]
    data = data[:, order]

    matrix_shape=[0,0,0]
    for i in range(3):
        matrix_shape[i]=np.size(np.unique(data[:,i]))
    
    # Extract Ex,Ey,Ez as complex 
    Ex=data[:,3]+1j*data[:,6] 
    Ey=data[:,4]+1j*data[:,7] 
    Ez=data[:,5]+1j*data[:,8]
    
    E3d = np.empty(matrix_shape+[3], dtype=np.complex128)
    
    # Reshape to 3D
    E3d[:,:,:,0]=np.reshape(Ex,matrix_shape,order='F') # 
    E3d[:,:,:,1]=np.reshape(Ey,matrix_shape,order='F') # 
    E3d[:,:,:,2]=np.reshape(Ez,matrix_shape,order='F') # 
    
    #E3d = E3d.transpose((0,3,2,1))

    return E3d


def read_field(root_path: str, num_dipoles: int = 8, field_type='E'):
    file_list = os.listdir(root_path)
    if field_type == 'E':
        fixed_order = ['x', 'y', 'z', 'ExRe', 'EyRe', 'EzRe', 'ExIm', 'EyIm', 'EzIm']
    else:
        fixed_order = ['x', 'y', 'z', 'HxRe', 'HyRe', 'HzRe', 'HxIm', 'HyIm', 'HzIm']
    # read field components from each dipole
    fields = []
    for i in range(1,num_dipoles+1):
        fname = fnmatch.filter(file_list, f"*AC{i}*")[0]
        field = read_field_from_one_dipole(os.path.join(root_path, fname), fixed_order=fixed_order)
        fields.append(field)

    return np.stack(fields, axis=-1)

def read_dipoles_voxels(dipole_path_processed: str):
    materials_dipoles = pd.read_csv(os.path.join(dipole_path_processed, "materials.txt"))
    voxels_dipoles = [np.load(os.path.join(dipole_path_processed, f)) for f in materials_dipoles["file"]]
    return voxels_dipoles, materials_dipoles

def voxelize_subject_stl(subject_path_raw: str):
    materials_subject = pd.read_csv(os.path.join(subject_path_raw, "input", "materials.txt"))

    meshes_subject = [trimesh.load_mesh(os.path.join(subject_path_raw, "input", f)) for f in materials_subject["file"]]
    voxels_subject = [local_voxelize(mesh, (0,0,0), 4, 100).matrix for mesh in meshes_subject]
    voxels_subject = [crop_voxel_mesh(voxel) for voxel in voxels_subject]
    return voxels_subject, materials_subject

def calculate_material_voxel_properties(voxels_dipoles: List[np.ndarray], 
                                        voxels_subject: List[np.ndarray], 
                                        materials_dipoles: pd.DataFrame, 
                                        materials_subject: pd.DataFrame,
                                        materials_air: Dict[str, float],
                                        feature_names: List[str]):
    
    material_features = []

    for feature_name in feature_names:
        # bit sketchy, find better way to merge voxels
        voxels_feature = sum([voxels_dipoles[i]*feature for i, feature in enumerate(materials_dipoles[feature_name])]) + \
                         sum([voxels_subject[i]*feature for i, feature in enumerate(materials_subject[feature_name])])
        
        air_voxels = 1 - sum(voxels_dipoles+voxels_subject)
        voxels_feature += air_voxels*materials_air[feature_name]

        material_features.append(voxels_feature)

    return material_features

def calculate_input_features(dipole_path_processed: str, 
                             subject_path_raw: str, 
                             materials_air: Dict[str, float],
                             feature_names: List[str]):
    voxels_dipoles, materials_dipoles = read_dipoles_voxels(dipole_path_processed)
    voxels_subject, materials_subject = voxelize_subject_stl(subject_path_raw)

    voxels_shape = [sum(voxels_dipoles+voxels_subject)]
    material_features = calculate_material_voxel_properties(voxels_dipoles, 
                                                            voxels_subject, 
                                                            materials_dipoles, 
                                                            materials_subject, 
                                                            materials_air,
                                                            feature_names)

    input_features = np.stack(voxels_shape + material_features)
    return input_features, voxels_subject

def preprocess_simulation(
        simulation_name: str,
        raw_path: str = "data/raw",
        dipole_path_processed: str = "data/dipoles/simple/processed",
        feature_names: List[str] = ["conductivity", "permittivity", "density"],
        materials_air: Dict[str, float] = {"conductivity": 0.0, "permittivity": 1.0006, "density": 1.293}
    ):

    raw_path_simulation = os.path.join(raw_path, simulation_name)

    # generate specific feature matrices
    input_features, subject_mask = calculate_input_features(dipole_path_processed, 
                                                            raw_path_simulation, 
                                                            materials_air,
                                                            feature_names)
    
    subject_mask = np.minimum(np.sum(subject_mask, axis=0), 1)


    # read efields
    Efields = read_field(os.path.join(raw_path_simulation, "E_field"), field_type='E')


    # read hfields
    Hfields = read_field(os.path.join(raw_path_simulation, "H_field"), field_type='H')


    sample = {
        'input': input_features,
        'subject': subject_mask,
        'efield': Efields,
        'hfield': Hfields,
    }

    return sample

    # create dir if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # save input features
    bytesio_object = io.BytesIO()
    np.save(bytesio_object, input_features)
    with open(os.path.join(output_path, f"{simulation_name}.input"), "wb") as f:
        f.write(bytesio_object.getbuffer())

    # save subject mask
    bytesio_object = io.BytesIO()
    np.save(bytesio_object, subject_mask)
    with open(os.path.join(output_path, f"{simulation_name}.subject"), "wb") as f:
        f.write(bytesio_object.getbuffer())

    # save Efield
    bytesio_object = io.BytesIO()
    np.save(bytesio_object, Efields)
    with open(os.path.join(output_path, f"{simulation_name}.efield"), "wb") as f:
        f.write(bytesio_object.getbuffer())
    # save Hfield
    bytesio_object = io.BytesIO()
    np.save(bytesio_object, Hfields)
    with open(os.path.join(output_path, f"{simulation_name}.hfield"), "wb") as f:
        f.write(bytesio_object.getbuffer())


if __name__ == "__main__":
    sample = preprocess_simulation("Sphere_r_12cm")
    print(sample["efield"])
    
    import h5py

    with h5py.File("test.h5", 'w') as f:
        f.create_dataset("input", data=sample["input"])
        f.create_dataset("efield", data=sample["efield"])