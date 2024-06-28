import numpy as np
import os
from typing import List

def generate_header(field_type:str = 'E'):
    f = field_type
    return f"x [mm]\ty [mm]\tz [mm]\t{f}xRe [V/m]\t{f}yRe [V/m]\t{f}zRe [V/m]\t{f}xIm [V/m]\t{f}yIm [V/m]\t{f}zIm [V/m]" + "\n" + "-"*153


def unsqueeze_field(field: np.ndarray):
    components = [np.reshape(component, (-1,), order='F') for component in field]
    reshaped_array = np.stack(components, axis=-1)
    return reshaped_array


def save_prediction_field(batched_field: np.ndarray, coil_id_list: List[int], simulation_name_list: List[str], root_path: str = "output/predictions/"):
    area  = np.meshgrid(np.arange(-240, 244, 4), np.arange(-150, 154, 4), np.arange(-230, 154, 4), indexing="ij")
    single_coordinates = [np.reshape(component, (-1,), order='F') for component in area]
    joined_coordinates = np.stack(single_coordinates, axis=-1)

    np.savetxt("test.txt", joined_coordinates, fmt='%d')


    for field, coil_id, simulation in zip(batched_field, coil_id_list, simulation_name_list):
        location_sorted_field = unsqueeze_field(field)
        file_name = os.path.join(root_path, simulation, f"Efield_[AC{coil_id+1}].txt")
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        field_with_coordinate = np.concatenate([joined_coordinates, location_sorted_field], axis=1)
        header = generate_header()
        np.savetxt(file_name, field_with_coordinate, fmt=['%4d']*3+['%13.8f']*6, header=header, comments='', delimiter=' ')