import torch
from torch.utils.data import Dataset
import os
import h5py

from typing import Any
import glob
import numpy as np
import pandas as pd

        
class MagnetSimulationDataset(torch.utils.data.Dataset):    
    def __init__(self, simulations_path: str = "data/processed_h5/",
                 coils_path: str = "data/dipoles/simple/processed",
                 phase_samples_per_simulation: int = 10,
                 enumerate_coils_individually: bool = False) -> None:
        super().__init__()

        # save parameters
        self.simulations_path = simulations_path
        self.coils_path = coils_path
        self.phase_samples_per_simulation = phase_samples_per_simulation 

        # read data
        self.simulation_list = glob.glob(os.path.join(self.simulations_path, "*.h5"))
        self.simulation_names = [os.path.basename(f)[:-3] for f in self.simulation_list]
        self.coils = self._read_coils()
        self.num_coils = len(self.coils)

        # enumerate coils
        self.enumerate_coils_individually = enumerate_coils_individually
        if enumerate_coils_individually:
            self.phase_samples_per_simulation = self.num_coils
            
        
        # normalization parameters
        efield_params = [0, 300]
        hfield_params = [0, 1]
        input_params = [[0, 0, 1, 1.293], [1, 5.e+7, 5.9e+1, 8.9e+3]]
        

        self.efield_params = [np.array(efield_params[0], dtype=np.float32).reshape(-1,1,1,1), 
                              np.array(efield_params[1], dtype=np.float32).reshape(-1,1,1,1)]
        self.hfield_params = [np.array(hfield_params[0], dtype=np.float32).reshape(-1,1,1,1), 
                              np.array(hfield_params[1], dtype=np.float32).reshape(-1,1,1,1)]
        self.input_params = [np.array(input_params[0], dtype=np.float32).reshape(-1,1,1,1), 
                             np.array(input_params[1], dtype=np.float32).reshape(-1,1,1,1)]

    def __getitem__(self, index) -> Any:
        file_index = index // self.phase_samples_per_simulation
        phase_index = index % self.phase_samples_per_simulation

        with h5py.File(self.simulation_list[file_index]) as f:
            item = {
                'input': f['input'][:],
                'subject': f['subject'][:],
                'efield': f['efield'][:],
                'hfield': f['hfield'][:],
                'simulation': self.simulation_names[file_index],
                'coil_id': phase_index if self.enumerate_coils_individually else -1,
            }
            
        # coil phase:
        if self.enumerate_coils_individually:
            phase, mask = self._get_coil_phase_and_mask(phase_index)
        else:   
            phase, mask = self._sample_phase_and_mask()
        coefficients = np.exp(phase*1j)*mask

        item['coil_coefficients'] = coefficients
        item['coil_phase'] = phase
        item['coil_mask'] = mask

        item['coils_complex'] = np.dot(np.stack(self.coils, axis=-1), coefficients)
        item['coils_real'] = np.stack([item['coils_complex'].real, item['coils_complex'].imag])



        # simulation phase shifter
        # process Efield
        Efield = np.dot(item["efield"], item["coil_coefficients"])
        Efield = np.transpose(Efield, axes=[3, 0, 1, 2])
        Efield = np.concatenate([Efield.real, Efield.imag], axis=0)
        item['efield'] = Efield
        
    
        # process Hfield
        Hfield = np.dot(item["hfield"], item["coil_coefficients"])
        Hfield = np.transpose(Hfield, axes=[3, 0, 1, 2])
        Hfield = np.concatenate([Hfield.real, Hfield.imag], axis=0)
        item['hfield'] = Hfield
        

        # normalize data
        item['input'] = (item['input']-self.input_params[0])/self.input_params[1]
        item['hfield'] = (item['hfield']-self.hfield_params[0])/self.hfield_params[1]
        item['efield'] = (item['efield']-self.efield_params[0])/self.efield_params[1]

        return item
    
    def __len__(self):
        return len(self.simulation_list)*self.phase_samples_per_simulation
    
    def _read_coils(self):
        materials_dipoles = pd.read_csv(os.path.join(self.coils_path, "materials.txt"))
        voxels = [np.load(os.path.join(self.coils_path, f)).astype(np.float32) for f in materials_dipoles["file"]]
        return voxels
    
    def _sample_phase_and_mask(self):
        phase = np.random.uniform(0, 2*np.pi, self.num_coils)
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)

        phase = phase.astype(np.float32)
        mask = mask.astype(np.float32)
        return phase, mask
    
    def _get_coil_phase_and_mask(self, index):
        phase = np.zeros(self.num_coils)
        mask = np.zeros(self.num_coils)
        mask[index] = 1
        
        phase = phase.astype(np.float32)
        mask = mask.astype(np.float32)
        return phase, mask

if __name__ == "__main__":
    ds = MagnetSimulationDataset()
    import tqdm

    for item in tqdm.tqdm(ds, smoothing=0):
        item