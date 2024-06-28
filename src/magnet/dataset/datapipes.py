
import numpy as np
import io
import os
import torch
import pandas as pd

from typing import List

from torchdata.datapipes.iter import (
    WebDataset, 
    Mapper, 
    FileLister, 
    FileOpener, 
    TarArchiveLoader,
    ShardingFilter,
    Cycler,
    Shuffler,
    Batcher,
    IterDataPipe,
    Repeater,
    ShardingRoundRobinDispatcher
    )

from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

class NumpyReader(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp
    
    def __iter__(self):
        for file, stream in self.source_dp:
            np_bytes = io.BytesIO(stream.read())
            yield file, np.load(np_bytes)



class NumpyReaderItemwise(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp

    def read_stream(self, stream):
        np_bytes = io.BytesIO(stream.read())
        return np.load(np_bytes).astype(np.float32)

    
    def __iter__(self):
        for item in self.source_dp:
            item['.efield'] = self.read_stream(item['.efield'])
            item['.hfield'] = self.read_stream(item['.hfield'])
            item['.input'] = self.read_stream(item['.input'])
            item['.subject'] = self.read_stream(item['.subject'])
            yield item

class ExhaustiveCoilPhaseGenerator(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, coil_voxel_path: str, num_coils: int = 8) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.coil_voxels_path = coil_voxel_path
        self.num_coils = num_coils

        self.coil_voxels = self.read_coil_voxels()

    def read_coil_voxels(self):
        materials_dipoles = pd.read_csv(os.path.join(self.coil_voxels_path, "materials.txt"))
        voxels = [np.load(os.path.join(self.coil_voxels_path, f)) for f in materials_dipoles["file"]]
        return voxels
    
    def sample_phase_and_mask(self, coil_id: int = 0):
        phase = np.zeros(self.num_coils)
        eye = np.eye(self.num_coils)
        mask = eye[coil_id]

        return phase, mask

    def __iter__(self):
        for item in self.source_dp:
            for coil_id in range(self.num_coils):
                item_new = item.copy()
                phase, mask = self.sample_phase_and_mask(coil_id)
                coefficients = np.exp(phase*1j)*mask
                # process coil

                item_new['.coil_coefficients'] = coefficients
                item_new['.coil_phase'] = phase
                item_new['.coil_mask'] = mask
                item_new['.coil_id'] = coil_id

                item_new['.coils_complex'] = np.dot(np.stack(self.coil_voxels, axis=-1), coefficients)
                item_new['.coils_real'] = np.stack([item_new['.coils_complex'].real, item_new['.coils_complex'].imag])
                yield item_new


class RandomCoilPhaseGenerator(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, coil_voxel_path: str, num_coils: int = 8) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.coil_voxels_path = coil_voxel_path
        self.num_coils = num_coils

        self.coil_voxels = self.read_coil_voxels()

    def read_coil_voxels(self):
        materials_dipoles = pd.read_csv(os.path.join(self.coil_voxels_path, "materials.txt"))
        voxels = [np.load(os.path.join(self.coil_voxels_path, f)) for f in materials_dipoles["file"]]
        return voxels
    
    def sample_phase_and_mask(self):
        phase = np.random.uniform(0, 2*np.pi, self.num_coils)
        mask = np.random.choice([0, 1], self.num_coils, replace=True)
        while np.sum(mask) == 0:
            mask = np.random.choice([0, 1], self.num_coils, replace=True)

        return phase, mask

    def __iter__(self):
        for item in self.source_dp:
            phase, mask = self.sample_phase_and_mask()
            coefficients = np.exp(phase*1j)*mask

            # process coil
            item['.coil_coefficients'] = coefficients
            item['.coil_phase'] = phase
            item['.coil_mask'] = mask

            item['.coils_complex'] = np.dot(np.stack(self.coil_voxels, axis=-1), coefficients)
            item['.coils_real'] = np.stack([item['.coils_complex'].real, item['.coils_complex'].imag])
            yield item


class CoilPhaseShifter(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp

    def __iter__(self):
        for item in self.source_dp:
            coefficients = item['.coil_coefficients']

            # process Efield
            Efield = item['.efield']*coefficients.reshape((1,1,1,1,-1))
            Efield = np.sum(Efield, axis=-1)
            Efield = np.transpose(Efield, axes=[3, 0, 1, 2])
            Efield = np.concatenate([Efield.real, Efield.imag], axis=0)
            item['.efield'] = Efield

            # process hfield
            Hfield = item['.hfield']*coefficients.reshape((1,1,1,1,-1))
            Hfield = np.sum(Hfield, axis=-1)
            Hfield = np.transpose(Hfield, axes=[3, 0, 1, 2])
            Hfield = np.concatenate([Hfield.real, Hfield.imag], axis=0)
            item['.hfield'] = Hfield
            yield item

class Normalizer(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe,
                 efield_params: List[float] = [0, 300],
                 hfield_params: List[float] = [0, 1],
                 input_params: List[float] = [[0, 0, 1, 1.293], [1, 5.e+7, 5.9e+1, 8.9e+3]]) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.efield_params = [np.array(efield_params[0]).reshape(-1,1,1,1), 
                              np.array(efield_params[1]).reshape(-1,1,1,1)]
        self.hfield_params = [np.array(hfield_params[0]).reshape(-1,1,1,1), 
                              np.array(hfield_params[1]).reshape(-1,1,1,1)]
        self.input_params = [np.array(input_params[0]).reshape(-1,1,1,1), 
                             np.array(input_params[1]).reshape(-1,1,1,1)]

    def __iter__(self):
        for item in self.source_dp:

            item['.input'] = (item['.input']-self.input_params[0])/self.input_params[1]
            item['.hfield'] = (item['.hfield']-self.hfield_params[0])/self.hfield_params[1]
            item['.efield'] = (item['.efield']-self.efield_params[0])/self.efield_params[1]
            yield item

class TypeCaster(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe) -> None:
        super().__init__()
        self.source_dp = source_dp
    def __iter__(self):
        for item in self.source_dp:

            # get the name of the simulation
            item['.simulation'] = os.path.basename(item['__key__'])

            # cast to float 32
            item['.coils_real'] = item['.coils_real'].astype(np.float32)
            item['.efield'] = item['.efield'].astype(np.float32)
            item['.hfield'] = item['.hfield'].astype(np.float32)
            item['.input'] = item['.input'].astype(np.float32)
            yield item


# prefetching
# io monitoring tools / profiler
def create_datapipes(
        root_path: str = "data/processed/train",
        coil_voxel_path: str = "data/dipoles/simple/processed",
        phase_samples_per_simulation: int = 10,
):
    datapipe = FileLister(root_path)
    datapipe = FileOpener(datapipe, mode="b")
    datapipe = NumpyReader(datapipe)
    datapipe = WebDataset(datapipe)
    datapipe = RandomCoilPhaseGenerator(datapipe, coil_voxel_path)
    datapipe = Cycler(datapipe, count=phase_samples_per_simulation)
    datapipe = ShardingFilter(datapipe)
    datapipe = ShardingRoundRobinDispatcher(datapipe, sharding_group_filter=SHARDING_PRIORITIES.MULTIPROCESSING)
    datapipe = CoilPhaseShifter(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = TypeCaster(datapipe)
    return datapipe

def create_prediction_datapipes(
        root_path: str = "data/processed/train",
        coil_voxel_path: str = "data/dipoles/simple/processed",
):
    datapipe = FileLister(root_path)
    datapipe = FileOpener(datapipe, mode="b")
    datapipe = NumpyReader(datapipe)
    datapipe = WebDataset(datapipe)
    datapipe = ExhaustiveCoilPhaseGenerator(datapipe, coil_voxel_path, 8)
    datapipe = ShardingFilter(datapipe)
    datapipe = CoilPhaseShifter(datapipe)
    datapipe = Normalizer(datapipe)
    datapipe = TypeCaster(datapipe)
    return datapipe

if __name__ == '__main__':

    # ds = create_datapipes()
    # for item in ds:
    #      field = item['.efield']
    #      #print(item['__key__'])
    #      print(item['.simulation'])


    ds = create_prediction_datapipes()

    from torch.utils.data import DataLoader
    import tqdm
    dl = DataLoader(ds, batch_size=4, num_workers=8)
    for item in tqdm.tqdm(dl):
        field = item['.efield']