from magnet.dataset.dataset import MagnetSimulationDataset


ds = MagnetSimulationDataset(
    simulations_path="data/processed_h5/train"
)

item = ds[0]

input = item['input']

import matplotlib.pyplot as plt

plt.imshow(input[0,:,:,50])
plt.show()