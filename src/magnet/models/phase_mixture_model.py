import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
from itertools import product
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset.datapipes import build_datapipes
    
class PhaseModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 4096),
            torch.nn.LeakyReLU()
        )
        self.image = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1, 4, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose3d(4, 4, kernel_size=3, stride=2),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose3d(4, 4, kernel_size=3, stride=2),
            torch.nn.Upsample((121,76,96), mode='trilinear'),
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.image(x.view(-1, 1, 16, 16, 16))
        return x
    
class PredictionModel(torch.nn.Module):
    def __init__(self, in_channels=5, out_channels=6) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 16, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(16, 16, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(16, 16, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(16, 16, kernel_size=3, padding='same'),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(16, out_channels, kernel_size=3, padding='same'),
        )

    def forward(self, x):
        return self.layers(x)

    
class Model(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.phase = PhaseModel()
        self.pred = PredictionModel()

    def forward(self, x, p):
        x_phase = self.phase(p)
        x = torch.cat([x, x_phase], dim=1)
        return self.pred(x)

    def training_step(self, batch, batch_idx):
        x, p, y = batch['.input'], batch['.coil'], batch['.Efield']
        y_hat = self(x, p)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
datapipe = build_datapipes()
train_dl = DataLoader(dataset=datapipe, batch_size=5, num_workers=8)
#model = Model.load_from_checkpoint("output/tb_logs/lightning_logs/version_2/checkpoints/last.ckpt")
model = Model()




model_checkpoint = ModelCheckpoint(save_last=True)
tb_logger = TensorBoardLogger(save_dir="output/tb_logs")
trainer = pl.Trainer(accelerator="gpu", 
                     devices=1, 
                     max_epochs=200, 
                     callbacks=[model_checkpoint],
                     logger=tb_logger
                     )


#trainer.fit(model, train_dl, ckpt_path="output/tb_logs/lightning_logs/version_1/checkpoints/last.ckpt")
trainer.fit(model, train_dl)

# plot results
import matplotlib.pyplot as plt

ds = datapipe
sample = next(iter(datapipe))

x, p, y = sample['.input'], sample['.coil'], sample['.Efield'].numpy()


y_pred = model(torch.stack([x]), torch.stack([p])).detach().numpy()[0]

E_abs_true = np.sum(y**2, axis=0)
E_abs_pred = np.sum(y_pred**2, axis=0)

fig, ax = plt.subplots(1,2)
c1 = ax[0].imshow(E_abs_pred[:,:,50])
fig.colorbar(c1)
c2 = ax[1].imshow(E_abs_true[:,:,50])
fig.colorbar(c2)
plt.show()
