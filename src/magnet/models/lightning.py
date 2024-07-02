import pytorch_lightning as pl
import torch
from magnet.utils.divergence_loss import *
from magnet.utils.zero_loss import *
from magnet.utils.faradays_loss import *

class LightningModel(pl.LightningModule):
    def __init__(self, net: torch.nn.Module, alpha: float = 1, beta: float = 1, lambda_efield: float = 1,
                 lambda_hfield: float = 1, lambda_physics: float = 100,
                 physics_loss: torch.nn.Module = ZeroLoss) -> None:
        super().__init__()
        self.net = net
        self.alpha = alpha # weight of the space for calculating loss
        self.beta = beta # weight of the subject for calculating loss
        self.lambda_efield = lambda_efield
        self.lambda_hfield = lambda_hfield
        self.lambda_physics = lambda_physics
        self.physics_loss = physics_loss
        

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        input, coil, y_efield,y_hfield, subject_mask = batch['input'], batch['coils_real'], batch['efield'], batch['hfield'], batch['subject']
        x = torch.cat([input,coil], dim=1)
        y = torch.cat([y_efield,y_hfield],dim=1)
        y_hat = self(x)
        y_hat_efield = y_hat[:,0:6,:,:,:]
        y_hat_hfield = y_hat[:,6:12,:,:,:]
        
        # define loss functions
        #criterion_physics = self.physics_loss()
        #criterion_voxel = torch.nn.MSELoss()
        
        mse_physics = self.physics_loss(y_hat,y)
        #mse_voxel = criterion_voxel(y_hat, y)
        mse_voxel_efield = (y_hat_efield-y_efield)**2
        mse_voxel_hfield = (y_hat_hfield-y_hfield)**2
        subject_mask = subject_mask.unsqueeze(1)

        #loss_voxel = torch.mean((subject_mask*(self.beta)*mse_voxel + (1-subject_mask)*self.alpha*mse_voxel))
        loss_voxel_efield = torch.mean((subject_mask*(self.beta)*mse_voxel_efield + (1-subject_mask)*self.alpha*mse_voxel_efield))
        loss_voxel_hfield = torch.mean((subject_mask*(self.beta)*mse_voxel_hfield + (1-subject_mask)*self.alpha*mse_voxel_hfield))
        loss_voxel = self.lambda_efield * loss_voxel_efield + self.lambda_hfield * loss_voxel_hfield

        loss_physics = torch.mean(subject_mask*(self.beta)*mse_physics)
        loss = loss_voxel + self.lambda_physics * loss_physics
        self.log("loss", loss, prog_bar=True)
        self.log("loss_voxel", loss_voxel, prog_bar=True)
        self.log("loss_voxel_efield", loss_voxel_efield, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input, coil, y_efield,y_hfield, subject_mask = batch['input'], batch['coils_real'], batch['efield'], batch['hfield'], batch['subject']
        x = torch.cat([input,coil], dim=1)
        y = torch.cat([y_efield,y_hfield],dim=1)
        #y_hat = torch.zeros_like(y)
        y_hat = self(x)
        y_hat_efield = y_hat[:,0:6,:,:,:]
        y_hat_hfield = y_hat[:,6:12,:,:,:]
        
        mse_efield = ((y_efield - y_hat_efield)**2) * (300**2)
        mse_hfield = ((y_hfield - y_hat_hfield)**2) * (1**2)

        subject_mask = subject_mask.unsqueeze(1)
        mse_efield_full = torch.mean(mse_efield)
        mse_efield_subject = torch.mean(torch.sum(mse_efield*subject_mask, axis=(0,2,3,4))/torch.sum(subject_mask, axis=(0,2,3,4)))
        mse_efield_space = torch.mean(torch.sum(mse_efield*(1-subject_mask), axis=(0,2,3,4))/torch.sum(1-subject_mask, axis=(0,2,3,4)))

        mse_hfield_full = torch.mean(mse_hfield)
        mse_hfield_subject = torch.mean(torch.sum(mse_hfield*subject_mask, axis=(0,2,3,4))/torch.sum(subject_mask, axis=(0,2,3,4)))
        mse_hfield_space = torch.mean(torch.sum(mse_hfield*(1-subject_mask), axis=(0,2,3,4))/torch.sum(1-subject_mask, axis=(0,2,3,4)))


        self.log("mse_efield_full", mse_efield_full, prog_bar=True)
        self.log("mse_efield_subject", mse_efield_subject, prog_bar=True)
        self.log("mse_efield_space", mse_efield_space, prog_bar=True)

        self.log("mse_hfield_full", mse_hfield_full, prog_bar=True)
        self.log("mse_hfield_subject", mse_hfield_subject, prog_bar=True)
        self.log("mse_hfield_space", mse_hfield_space, prog_bar=True)

        return 1


    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input, coil, y_efield,y_hfield, subject_mask = batch['input'], batch['coils_real'], batch['efield'], batch['hfield'], batch['subject']
        x = torch.cat([input,coil], dim=1)
        y = torch.cat([y_efield,y_hfield],dim=1)
        y_hat = self(x)

        
        mse = ((y - y_hat)**2) * (300**2)
        
        subject_mask = subject_mask.unsqueeze(1)
        mse_full = torch.mean(mse, dim=(-1,-2,-3))
        mse_subject = torch.sum(mse*subject_mask, dim=(-1,-2,-3))/torch.sum(subject_mask, dim=(-1,-2,-3))
        mse_space = torch.sum(mse*(1-subject_mask), dim=(-1,-2,-3))/torch.sum(1-subject_mask, dim=(-1,-2,-3))
        

        pred_dict = {
            'simulation': batch['simulation'],
            'pred': y_hat.detach().cpu().numpy(),
            'target': y.detach().cpu().numpy(),
            'coil_id': batch['coil_id'].detach().cpu().numpy(),
            'mse_full': mse_full.detach().cpu().numpy(),
            'mse_subject': mse_subject.detach().cpu().numpy(), 
            'mse_space': mse_space.detach().cpu().numpy(),
        }
        return pred_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)