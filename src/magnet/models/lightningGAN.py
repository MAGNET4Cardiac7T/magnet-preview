import pytorch_lightning as pl
import torch
import numpy as np
import wandb
from .vox2vox import *
from .unet import *
from magnet.utils.dice_loss import *
from torch.autograd import Variable
from magnet.utils.divergence_loss import *
from magnet.utils.faradays_loss import *
from magnet.utils.zero_loss import *


class LightningModelGAN(pl.LightningModule):
    def __init__(self, generator: UNet(6,12), discriminator: Discriminator(6),img_height: int,
                 img_width: int,img_depth: int, alpha: float = 1, beta: float = 1,
                 glr: float = 0.0002, dlr: float = 0.0002, b1: float = 0.5, b2: float = 0.999,
                 d_threshold: int = 8,lambda_voxel: float = 10000, lambda_efield: float = 1,
                 lambda_hfield: float = 1, lambda_physics: float = 100, 
                 physics_loss: torch.nn.Module= ZeroLoss) -> None:
        super().__init__()
        #self.save_hyperparameters()
        self.generator = generator
        self.discriminator = discriminator
        self.alpha = alpha
        self.beta = beta #weight for subject
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.glr = glr
        self.dlr = dlr
        self.b1 = b1
        self.b2 = b2
        self.d_threshold = d_threshold
        self.lambda_voxel = lambda_voxel
        self.lambda_efield = lambda_efield
        self.lambda_hfield = lambda_hfield
        self.lambda_physics = lambda_physics
        self.physics_loss = physics_loss
        self.automatic_optimization = False

    # ich weiß nicht, ob das so richtig ist
    def forward(self, x):
       return self.generator(x)

    def training_step(self, batch, batch_idx):
        input, coil, y_efield,y_hfield, subject_mask = batch['input'], batch['coils_real'], batch['efield'], batch['hfield'], batch['subject']
        x = torch.cat([input,coil], dim=1)
        y = torch.cat([y_efield,y_hfield],dim=1)
        #y_hat = self(x)
        subject_mask = torch.unsqueeze(subject_mask,1)
        lambda_voxel = self.lambda_voxel
        lambda_physics = self.lambda_physics
         # Loss functions
        criterion_GAN = torch.nn.MSELoss()

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, self.img_height // 2 ** 4, self.img_width // 2 ** 4, self.img_depth // 2 ** 4)
        
        optimizer_G, optimizer_D = self.optimizers()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        discriminator_update = 'False'
        # Model inputs
        real_A = Variable(x.type(Tensor)) #vorher: real_A = Variable(batch["A"].unsqueeze_(1).type(Tensor))
        real_B = Variable(y.type(Tensor)) #vorher: real_B = Variable(batch["B"].unsqueeze_(1).type(Tensor))
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)


        # ---------------------
        #  Train Discriminator, only update every disc_update batches
        # ---------------------
        # Real loss
        fake_B = self.generator(real_A)
        pred_real = self.discriminator(real_B, real_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = self.discriminator(fake_B.detach(), real_A)
        loss_fake = criterion_GAN(pred_fake, fake)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
        d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
        d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

        if d_total_acu <= self.d_threshold:
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            discriminator_update = 'True'

        # ------------------
        #  Train Generators
        # ------------------
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        # GAN loss
        fake_B = self.generator(real_A)
        pred_fake = self.discriminator(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Voxel-wise loss
        mse_voxel = (fake_B-real_B)**2 #old: criterion_voxelwise(fake_B, real_B)
        y_hat = fake_B
        y_hat_efield = y_hat[:,0:6]
        y_hat_hfield = y_hat[:,6:12]
        mse_voxel_efield = (y_hat_efield-y_efield)**2
        mse_voxel_hfield = (y_hat_hfield-y_hfield)**2
        mse_physics = self.physics_loss(y_hat,y)
        
        loss_voxel_efield = torch.mean((subject_mask*(self.beta)*mse_voxel_efield + (1-subject_mask)*self.alpha*mse_voxel_efield))
        loss_voxel_hfield = torch.mean((subject_mask*(self.beta)*mse_voxel_hfield + (1-subject_mask)*self.alpha*mse_voxel_hfield))
        loss_voxel = self.lambda_efield * loss_voxel_efield + self.lambda_hfield * loss_voxel_hfield
        loss_physics = torch.mean((subject_mask*(self.beta)*mse_physics))
        # Total loss
        loss_G = loss_GAN + lambda_voxel * loss_voxel + lambda_physics * loss_physics
        loss_G.backward()

        optimizer_G.step()

        self.log("loss", loss_G, prog_bar=True)
        self.log("loss_voxel", loss_voxel, prog_bar=True)
        self.log("loss_physics", loss_physics, prog_bar=True)
        wandb.log({"loss": loss_G, "loss_voxel": loss_voxel,"loss_physics":loss_physics})
        return loss_G
    

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
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.glr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.dlr, betas=(self.b1, self.b2))
        return optimizer_G,optimizer_D


    '''
    # if necessary we could add parameters
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=2, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="mri", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--glr", type=float, default=0.0002, help="adam: generator learning rate")
    parser.add_argument("--dlr", type=float, default=0.0002, help="adam: discriminator learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=128, help="size of image height")
    parser.add_argument("--img_width", type=int, default=128, help="size of image width")
    parser.add_argument("--img_depth", type=int, default=128, help="size of image depth")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--disc_update", type=int, default=5, help="only update discriminator every n iter")
    parser.add_argument("--d_threshold", type=int, default=.8, help="discriminator threshold")
    parser.add_argument("--threshold", type=int, default=-1, help="threshold during sampling, -1: No thresholding")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="interval between sampling of images from generators"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
    
    '''