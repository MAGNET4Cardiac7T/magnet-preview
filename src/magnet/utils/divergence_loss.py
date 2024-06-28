import torch

class DivergenceLoss(torch.nn.Module):
    def __init__(self, dx : float = 1, dy : float = 1, dz : float = 1, theoretical: bool=True):
        super(DivergenceLoss, self).__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theoretical = theoretical

    def forward(self,pred, target):
        theoretical = self.theoretical
        
        conv_layer = torch.nn.Conv3d(3, 1, kernel_size=3, padding=0, bias=False)
        # Set the weights of the convolutional layer for centered finite difference
        conv_layer.weight.data.zero_()
        conv_layer.weight.data[:, 0, 1, 1, 0] = -0.5
        conv_layer.weight.data[:, 1, 1, 0, 1] = -0.5
        conv_layer.weight.data[:, 2, 0, 1, 1] = -0.5
        conv_layer.weight.data[:, 0, 1, 1, 2] = 0.5
        conv_layer.weight.data[:, 1, 1, 2, 1] = 0.5
        conv_layer.weight.data[:, 2, 2, 1, 1] = 0.5
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        conv_layer = conv_layer.type(Tensor)

        pred_real = pred[:,0:3,:,:,:]
        pred_imag = pred[:,3:6,:,:,:]
        div_hat_real = conv_layer(pred_real)
        div_hat_imag = conv_layer(pred_imag)
        div_hat = div_hat_real + div_hat_imag

        div_hat = torch.nn.functional.pad(div_hat,[1,1,1,1,1,1])

        # Apply the convolution
        #grad_tensor = torch.rand(pred.shape)
        #grad_tensor = torch.zeros(pred.shape)
        divergence = 0

        if (theoretical == True):    
            divergence = (div_hat)**2
        else:
            target_real = target[:,0:3,:,:,:]
            target_imag = target[:,3:6,:,:,:]
            div_real = conv_layer(target_real)
            div_imag = conv_layer(target_imag)
            div = div_real + div_imag
            div = torch.nn.functional.pad(div,[1,1,1,1,1,1])

            mse_div = (div-div_hat)**2
            
            divergence= mse_div

        #alternative for loss would be:
        #mse_div = torch.sum(div_hat)
        return divergence
        