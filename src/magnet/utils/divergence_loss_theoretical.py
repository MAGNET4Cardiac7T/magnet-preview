import torch

class DivergenceLossTheoretical(torch.nn.Module):
    def init(self):
        super(DivergenceLossTheoretical, self).init()

    def forward(self,pred, target):
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

        # Apply the convolution
        #grad_tensor = torch.rand(pred.shape)
        #grad_tensor = torch.zeros(pred.shape)
        pred_real = pred[:,0:3,:,:,:]
        pred_imag = pred[:,3:6,:,:,:]
        div_hat_real = conv_layer(pred_real)
        div_hat_imag = conv_layer(pred_imag)
        div_hat = div_hat_real + div_hat_imag

        div_hat = torch.nn.functional.pad(div_hat,[1,1,1,1,1,1])

        #alternative for loss would be:
        #mse_div = torch.sum(div_hat)
        return abs(div_hat)
        