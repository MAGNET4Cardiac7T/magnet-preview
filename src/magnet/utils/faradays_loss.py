import torch
import math

class FaradaysLoss(torch.nn.Module):
    def __init__(self, dx : float = 1, dy : float = 1, dz : float = 1, theoretical: bool=True, omega: float = 1):
        super(FaradaysLoss,self).__init__()
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.theoretical = theoretical
        self.omega = omega
    

    def forward(self,pred,target):
        omega = self.omega
        theoretical = self.theoretical
        pred_efield_real = pred[:,0:3,:,:,:]
        pred_efield_imag = pred[:,3:6,:,:,:]
        pred_hfield_real = pred[:,6:9,:,:,:]
        pred_hfield_imag = pred[:,9:12,:,:,:]
        pred_hfield_complex = pred_hfield_real + 1j* pred_hfield_imag
        conv_layer = torch.nn.Conv3d(3, 3, kernel_size=3, padding=0, bias=False)
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        conv_layer.weight.data.zero_()
        conv_layer.weight.data[0,2,1,0,1] = -1/(2*self.dy)
        conv_layer.weight.data[0,2,1,2,1] = 1/(2*self.dy)
        conv_layer.weight.data[0,1,0,1,1] = 1/(2*self.dz)
        conv_layer.weight.data[0,1,2,1,1] = -1/(2*self.dz)

        conv_layer.weight.data[1,0,0,1,1] = -1/(2*self.dz)
        conv_layer.weight.data[1,0,2,1,1] = 1/(2*self.dz)
        conv_layer.weight.data[1,2,1,1,0] = 1/(2*self.dx)
        conv_layer.weight.data[1,2,1,1,2] = -1/(2*self.dx)

        conv_layer.weight.data[2,1,1,1,0] = -1/(2*self.dx)
        conv_layer.weight.data[2,1,1,1,2] = 1/(2*self.dx)
        conv_layer.weight.data[2,0,1,0,1] = 1/(2*self.dy)
        conv_layer.weight.data[2,0,1,2,1] = -1/(2*self.dy)

        conv_layer = conv_layer.type(Tensor)
        

        if (theoretical == True):
            
            curls_hat = conv_layer(pred_efield_real) + 1j * conv_layer(pred_efield_imag)
            curls_hat = torch.nn.functional.pad(curls_hat,[1,1,1,1,1,1])
            
            faradays_law_hat = curls_hat - 1j* omega * pred_hfield_complex
            faradays = abs(faradays_law_hat)**2


            
        else:
            target_efield_real = target[:,0:3,:,:,:]
            target_efield_imag = target[:,3:6,:,:,:]
            target_hfield_complex = target[:,6:9,:,:,:] + 1j* target[:,9:12,:,:,:]


            curls_hat = conv_layer(pred_efield_real) + 1j * conv_layer(pred_efield_imag)
            curls = conv_layer(target_efield_real) + 1j * conv_layer(target_efield_imag)

            curls_hat = torch.nn.functional.pad(curls_hat,[1,1,1,1,1,1])            
            curls = torch.nn.functional.pad(curls,[1,1,1,1,1,1])
            

            faradays_law_hat = curls_hat - 1j* omega * pred_hfield_complex
            faradays_law = curls - 1j* omega * target_hfield_complex

            mse_faradays = abs(faradays_law-faradays_law_hat)**2

            faradays = mse_faradays

        return faradays





        


