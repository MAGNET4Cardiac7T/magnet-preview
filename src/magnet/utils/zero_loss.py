import torch

class ZeroLoss(torch.nn.Module):
    def init(self, *args, **kwargs):
        super(ZeroLoss, self).init()

    def forward(self,pred, target):
        return torch.zeros_like(pred)