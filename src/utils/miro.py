import torch
import torch.nn as nn
import torch.nn.functional as F
class MeanEncoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x
class VarianceEncoder(nn.Module):
    def __init__(self, shape, init=0.1, channelwise=False, eps=1e-5):
        super().__init__()
        self.shape = shape
        self.eps = eps
        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = shape
        if channelwise:
            if len(shape) == 4:
                b_shape = (1, shape[1], 1, 1)
            elif len(shape ) == 3:
                b_shape = (1, 1, shape[2])
            else:
                raise ValueError()
        self.b = nn.Parameter(torch.full(b_shape, init))
    def forward(self, x):
        return F.softplus(self.b) + self.eps
class MIRO(nn.Module):
    def __init__(self,shape):
        super(MIRO, self).__init__()
        self.mean_encoders = MeanEncoder(shape)
        self.var_encoders = VarianceEncoder(shape)
    def update(self, pre_f, f):
        reg_loss = 0.
        mean_enc = self.mean_encoders
        var_enc = self.var_encoders
        mean = mean_enc(pre_f)
        var = var_enc(pre_f)
        vlb = (mean - f).pow(2).div(var) + var.log()
        reg_loss += vlb.mean() / 2.
        return reg_loss,var