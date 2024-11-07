import torch
from torch import nn
import torch.nn.functional as F


 

def get_preproc_model(args, dim_in, dim_out=2):
    #n_out = 2
    # 确保 dim_in 是有效的
    if dim_in <= 0:
        raise ValueError("dim_in must be a positive integer")
    net_preproc = MLPReg_preproc(dim_in, args.n_hidden, dim_out = dim_out)

    return net_preproc



#---------------------------------------------------------------------------
#
#               REGression
#
#---------------------------------------------------------------------------

class MLPReg_preproc(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPReg_preproc, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.sigmoid = nn.LeakyReLU(0.2)
        self.layer_out = nn.Linear(dim_hidden,dim_out)


    def forward(self, x):
        x = self.layer_input(x)
        x = self.sigmoid(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x


