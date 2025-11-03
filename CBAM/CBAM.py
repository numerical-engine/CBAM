import torch
import torch.nn as nn

from CBAM import utils

class CBAM_base(nn.Module):
    def __init__(self, dim:int, in_features:int, clatent_dim:int = None, cact:any = nn.ReLU(), cscale_act:any = nn.Sigmoid(), skernel_size:int|tuple[int] = 7)->None:
        super().__init__()
        if clatent_dim is None: clatent_dim = in_features
        self.dim = dim

        self.fc = nn.ModuleList([nn.Linear(in_features, clatent_dim), cact, nn.Linear(clatent_dim, in_features)])
        if self.dim == 1:
            self.conv = nn.Conv1d(2, 1, skernel_size, padding = "same")
        elif self.dim == 2:
            self.conv = nn.Conv2d(2, 1, skernel_size, padding = "same")
        else:
            self.conv = nn.Conv3d(2, 1, skernel_size, padding = "same")

    def channel_attention(self, x:torch.Tensor)->torch.Tensor:
        s_avg = utils.global_avg_pooling(x, self.dim)
        s_max = utils.global_max_pooling(x, self.dim)

        for layer in self.fc:
            s_avg = layer(s_avg)
            s_max = layer(s_max)
        
        return utils.channel_reshape(s_avg + s_max, self.dim)*x
    
    def spatial_attention(self, x:torch.Tensor)->torch.Tensor:
        s_avg = torch.mean(x, dim = 1, keepdim = True)
        s_max = torch.max(x, dim = 1, keepdim = True)[0]
        s = self.conv(torch.cat((s_avg, s_max), dim = 1))

        return s*x

    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x

class CBAM1d(CBAM_base):
    def __init__(self, in_features:int, clatent_dim:int = None, cact:any = nn.ReLU(), cscale_act:any = nn.Sigmoid())->None:
        super().__init__(1, in_features, clatent_dim, cact, cscale_act)

class CBAM2d(CBAM_base):
    def __init__(self, in_features:int, clatent_dim:int = None, cact:any = nn.ReLU(), cscale_act:any = nn.Sigmoid())->None:
        super().__init__(2, in_features, clatent_dim, cact, cscale_act)

class CBAM3d(CBAM_base):
    def __init__(self, in_features:int, clatent_dim:int = None, cact:any = nn.ReLU(), cscale_act:any = nn.Sigmoid())->None:
        super().__init__(3, in_features, clatent_dim, cact, cscale_act)