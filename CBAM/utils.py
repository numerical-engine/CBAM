import torch
import torch.nn.functional as F

def channel_reshape(x:torch.Tensor, dim:int)->torch.Tensor:
    assert x.dim() == 2
    batch_size, channel_size = x.shape
    if dim == 1:
        return x.view(batch_size, channel_size, 1)
    elif dim == 2:
        return x.view(batch_size, channel_size, 1, 1)
    else:
        return x.view(batch_size, channel_size, 1, 1, 1)


def global_avg_pooling(x:torch.Tensor, dim:int)->torch.Tensor:
    if dim == 1:
        return F.avg_pool1d(x, kernel_size = x.shape[2])[:,:,0]
    elif dim == 2:
        return F.avg_pool2d(x, kernel_size = x.shape[2:])[:,:,0,0]
    else:
        return F.avg_pool3d(x, kernel_size = x.shape[2:])[:,:,0,0,0]

def global_max_pooling(x:torch.Tensor, dim:int)->torch.Tensor:
    if dim == 1:
        return F.max_pool1d(x, kernel_size = x.shape[2])[:,:,0]
    elif dim == 2:
        return F.max_pool2d(x, kernel_size = x.shape[2:])[:,:,0,0]
    else:
        return F.max_pool3d(x, kernel_size = x.shape[2:])[:,:,0,0,0]