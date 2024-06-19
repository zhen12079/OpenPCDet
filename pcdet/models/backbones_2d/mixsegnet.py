import torch 
import torch.nn as nn
import numpy as np
from functools import partial
from einops.layers.torch import Rearrange

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    

class Permute(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.permutu(self.shape)


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim*expansion_factor),
        nn.ReLU(),
        dense(dim*expansion_factor, dim)
    )


def FeedForward_conv2d(dim, expansion_factor=4):
    return nn.Sequential(
        nn.Conv2d(dim, dim*expansion_factor, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(dim*expansion_factor, dim, kernel_size=1)
    )


class MixSegNet(nn.Module):
    def __init__(self, image_size=144, channles=64, patch_size=8, dim=512, depth=5, output_channels=1024, expansion_factor=4, dropout=0., is_with_shared_mlp=False, cfg=None):
        super(MixSegNet, self).__init__()
        self.cfg = cfg
        assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        out_img_size = int(image_size/patch_size)
        out_in_channels = int(dim/(patch_size*patch_size))

        self.mixsegnet = nn.Serquential(
            Rearrange('b c (h p1) (w p2 ex) -> b (p1 p2 c) (h w) ex', p1=patch_size, p2=patch_size, ex=1),
            nn.Conv2d((patch_size**2)*channels, dim, kernel_size=1),
            Permute([0,2,3,1]),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward_conv2d(num_patches, expansion_factor)),
                Permute([0,3,2,1]),
                PreNormResidual(num_patches, FeedForward_conv2d(dim, expansion_factor)),
                Permute([0,3,2,1]),
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Rearrange('b (h w) ex (p1 p2 c) -> b c (h p1) (w p2 ex)', h=out_img_size, p1=patch_size, p2=patch_size, ex=1),
        )
        if is_with_shared_mlp:
            self.is_with_shared_mlp = True
            self.shared_mlp = nn.Conv2d(in_channels=out_in_channels, out_channels=output_channels, kernel_size=1)
        else:
            self.is_with_shared_mlp=False

    
    def forward(self, x, is_get_features=False):
        out = self.mixsegnet(x)
        if self.is_with_shared_mlp:
            out = self.shared_mlp(out)

        if is_get_features:
            list_feature = []
            list_feature.append(x)
            for i in range(len(self.mixsegnet)):
                x = self.mixsegnet[i](x)
                list_feature.append(x)
            return x, list_feature
        return out

