
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm

from ..common.conv_module import ConvModule
from ..common.base_module import BaseModule
from ..common.inverted_residual import InvertedResidual

class CRU(nn.Module):
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 3/4, # 2/3
                 group_size:int = 2,
                 group_kernel_size:int = 5,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        if self.up_channel % 2 != 0:
            self.up_channel = up_channel = self.up_channel + 1
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel, kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel, kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Sequential(nn.Conv2d(up_channel, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size),
                                            nn.ReLU(),
                                            nn.Conv2d(op_channel, op_channel, 1, 1, 0)
                                            )
        self.PWC1           = nn.Sequential(nn.Conv2d(up_channel, op_channel, 3, 1, 1, groups=1, bias=False), # nn.Conv2d(up_channel, op_channel,kernel_size=1, bias=False)
                                            nn.BatchNorm2d(op_channel),
                                            nn.ReLU(),
                                            nn.Conv2d(op_channel, op_channel, 3, 1, 1, groups=1, bias=False)
                                           )                                  
        #low
        self.PWC2           = nn.Conv2d(low_channel, op_channel-low_channel, kernel_size=1, bias=False)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        
        return Y1, Y2

class GCBlock4(nn.Module):
    def __init__(self, inplane, ouplane, stride=1):
        super(GCBlock4, self).__init__()
        
        self.inplane = inplane
        self.stride=stride
        self.scale =  49 ** -0.5
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_s = nn.Sequential(nn.Conv2d(inplane, inplane, 3, 1, 1, groups=1, bias=False),
                                    nn.Conv2d(inplane, inplane, 3, 1, 1, groups=1, bias=False),
                                    nn.Conv2d(inplane, inplane, 3, 1, 1, groups=1, bias=False)
                                   )
                                     
        self.conv_q = nn.Sequential(nn.Conv2d(inplane, inplane, 1, 1, 0, groups=inplane, bias=False), # inplane
                                    nn.Sigmoid())
        self.conv_k = nn.Sequential(nn.Conv2d(inplane, inplane, 1, 1, 0, groups=inplane, bias=False), # inplane
                                    nn.Sigmoid())                            
        self.conv_v = nn.Conv2d(inplane, inplane, 1, 1, 0, groups=inplane, bias=False)  # inplane
        
        self.spatial_v13 = CRU(inplane) 
                                        
        self.spatial_v2 = nn.Sequential(nn.Conv2d(inplane, inplane//4, 1, 1, 0, groups=1, bias=False),
                                       nn.BatchNorm2d(inplane//4),
                                       nn.ReLU(),
                                       nn.Conv2d(inplane//4, inplane, 1, 1, 0, groups=1, bias=False)
                                       )
                          
        self.conv_mlp = nn.Sequential(nn.Conv2d(inplane, ouplane, 1, 1, 0, groups=1, bias=False),
                                      nn.BatchNorm2d(ouplane),
                                      nn.ReLU()
                                      )  
                         
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)
            
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
      

    def forward(self, x):
    
        if self.stride > 1:
            x = self.pool(x)
        
        bs, c, h, w = x.size() 
        v = self.conv_v(x)
        
        v_s = F.interpolate(v, (20, 20), mode='bilinear', align_corners=False)
        v_spatial1, v_spatial3 = self.spatial_v13(v_s) # self.spatial_v1(v_s)
        # v_spatial3 = self.spatial_v3(v_s)
        v_spatial2 = self.spatial_v2(self.avg_pool(v))
        v_spatial2_pos = (torch.sigmoid(v_spatial2) > 0.5).float()
        v_spatial2_neg = (torch.sigmoid(v_spatial2) < 0.5).float()
        v_spatial_con = v_spatial1 * v_spatial2_pos + v_spatial3 * v_spatial2_neg
        v_spatial_ori = F.interpolate(v_spatial_con, (h, w), mode='bilinear', align_corners=False)
        v_spatial = torch.sigmoid(v_spatial_ori)

                                
        y = self.conv_k(x) * v
        y = self.conv_s(y) 
        y = y * self.scale * v_spatial
                       
        q = self.conv_q(x)
        
        y = q * y
        
        y = self.conv_mlp(y)
        
        if self.stride > 1:
            x = self.shortcut(x)
                    
        y = y + x
            
        return y   

class PiDiNet(BaseModule):
    """PiDiNet backbone.

    Args:
        arch (str): Architecture of mobilnetv3, from {small, large}.
            Default: small.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        out_indices (None or Sequence[int]): Output from which stages.
            Default: None, which means output tensors from final stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed.
            Default: False.
    """
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'tiny':  [[20, 20, 1, 3],
                  [20, 40, 2, 1],
                  [40, 40, 1, 3],
                  [40, 80, 2, 1],
                  [80, 80, 1, 3],
                  [80, 80, 2, 1],
                  [80, 80, 1, 3],
                  ],
        'small': [[32, 32, 1, 3],
                  [32, 64, 2, 1],
                  [64, 64, 1, 3],
                  [64, 128, 2, 1],
                  [128, 128, 1, 3],
                  [128, 128, 2, 1],
                  [128, 128, 1, 3],
                  ],
        'normal': [[60, 60, 1, 3],
                  [60, 120, 2, 1],
                  [120, 120, 1, 3],
                  [120, 240, 2, 1],
                  [240, 240, 1, 3],
                  [240, 240, 2, 1],
                  [240, 240, 1, 3],
                  ]
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
                 out_indices=None,
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='Kaiming',
                         layer=['Conv2d'],
                         nonlinearity='leaky_relu'),
                     dict(type='Normal', layer=['Linear'], std=0.01),
                     dict(type='Constant', layer=['BatchNorm2d'], val=1)
                 ]):
        super(PiDiNet, self).__init__(init_cfg)
        assert arch in self.arch_settings
        if out_indices is None:
            out_indices = (15, ) if arch == 'small' else (16, )
        """    
        for order, index in enumerate(out_indices):
            if index not in range(0, len(self.arch_settings[arch]) + 2):
                raise ValueError(
                    'the item in out_indices must in '
                    f'range(0, {len(self.arch_settings[arch]) + 2}). '
                    f'But received {index}')
        """
        if frozen_stages not in range(-1, len(self.arch_settings[arch]) + 2):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch]) + 2}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.layers = self._make_layer()
        self.feat_dim = self.arch_settings[arch][-1][1]

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        if self.arch == 'tiny':
            in_channels = 20
        elif self.arch == 'small':
            in_channels = 32
        elif self.arch == 'normal':
            in_channels = 60

        layer = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.add_module('layer0', layer)
        layers.append('layer0')

        counter = 0
        for i, params in enumerate(layer_setting):
            in_channels, out_channels, stride, repeat_num= params
            for j in range(repeat_num):
                layer = GCBlock4(in_channels, out_channels, stride)
                layer_name = 'layer{}'.format(counter + j + 1)
                self.add_module(layer_name, layer)
                layers.append(layer_name)
            counter += repeat_num

        return layers

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(PiDiNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        #dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor'))
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
    
    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
