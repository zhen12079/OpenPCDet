import sys

import numpy as np
import torch
import torch.nn as nn
from .backbone_cspdarknet53 import ResblockBody,BasicConv

class BaseBEVBackbone_cspbased(nn.Module):
    def __init__(self, model_cfg, input_channels=64):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []
        #import pdb;pdb.set_trace()
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.conv1 = BasicConv(input_channels, input_channels, kernel_size=3, stride=1)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        # self.res_backbone = self.model_cfg.get('res_backbone',False)

        for idx in range(num_levels):
            if idx==0:
                cur_layers = [ResblockBody(c_in_list[idx], num_filters[idx], 1, first=True, stride=layer_strides[idx])]
            else:
                cur_layers = [ResblockBody(c_in_list[idx], num_filters[idx], 1, first=False,
                                           stride=layer_strides[idx])]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    ResblockBody(num_filters[idx], num_filters[idx], layer_nums[idx], first=False, stride=1),
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']#torch.Size([4, 64, 496, 432])
        # print("input",spatial_features.shape)
        if isinstance(data_dict, dict):
            x = data_dict['spatial_features']
        else:
            x = data_dict
        ups = []
        ret_dict = {}
        # x = spatial_features
        x = self.conv1(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))#torch.Size([4, 128, 248, 216])
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        # data_dict['spatial_features_2d'] = x#torch.Size([4, 384, 248, 216])
        if isinstance(data_dict, dict):
            data_dict['spatial_features_2d'] = x
        else:
            data_dict = x
        return data_dict

if __name__ == "__main__":
    sys.path.append("/userdata/31289/object_detect")
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
    cfg_from_yaml_file("/userdata/31289/object_detect/tools/cfgs/leap_models/baseline_polyloss_fx.yaml", cfg)

    net = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D)
    from print_model_stat import print_model_stat
    data = torch.randn(1,64, 496, 432)
    print_model_stat(net,data)
    res = net(data)
    print(res.shape)
