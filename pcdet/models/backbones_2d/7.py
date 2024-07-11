import numpy as np
import torch
import torch.nn as nn
from .layers import Down, Up
from .mixsegnet import MixSegNet
from .vit_model import VisionTransformer

class BaseBEVBackbone_multitask(nn.Module):
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
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.res_backbone = self.model_cfg.get('res_backbone',False)
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=1, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                if self.res_backbone:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01)
                    ])
                else:
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
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
################################################################### SEG BACKBONE ##############################################
        if self.model_cfg.get('OTHER_TASKS', False) and "seg" in self.model_cfg.OTHER_TASKS:
            self.segblocks = nn.ModuleList()
            self.segblocks.append(Down(64, 128, dilation=1, group_conv=False, circular_padding=False, dsc=False))
            self.segblocks.append(Down(128, 128, dilation=1, group_conv=False, circular_padding=False, dsc=False))
            self.segblocks.append(Up(256, 64,
                            circular_padding=False,
                            bilinear=False,
                            group_conv=False,
                            dsc=False,
                            use_dropblock=False,
                            drop_p=0.5))
            self.segblocks.append(Up(128, 64,
                            circular_padding=False,
                            bilinear=False,
                            group_conv=False,
                            dsc=False,
                            use_dropblock=False,
                            drop_p=0.5))
            # self.seg_out_conv = nn.Sequential(
            #     nn.Conv2d(64, 64, 3, stride=2, padding=1,bias=False),
            #     nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            # )
            self.segblocks.append(nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=2, padding=1,bias=False),
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.seg_out_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    384, 384,
                    2,
                    stride=2, bias=False
                ),
                nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
                nn.ReLU())
################################################################### LANE BACKBONE ##############################################
        if self.model_cfg.get('OTHER_TASKS', False) and "lane" in self.model_cfg.OTHER_TASKS:

            self.laneblocks = nn.ModuleList()
            self.laneblocks.append(nn.Sequential(
                nn.ConvTranspose2d(64, 128, (2, 1),stride=(2, 1), bias=False),
                nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.laneblocks.append(nn.Sequential(
                nn.Conv2d(128, 64, 3, stride=2, padding=1,bias=False),
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            # self.laneblocks.append(nn.Sequential(
            #     nn.ConvTranspose2d(384, 384, 2,stride=2, bias=False),
            #     nn.BatchNorm2d(384, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            #     nn.Conv2d(384, 256, 3, stride=1, padding=1,bias=False),
            #     nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(256, 128, 2,stride=2, bias=False),
            #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            #     nn.ReLU()
            # ))
            # self.laneblocks.append(nn.Sequential(
            #     nn.ConvTranspose2d(128, 128, (3,1),stride=(3,1), bias=False),
            #     nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            # ))
            # self.laneblocks.append(nn.Sequential(
            #     nn.Conv2d(128, 64, 3, stride=2, padding=1,bias=False),
            #     nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            #     nn.ReLU(),
            # ))

            self.mixsegnet = MixSegNet(
                image_size=144,
                channels=64,
                patch_size=8,
                dim=512,
                depth=1,
                output_channels=1024,
                expansion_factor=4,
                dropout=0.,
                is_with_shared_mlp=True
                )

            # self.vit = VisionTransformer(
            #         img_size=144,
            #         patch_size=8,
            #         in_channels=128,
            #         hidden_size=512,
            #         num_layers=3,
            #         drop_ratio=0.
            #     )
    def forward(self, data_dict):
        if isinstance(data_dict, dict):
            x = data_dict['spatial_features']
        else:
            x = data_dict
################################################################### DETECT BACKBONE ##############################################
        ups = []
        ret_dict = {}
        for i in range(len(self.blocks)):
            if self.res_backbone:
                x = self.blocks[i][:4](x)
                for mm in range(self.model_cfg.LAYER_NUMS[i]):
                    identity = x
                    out = self.blocks[i][4+mm*5:4+(mm+1)*5](x)
                    x = x + out
            else:
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
###############################################################################################################################

################################################################### SEG BACKBONE ##############################################
        if self.model_cfg.get('OTHER_TASKS', False) and "seg" in self.model_cfg.OTHER_TASKS:
            seg_input = data_dict['spatial_features']
            seg_x1=self.segblocks[0](seg_input)
            seg_x2=self.segblocks[1](seg_x1)

            seg_x=self.segblocks[2](seg_x2,seg_x1)
            seg_x=self.segblocks[3](seg_x,seg_input)
            seg_x=self.segblocks[4](seg_x)
            
            data_dict['seg_features_2d'] = seg_x

            seg_out_high=self.seg_out_conv(x)
            data_dict['seg_features_2d_up'] = seg_out_high
###############################################################################################################################

################################################################### LANE BACKBONE ##############################################
        if self.model_cfg.get('OTHER_TASKS', False) and "lane" in self.model_cfg.OTHER_TASKS:
            #独立投影
            # lane_input = data_dict['spatial_features_lane'] #[4, 64, 144, 144]

            #检测低层
            # lane_input = data_dict['spatial_features'] #[b:64,496,432]
            # lane_clip = lane_input[:,:,200:-200,72:-72] #[b:64,96,288] 96 288
            # lane_x = self.laneblocks[0](lane_clip) #288x288
            # lane_x = self.laneblocks[1](lane_x) #144x144

#[0.0, 69.12, -11.52, 11.52, -2.5, 0.5]
#[0.48, 0.16]

            #检测低层
            lane_input = data_dict['spatial_features'] #[b:64,496,432]
            lane_clip = lane_input[:,:,176:-176,0:288] #[b:64,96,288] 144 288
            lane_x = self.laneblocks[0](lane_clip) #288x288
            lane_x = self.laneblocks[1](lane_x) #144x144

            # #检测高层
            # lane_x_high = self.laneblocks[2](x)
            # lane_x_high_clip = lane_x_high[:,:,200:-200,72:-72]
            # lane_x_high_clip = self.laneblocks[3](lane_x_high_clip) #288x288
            # lane_x_high_clip = self.laneblocks[4](lane_x_high_clip) #144x144

            # lane_x = torch.cat((lane_x_high_clip, lane_x), 1)
            lane_out = self.mixsegnet(lane_x)
            data_dict['lane_features_2d'] = lane_out

###############################################################################################################################
        if isinstance(data_dict, dict):
            data_dict['spatial_features_2d'] = x
        else:
            data_dict = x
        return data_dict
if __name__ == "__main__":
    import sys
    sys.path.append("/userdata/31289/object_detect")
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
    cfg_from_yaml_file("/userdata/31289/object_detect/tools/cfgs/leap_models/baseline_polyloss_fx.yaml", cfg)

    net = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D)
    from print_model_stat import print_model_stat
    data = torch.randn(1,64, 496, 432)
    print_model_stat(net,data)
    res = net(data)
    print(res.shape)
