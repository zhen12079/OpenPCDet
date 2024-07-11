import numpy as np
import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib, tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor
quant_desc_input=QuantDescriptor(calib_method='histogram')
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.TensorQuantizer.use_fb_fake_quant=True

class BaseBEVBackbone_qat(nn.Module):
    def __init__(self, model_cfg, input_channels):
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
        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        self.res_backbone = self.model_cfg.get('res_backbone',False)
        self.with_quantization = self.model_cfg.WITH_QUANTIZATION
        # self.res_quantizer = quant_nn.TensorQuantizer(tensor_quant.QUANT_DESC_8BIT_PER_TENSOR)
        for idx in range(num_levels):
            if self.with_quantization:
                cur_layers = [
                    # nn.ZeroPad2d(1),
                    quant_nn.QuantConv2d(
                        c_in_list[idx], num_filters[idx], kernel_size=3,
                        stride=layer_strides[idx], padding=1, bias=False,
                        # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                        # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR

                    ),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ]
            else:
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
                    if self.with_quantization:
                        cur_layers.extend([
                            quant_nn.QuantConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False,
                                                 # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                 # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                 ),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            quant_nn.QuantConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False,
                                                 # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                 # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                 ),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01)
                        ])
                    else:
                        cur_layers.extend([
                            quant_nn.QuantConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False,
                                                 # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                 # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                 ),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            quant_nn.QuantConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False,
                                                 # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                 # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                 ),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01)
                        ])
                else:
                    if self.with_quantization:
                        cur_layers.extend([
                            quant_nn.QuantConv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False,
                                                 # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                 # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                 ),
                            nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
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
                    if self.with_quantization:
                        self.deblocks.append(nn.Sequential(
                            quant_nn.QuantConvTranspose2d(
                                num_filters[idx], num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx], bias=False,
                                # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))
                    else:
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
                    if self.with_quantization:
                        self.deblocks.append(nn.Sequential(
                            quant_nn.QuantConv2d(
                                num_filters[idx], num_upsample_filters[idx],
                                stride,
                                stride=stride, bias=False,
                                # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU()
                        ))
                    else:
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
            if self.with_quantization:
                self.deblocks.append(nn.Sequential(
                    quant_nn.QuantConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False,
                                                  # quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
                                                  # quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR
                                                  ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ))
            else:
                self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                ))

        self.num_bev_features = c_in

        self.with_quantization = self.model_cfg.get('WITH_QUANTIZATION', False)
        # if self.with_quantization: # 量化
        #     self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            #import pdb;pdb.set_trace()
            # if self.res_backbone:
            #     x = self.blocks[i][:4](x)
            #     for mm in range(self.model_cfg.LAYER_NUMS[i]):
            #         identity = x
            #         out = self.blocks[i][4+mm*5:4+(mm+1)*5](x)
            #         if self.with_quantization: # 量化
            #             x = self.res_quantizer(identity) + out
            #             # x = x + out
            #         else:
            #             x = x + out
            # else:
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
