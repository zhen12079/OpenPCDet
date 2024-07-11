import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch.onnx


# Mish激活函数: f(x) = x * tanh(log(1 + e^x))
# pytorch中F.softplus(x)等同于torch.log(1 + torch.exp(x))
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        # x * torch.tanh(torch.log(1 + torch.exp(x)))
        return x * torch.tanh(F.softplus(x))


# 卷积块
# conv + batchnormal + mish
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,groups=1):
        super(BasicConv, self).__init__()
        # yolov4中只有1x1、3x3卷积, 1x1 padding=0, 3x3 padding=1, 卷积操作默认bias=False
        self.groups = groups
        if self.groups==1:
            self.conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=kernel_size // 2,
                                  bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            ineral_channels = self.groups
            self.conv_in = nn.Conv2d(in_channels=in_channels,
                                  out_channels=ineral_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False)
            self.conv_mid = nn.Conv2d(in_channels=ineral_channels,
                                  out_channels=ineral_channels,
                                  kernel_size=kernel_size,
                                  groups=groups,
                                  stride=1,
                                padding=kernel_size // 2,
                                  bias=False)
            self.conv_out = nn.Conv2d(in_channels=ineral_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False)
            self.bn1 = nn.BatchNorm2d(ineral_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = Mish()
        # self.activation = nn.ReLU()
    def forward(self, x):
        if self.groups==1:
            x = self.conv(x)
            x = self.bn(x)
            x = self.activation(x)
        else:
            x = self.conv_in(x)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.conv_mid(x)
            x = self.bn1(x)
            x = self.activation(x)
            x = self.conv_out(x)
            x = self.bn2(x)
            # x = self.activation(x)
        return x


# CSPDarknet结构块的组成部分，内部堆叠的残差块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(in_channels=channels, out_channels=hidden_channels, kernel_size=1),
            BasicConv(in_channels=hidden_channels, out_channels=channels, kernel_size=3),
        )

    def forward(self, x):
        return x + self.block(x)


# CSPNet
# 存在一个残差边，这个残差边绕过了很多残差结构
class ResblockBody(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first,stride):
        super(ResblockBody, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=stride)

        if first:
            self.split_conv0 = BasicConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            self.split_conv1 = BasicConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels // 2),
                BasicConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
            )

            self.concat_conv = BasicConv(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1)


        else:
            self.split_conv0 = BasicConv(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1)
            self.split_conv1 = BasicConv(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(channels=out_channels // 2) for _ in range(num_blocks)],
                BasicConv(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel_size=1)
            )

            self.concat_conv = BasicConv(in_channels=out_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


# CSPDarknet
class LeapCspNet(nn.Module):
    def __init__(self, model_cfg=None, input_channels=64):
        super(LeapCspNet, self).__init__()
        self.model_cfg = model_cfg
        if hasattr(model_cfg,"LAYER_NUMS"):
            self.layers = model_cfg.LAYER_NUMS
        else:
            self.layers = [1, 2, 2, 2]
        self.deblocks = nn.ModuleList()
        self.inplanes = 64
        self.conv1 = BasicConv(input_channels, self.inplanes, kernel_size=3, stride=1)
        self.feature_channels = [64, 64, 96, 128]
        upsample_strides = [1,2,4,8]
        num_filters = [64,64,96,128]
        num_upsample_filters = [96,96,96,96]
        for idx in range(len(upsample_strides)):
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[idx], num_upsample_filters[idx],
                    upsample_strides[idx],
                    stride=upsample_strides[idx], bias=False
                ),
                nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))
        self.stages = nn.ModuleList([
            ResblockBody(self.inplanes, self.feature_channels[0], self.layers[0], first=True,stride=2),
            ResblockBody(self.feature_channels[0], self.feature_channels[1], self.layers[1], first=False,stride=2),
            ResblockBody(self.feature_channels[1], self.feature_channels[2], self.layers[2], first=False,stride=2),
            ResblockBody(self.feature_channels[2], self.feature_channels[3], self.layers[3], first=False,stride=2)
        ])

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, data_dict):
        if isinstance(data_dict, dict):
            x = data_dict['spatial_features']
        else:
            x = data_dict
        x = self.conv1(x)

        out1 = self.stages[0](x)
        out2 = self.stages[1](out1)
        out3 = self.stages[2](out2)
        out4 = self.stages[3](out3)


        x = torch.cat([self.deblocks[0](out1),self.deblocks[1](out2),self.deblocks[2](out3),self.deblocks[3](out4)], dim=1)
        if isinstance(data_dict, dict):
            data_dict['spatial_features_2d'] = x
        else:
            data_dict = x
        return data_dict



if __name__ == "__main__":
    net = LeapCspNet()
    from print_model_stat import print_model_stat
    data = torch.randn(1,64, 496, 432)
    print_model_stat(net,data)
    res = net(data)
    print(res.shape)
    # model = load_model(net, './weights/CSPDarknet53.pth')
    # net.eval()
    # print(net)
    # data = torch.randn(4, 64, 496, 432)
    # data_dict = {}
    # data_dict['spatial_features'] = data
    # data_dict_result = net(data_dict)
    # print(data_dict_result["spatial_features_2d"].shape)
    # torch.onnx.export(net, data, "backbone_CSPDarknet53.onnx", verbose=True, input_names=['input'], output_names=['output'])
    # print(data_dict_result)

