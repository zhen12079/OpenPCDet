import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch.onnx
from torchvision import models

class MobileNetV2(nn.Module):
    def __init__(self, model_cfg=None, input_channels=64):
        super(MobileNetV2, self).__init__()
        self.model_cfg = model_cfg
        self.base_model = models.mobilenet_v2()
        # print(self.base_model)
        # data = torch.randn(4, 64, 496, 432)
        self.base_model.features[0][0] = nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        # self.base_model.features[0][1].num_features = 64
        self.base_model.features[0][1] = nn.BatchNorm2d(input_channels, eps=1e-3, momentum=0.01)
        # self.base_model.features[1].conv[0][0].in_channels=64
        self.base_model.features[1].conv[0][0] = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1,groups=32,bias=False)
        self.base_model.features[2].conv[1][0] = nn.Conv2d(in_channels=96,out_channels=96,kernel_size=3,stride=1,padding=1,groups=96,bias=False)
        self.base_model.features[4].conv[1][0] = nn.Conv2d(in_channels=144,out_channels=144,kernel_size=3,stride=1,padding=1,groups=144,bias=False)
        self.base_model.features[11].conv[1][0] = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=2,padding=1,groups=384,bias=False)
        # self.base_model.features[4].conv[2] = nn.Conv2d(in_channels=144,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        # self.base_model.features[4].conv[3] = nn.BatchNorm2d(64, eps=1e-3, momentum=0.01)
        self.base_model.features.__delitem__(18)
        self.base_model.features.__delitem__(17)
        self.base_model.features.__delitem__(16)
        self.base_model.features.__delitem__(15)
        self.base_model = self.base_model.features
        self.deblocks = nn.ModuleList()
        upsample_strides = [1, 2, 4]
        num_filters = [64, 96, 160]
        num_upsample_filters = [128, 128, 128]
        self.num_bev_features = sum(num_upsample_filters)
        for idx in range(len(upsample_strides)):
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
    def forward(self, data_dict):
        x = data_dict['spatial_features']
        for i in range(8):
            x = self.base_model[i](x)
        out1 = x
        for i in range(8, 12):
            x = self.base_model[i](x)
        out2 = x
        for i in range(12, 15):
            x = self.base_model[i](x)
        out3 = x
        x = torch.cat([self.deblocks[0](out1), self.deblocks[1](out2), self.deblocks[2](out3)], dim=1)
        data_dict['spatial_features_2d'] = x
        return data_dict
# print(self.base_model.features[0][1].weight.data.shape)
# self.base_model.features[0][1].weight.data.fill_(1)
# self.base_model.features[0][1].bias.data.zero_()
if __name__ == "__main__":
    net = MobileNetV2()
    # data = torch.randn(4, 64, 496, 432)
    # out1 = net(data)
    print(net)
    # print(out2.shape)
    # print(out3.shape)
    # # print(self.base_model.features[7](data).shape)
    # for i in range(8):
    #     data = self.base_model.features[i](data)
    # print(data.shape)
    # for i in range(8,12):
    #     data = self.base_model.features[i](data)
    # print(data.shape)
    # for i in range(12,15):
    #     data = self.base_model.features[i](data)
    # print(data.shape)

    torch.onnx.export(net, data, "backbone_mobilenet_v2.onnx", verbose=True, input_names=['input'], output_names=['output'])
# if __name__ == "__main__":
#     net = CSPDarknet53([1, 2, 4, 8, 4])
#
#     # model = load_model(net, './weights/CSPDarknet53.pth')
#     net.eval()
#     print(net)
#     # data = torch.randn(4, 64, 496, 432)
#     # data_dict = {}
#     # data_dict['spatial_features'] = data
#     # data_dict_result = net(data_dict)
#     # print(data_dict_result["spatial_features_2d"].shape)
#     # torch.onnx.export(net, data, "backbone_CSPDarknet53.onnx", verbose=True, input_names=['input'], output_names=['output'])
#     # print(data_dict_result)
#
