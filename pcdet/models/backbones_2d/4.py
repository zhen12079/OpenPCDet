import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
import numpy as np
import torch.onnx
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, model_cfg=None, input_channels=64):
        super(ResNet50, self).__init__()
        self.model_cfg = model_cfg
        self.base_model = models.resnet50()
        self.base_model.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=64,kernel_size=7,stride=1,padding=3,bias=False)
        # print(len([i for i in self.base_model.children()]))
        # self.base_model = [i for i in self.base_model_resnet50.children()][:-3]
        # self.base_model = nn.Sequential(*self.base_model)
        self.base_model.layer4 = nn.Sequential()
        self.base_model.avgpool = nn.Sequential()
        self.base_model.fc = nn.Sequential()
        self.deblocks = nn.ModuleList()
        upsample_strides = [1, 2, 4]
        num_filters = [256, 512, 1024]
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
        if isinstance(data_dict, dict):
            x = data_dict['spatial_features']
        else:
            x = data_dict
        # x = data_dict['spatial_features']
        # x = data_dict
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)
        x = self.base_model.layer1(x)
        out1 = x
        x = self.base_model.layer2(x)
        out2 = x
        x = self.base_model.layer3(x)
        out3 = x
        # x = base_model.layer4(x)
        x = torch.cat([self.deblocks[0](out1), self.deblocks[1](out2), self.deblocks[2](out3)], dim=1)
        if isinstance(data_dict, dict):
            data_dict['spatial_features_2d'] = x
        else:
            data_dict = x
        # data_dict['spatial_features_2d'] = x
        return data_dict
# print(self.base_model.features[0][1].weight.data.shape)
# self.base_model.features[0][1].weight.data.fill_(1)
# self.base_model.features[0][1].bias.data.zero_()
if __name__ == "__main__":
    # base_model = models.resnet50()
    # print(base_model)
    from print_model_stat import print_model_stat
    data = torch.randn(1,64, 496, 432)
    net = ResNet50()
    print_model_stat(net,data)


    # out1 = net(data)
    # print(out1.shape)
    # print(out2.shape)
    # print(out3.shape)
    # base_model.conv1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=7,stride=1,padding=3,bias=False)
    # print(base_model.children())
    # x = data
    # x = base_model.conv1(x)
    # x = base_model.bn1(x)
    # x = base_model.relu(x)
    # x = base_model.maxpool(x)
    # x = base_model.layer1(x)
    # print(x.shape)
    # x = base_model.layer2(x)
    # print(x.shape)
    # x = base_model.layer3(x)
    # print(x.shape)
    # x = base_model.layer4(x)
    # print(x.shape)

    # for i in range(8):
    #     data = self.base_model.features[i](data)
    # print(data.shape)
    # for i in range(8,12):
    #     data = self.base_model.features[i](data)
    # print(data.shape)
    # for i in range(12,15):
    #     data = self.base_model.features[i](data)
    # print(data.shape)

    # torch.onnx.export(net, data, "backbone_resnet50.onnx", verbose=True, input_names=['input'], output_names=['output'])
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
