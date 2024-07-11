'''
Author: Jineng Han
Date: 2023-03-06 13:41:53
LastEditors: Jineng Han
LastEditTime: 2023-03-23 15:50:45
FilePath: /semantic_segmentation/models/layers.py
Description:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# from dropblock import DropBlock2D
from pytorch_quantization.nn.modules import _utils


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 group_conv,
                 dilation=1,
                 dsc=False,
                 deform=False):
        super(DoubleConv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch,
                          out_ch,
                          3,
                          padding=1,
                          dilation=dilation,
                          groups=min(in_ch, out_ch)), nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
        elif dsc:
            conv1 = nn.Sequential(
                nn.Conv2d(in_ch,
                          in_ch,
                          3,
                          padding=1,
                          dilation=dilation,
                          groups=in_ch), nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
            conv2 = nn.Sequential(
                nn.Conv2d(out_ch,
                          out_ch,
                          3,
                          padding=1,
                          dilation=dilation,
                          groups=out_ch), nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
            self.conv = nn.Sequential(*conv1, *conv2)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=dilation),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConvCircular(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self,
                 in_ch,
                 out_ch,
                 group_conv,
                 dilation=1,
                 dsc=False,
                 deform=False):
        super(DoubleConvCircular, self).__init__()
        assert not (group_conv and dsc
                    and deform), 'only one of group conv and dsc can be True'
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch,
                          out_ch,
                          3,
                          padding=(1, 0),
                          dilation=dilation,
                          groups=min(in_ch, out_ch)), nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
        elif dsc:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch,
                          in_ch,
                          3,
                          padding=(1, 0),
                          dilation=dilation,
                          groups=in_ch), nn.BatchNorm2d(in_ch),
                nn.LeakyReLU(inplace=True), nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch,
                          out_ch,
                          3,
                          padding=(1, 0),
                          dilation=dilation,
                          groups=out_ch), nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True), nn.Conv2d(out_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), dilation=dilation),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        # add circular padding
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x


class InConv(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation=1,
                 input_batch_norm=True,
                 circular_padding=False,
                 dsc=False,
                 group_conv=False):
        super(InConv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    DoubleConvCircular(in_ch,
                                       out_ch,
                                       dilation=dilation,
                                       dsc=dsc,
                                       group_conv=group_conv))
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    DoubleConv(in_ch,
                               out_ch,
                               dsc=dsc,
                               group_conv=group_conv,
                               dilation=dilation))
        else:
            if circular_padding:
                self.conv = DoubleConvCircular(in_ch,
                                               out_ch,
                                               dsc=dsc,
                                               group_conv=group_conv,
                                               dilation=dilation)
            else:
                self.conv = DoubleConv(in_ch,
                                       out_ch,
                                       dsc=dsc,
                                       group_conv=group_conv,
                                       dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 dilation,
                 group_conv,
                 circular_padding,
                 dsc=False,
                 deform=False):
        super(Down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConvCircular(in_ch,
                                   out_ch,
                                   group_conv=group_conv,
                                   dilation=dilation,
                                   dsc=dsc,
                                   deform=deform))
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_ch,
                           out_ch,
                           group_conv=group_conv,
                           dilation=dilation,
                           dsc=dsc,
                           deform=deform))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 circular_padding,
                 bilinear=False,
                 group_conv=False,
                 dsc=False,
                 deform=False,
                 use_dropblock=False,
                 drop_p=0.5):
        super().__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch // 2,
                                         in_ch // 2,
                                         2,
                                         stride=2,
                                         groups=in_ch // 2)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        if circular_padding:
            self.conv = DoubleConvCircular(in_ch,
                                           out_ch,
                                           group_conv=group_conv,
                                           dsc=dsc,
                                           deform=deform)
        else:
            self.conv = DoubleConv(in_ch,
                                   out_ch,
                                   group_conv=group_conv,
                                   dsc=dsc,
                                   deform=deform)

        self.use_dropblock = use_dropblock
        if self.use_dropblock:
            # self.dropblock = DropBlock2D(block_size=7, drop_prob=drop_p)
            pass

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.use_dropblock:
            # x = self.dropblock(x)
            pass
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class QuantUpsample(torch.nn.Upsample, _utils.QuantInputMixin):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, **kwargs) -> None:
        super(QuantUpsample, self).__init__(size, scale_factor, mode, align_corners, recompute_scale_factor)
        quant_desc_input = _utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True, **kwargs)
        self.init_quantizer(quant_desc_input)

    def forward(self, input):
        quant_input = self._input_quantizer(input)
        return super(QuantUpsample, self).forward(quant_input)
