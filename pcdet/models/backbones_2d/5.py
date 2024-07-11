import torch
from torch import nn

from .norm import build_norm_layer
from .base import spconv, SparseConv2d, Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense


class SpRes18(nn.Module):
    def __init__(self,
                 model_cfg, input_channels=64):
        super(SpRes18, self).__init__()
        self.model_cfg = model_cfg

        dense_block = post_act_block_dense
        norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.conv1 = spconv.SparseSequential(
            Sparse2DBasicBlockV(64, 64, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res1"),
            # Sparse2DBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv2d(
                64, 64, 3, 2, padding=1, bias=False
            ),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            # Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            # Sparse2DBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv2d(
                64, 128, 3, 2, padding=1, bias=False
            ),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            # Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            # Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            # Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            # Sparse2DBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv2d(
                128, 256, 3, 2, padding=1, bias=False
            ),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
            # Sparse2DBasicBlock(256, 256, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 256)[1],
            nn.ReLU(),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(256, 256, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256,
        }
        self.backbone_strides = {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
            'x_conv5': 16,
        }
        self.deblocks = nn.ModuleList()
        upsample_strides = [2, 4, 8]
        num_filters = [128, 256, 256]
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
    def forward(self, data_dict):
        x = data_dict['spatial_features']
        x = x.permute(0, 2, 3, 1)
        sp_tensor = spconv.SparseConvTensor.from_dense(x)
        x_conv1 = self.conv1(sp_tensor)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        out1 = x_conv3.dense()#4x
        x_conv4 = self.conv4(x_conv3)
        out2 = x_conv4.dense()#8x
        out3 = self.conv5(out2)#16x
        x = torch.cat([self.deblocks[0](out1), self.deblocks[1](out2), self.deblocks[2](out3)], dim=1)
        data_dict['spatial_features_2d'] = x
        return data_dict

if __name__ == "__main__":
    net = SpRes18().cuda()
    print(net)
    data = torch.randn(1, 64, 496, 432)
    data = data.permute(0,2,3,1).cuda()
    # sp_data = spconv.SparseConvTensor.from_dense(data)
    res = net(data)
    print(res.shape)
    # torch.onnx.export(net, data, "backbone_sp_res18.onnx", verbose=True, input_names=['input'],
    #                   output_names=['output'],opset_version=12)


