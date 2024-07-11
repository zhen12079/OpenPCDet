import copy 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from copy import deepcopy
import cv2
from .losses import lovase_softmax
from .focal_loss import focal_loss


class LaneHead_multitask(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.lane_out_conv = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 7, 1)
        )
        self.lane_aux_conf_predictor = nn.Sequential(
            nn.Conv2d(1024, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2028, 1, 1)
        )
        self.act_sigmoid = nn.Sigmoid()
        self.focal_loss = focal_loss
        self.forward_ret_dict = {}

    def label_formatting(self, raw_label):
        num_of_labels = len(raw_label)
        w = raw_label.shape[1]
        h = raw_label.shape[2]
        label_tensor = np.zeros((num_of_labels, 2, w, h), dtype=np.longlong)

        for k in range(num_of_labels):
            label_temp = np.zeros((w, h, 2), dtype=np.longlong)
            label_data = raw_label[k]

            for i in rangr(w):
                for j in range(h):
                    y_idx = w -i -1
                    x_idx = h -j -1

                    line_num = int(label_data[i][j])
                    if line_num == 255:
                        label_temp[y_idx][x_idx][1] = 0
                        label_temp[y_idx][x_idx][0] = 6
                    else:
                        label_temp[y_idx][x_idx][1] = 1
                        label_temp[y_idx][x_idx][0] =line_num
            label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))
        return (torch.tensor(label_tensor))


    def loss(self, out, batch):
        lanes_label = batch
        lanes_label = self.label_formatting(lanes_label)

        y_pred_cls = out[:, 0:7, :, :]
        y_pred_conf = out[:, 7, :, :]

        y_label_cls = lanes_label[:, 0, :, :].cuda()
        y_label_conf = lanes_label[:, 1, :, :].cuda()

        cls_loss = 0
        cls_loss += nn.CrossEntropyLoss()(y_pred_cls, y_label_cls)

        # y_pred_cls = y_pred_cls.permute(0, 2, 3, 1).reshape(-1, 7).contiguous()
        # y_label_cls = y_label_cls.reshape(-1, 1).contiguous()
        # cls_loss = self.focal_loss(y_pres_cls, y_label_cls) * 30

        # print(cls_loss.upu().item())
        # cls_loss += nn.BCELoss()(y_pred_cls, y_label_cls)


        ## dice loss 
        numerator = 2 * torch.sum(torch.mul(y_pred_conf, y_label_conf))
        denominator = torch.sum(torch.square(y_pred_conf)) + torch.sum(torch.square(y_label_conf)) + 1e-6
        dice_coff = numerator / denominator

        conf_loss = 1 - dice_coff

        return conf_loss, cls_loss

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['lane_pred_dicts']
        target_dicts = self.forward_ret_dict['lane_target_dicts']

        conf_loss, cls_loss = self.loss(pred_dicts, target_dicts)

        loss = conf_loss + cls_loss
        loss *= self.model_cfg.LOSS_WEIGHTS
        tb_dict = {}
        tb_dict['lane_conf_loss'] = conf_loss.item()
        tb_dict['lane_cls_loss'] = cls_loss.item()

        return loss, tb_dict

    
    def forward(self, data_dict):
        lane_feature_2d = data_dict['lane_features_2d']
        class_output = self.lane_out_conv(lane_feature_2d)
        conf_output = self.act_sigmoid((self.lane_aux_conf_predictor(lane_feature_2d)))
        out = torch.cat((class_output, conf_output), 1)

        if self.training:
            self.forward_ret_dict['lane_target_dicts'] = data_dict['gt_lane']
            self.forward_ret_dict['lane_pred_dicts'] = out
            data_dict['pred_lane'] = out
        else:
            data_dict['pred_lane'] = out

        return data_dict













import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import copy
from copy import deepcopy
import cv2
from .losses import lovasz_softmax
from .focal_loss import focal_loss

class LaneHead_multitask(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.lane_out_conv =  nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 7, 1)
        )
        self.lane_aux_conf_predictor =  nn.Sequential(
            nn.Conv2d(1024, 2048, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 1, 1)
        )
        self.act_sigmoid = nn.Sigmoid()
        self.focal_loss = focal_loss(alpha=[0.75,0.75,0.75,0.75,0.75,0.75,0.25],gamma=2,size_average=True)
        self.forward_ret_dict = {}
    def label_formatting(self, raw_label):
        # Output image: top-left of the image is farthest-left
        num_of_labels = len(raw_label)
        w = raw_label.shape[1]
        h = raw_label.shape[2]
        label_tensor = np.zeros((num_of_labels, 2, w, h), dtype = np.longlong)

        for k in range(num_of_labels):
            label_temp = np.zeros((w,h,2), dtype = np.longlong)
            label_data = raw_label[k]

            for i in range(w):
                for j in range(h):

                    y_idx = w - i - 1
                    x_idx = h - j - 1
                    # y_idx = i
                    # x_idx = j

                    line_num = int(label_data[i][j])
                    if line_num == 255:
                        label_temp[y_idx][x_idx][1] = 0
                        # classification
                        label_temp[y_idx][x_idx][0] = 6
                    else: 
                        # confidence
                        label_temp[y_idx][x_idx][1] = 1
                        # classification
                        label_temp[y_idx][x_idx][0] = line_num

            label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))

        return(torch.tensor(label_tensor))
    def loss(self, out, batch):
        lanes_label = batch
        lanes_label = self.label_formatting(lanes_label) #channel0 = line number, channel1 = confidence
        
        y_pred_cls = out[:, 0:7, :, :]
        y_pred_conf = out[:, 7, :, :]

        y_label_cls = lanes_label[:, 0, :, :].cuda()
        y_label_conf = lanes_label[:, 1, :, :].cuda()

        # print(y_pred_cls.shape)
        # print(y_label_cls.shape)

        # print(y_pred_cls.min(),y_pred_cls.max())
        # print(y_label_cls.min(),y_label_cls.max())


        cls_loss = 0
        cls_loss += nn.CrossEntropyLoss()(y_pred_cls, y_label_cls)

        # y_pred_cls = y_pred_cls.permute(0,2,3,1).reshape(-1,7).contiguous()
        # y_label_cls = y_label_cls.reshape(-1,1).contiguous()
        # cls_loss = self.focal_loss(y_pred_cls,y_label_cls)*30

        # print(cls_loss.cpu().item())

        # cls_loss += nn.BCELoss()(y_pred_cls, y_label_cls)

        ## Dice Loss ###
        numerator = 2 * torch.sum(torch.mul(y_pred_conf, y_label_conf))
        denominator = torch.sum(torch.square(y_pred_conf)) + torch.sum(torch.square(y_label_conf)) + 1e-6
        dice_coeff = numerator / denominator

        # print("F1:",dice_coeff.cpu().item())

        conf_loss = (1 - dice_coeff)

        # loss = conf_loss + cls_loss

        # ret = {'loss': loss, 'loss_stats': {'conf': conf_loss, 'cls': cls_loss}}

        return conf_loss, cls_loss
    def get_loss(self):
        pred_dicts = self.forward_ret_dict['lane_pred_dicts']
        target_dicts = self.forward_ret_dict['lane_target_dicts']

        conf_loss, cls_loss = self.loss(pred_dicts,target_dicts)

        loss = conf_loss + cls_loss
        loss *= self.model_cfg.LOSS_WEIGHTS
        tb_dict = {}
        tb_dict['lane_conf_loss'] = conf_loss.item()
        tb_dict['lane_cls_loss'] = cls_loss.item()
        return loss, tb_dict


    def forward(self, data_dict):

        lane_features_2d = data_dict['lane_features_2d']
        class_output = self.lane_out_conv(lane_features_2d)
        conf_output = self.act_sigmoid((self.lane_aux_conf_predictor(lane_features_2d)))
        out = torch.cat((class_output, conf_output), 1)

        if self.training:
            self.forward_ret_dict['lane_target_dicts'] = data_dict['gt_lane']
            self.forward_ret_dict['lane_pred_dicts'] = out
        else:
            data_dict['pred_lane'] = out


        return data_dict

