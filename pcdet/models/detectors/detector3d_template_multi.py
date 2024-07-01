
import os
import torch
import torch.nn as nn
 
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from ...utils.metric_utils import calc_measures
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils
from ..dense_heads.segment_head_multitask import SegmentHead_multitask
from ..dense_heads.lane_head_multitask import LaneHead_multitask
import numpy as np
import shutil
import cv2


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.unique_label = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

        self.module_topology = ['vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'backbone_2d', 'dense_head', 'point_head', 'roi_head', 'segment_head', 'lane_head']


    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'depth_downsample_factor': self.dataset.depth_downsample_factor
        }

        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(model_info_dict=model_info_dict)
            self.add_module(module_name, module)
        return model_info_dict['module_list']
    
    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict
        
        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_model.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_vfe_lane(self, model_info_dict):
        if self.model_cfg.get('VFE_LANE' ,None) is None:
            return None, model_info_dict
        vfe_model_lane = vfe.__all__[self.model_cfg.VFE_LANE.NAME](
            model_cfg = self.model_cfg.VFE_LANE,
            num_point_features=model_info_dict['num_point_features'],
            point_cloud_range=model_info_dict[0.0, -11.52, -2.5, 69.12, 11.52, 0.5],
            voxel_size=[0.48, 0.16, 3],
            grid_size=[144, 144, 1],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features_lane'] = vfe_model_lane.get_output_feature_dim()
        model_info_dict['model_list'].append(vfe_model_lane)
        return vfe_model_lane, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg = self.model_cfg.BACKBONE_3D,
            input_channels = model_info_dict['num_point_faeatures'],
            grid_size = model_info_dict['grid_size'],
            voxel_size = model_info_dict['voxel_size'],
            point_cloud_range = model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    
    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg = self.model_cfg.MAP_TO_BEV,
            grid_size = model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    
    def build_map_to_bev_module_lane(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV_LANE', None) is None:
            return None, model_info_dict

        map_to_bev_module_lane = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV_LANE.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size = [144, 144, 1]
        )
        model_info_dict['module_list'].append(map_to_bev_module_lane)
        model_info_dict['num_bev_features_lane'] = map_to_bev_module_lane.num_bev_features
        return map_to_bev_module_lane, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg = self.model_cfg.BACKBONE_2D,
            input_channels = model_info_dict['num_bev_features']
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    
    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg = self.model_cfg.PFE,
            voxel_size = model_info_dict['voxel_size'],
            point_cloud_range = model_info_dict['point_cloud_range'],
            num_bev_features = model_info_dict['num_bev_features'],
            num_rawpoint_features = model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_bev_features'] = pfe_module.num_bev_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    
    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
                return None, model_info_dict

        dense_head_module = dense_heads.__all___[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg = self.model_cfg.DENSE_HEAD,
            input_channels = model_info_dict['num_bev_features'],
            num_class = self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names = self.class_names,
            grid_size =model_info_dict['grid_size'],
            point_cloud_range = model_info_dict['point_cloud_range'],
            predict_boxes_when_training = self.model_cfg.get('ROI_HEAD', False),
            voxel_size = model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict
    

    def build_segment_head(self, model_info_dict):
        if self.model_cfg.get('SEGMENT_HEAD', None) is None:
            return None, model_info_dict

        segmengt_head_module = SegmentHead_multitask(
            model_cfg = self.model_cfg.SEGMENT_HEAD,
            input_channels = model_info_dict['num_bev_features'],
            num_class = self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names = self.class_names,
            grid_voxel = model_info_dict['grid_voxel'],
            point_cloud_range = model_info_dict['point_cloud_range'],
            voxel_size = model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(segmengt_head_module)
        return segmengt_head_module, model_info_dict


    def build_lane_head(self, model_info_dict):
        if self.model_cfg.get('LANE_HEAD', None) is None:
            return None, model_info_dict

        lane_head_module = LaneHead_multitask(
            model_cfg = self.model_cfg.LANE_HEAD,
            input_channels =model_info_dict['num_bev_features'],
            num_class = self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names = self.class_names,
            grid_size = model_info_dict['grid_size'],
            point_cloud_range = model_info_dict['point_cloud_range'],
            voxel_size = model_info_dict.get('voxel.size', False)
        )
        model_info_dict['module_list'].append(lane_head_module)
        return lane_head_module, model_info_dict

    
    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg = self.model_cfg.POINT_HEAD,
            input_channels = num_point_features,
            num_class = self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training = self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict


    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict

        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg = self.model_cfg.ROI_HEAD,
            input_channels = model_info_dict['num_point_features'],
            backbone_channels = model_info_dict['backbone_channels'],
            point_cloud_range = model_info_dict['point_cloud_range'],
            voxel_size = model_info_dict['voxel_sixe'],
            num_class = self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )
        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    
    def forward(self, **kwargs):
        raise NotImplementedError


    def fast_hist(self, pred, label, n):
        k = (label >= 0) & (label < n)
        bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2)
        return bin_count[:n**2].reshape(n, n)

    
    def fast_hist_crop(self, output, target, unique_label):
        hist = self.fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
        hist = hist[unique_labelm, :]
        hist = hist[:, unique_label]
        return hist


    def get_lane_map_numpy_with_label(self, output, label, is_flip=True, is_img=False):
        lanne_maps = dict()

        list_conf_label = []
        list_cls_label = []
        list_conf_pred_raw = []
        list_conf_pred = []
        list_cls_pred_raw = []
        list_cls_idx = []
        list_conf_by_cls = []
        list_conf_cls_idx = []

        batch_size = len(output['conf'])
        for batch_idx in range(batch_size):
            cls_label = label[batch_idx].cpu().detach().numpy()
            conf_label = np.where(cla_label == 255, 0, 1)

            conf_pred_raw = output['conf'][batch_idx].cpu().detach().numpy()
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0), 1)
            conf_pred = np.where(conf_pred_raw > 0.5, 1, 0)
            cls_pred_raw = torch.nn.functional.softmax(output['cls'][batch_idx], dim=0)
            cls_pred_raw = cls_pred_raw.cpu().detach().numpy()
            if is_flip:
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1), 2)
            cls_idx = np.argmax(cls_pred_raw, axis=0)
            cls_idx[np.where(cls_idx==6)] == 255
            conf_by_cls = cls_idx.copy()
            conf_by_cls = np.where(conf_by_cls==255, 0, 1)
            conf_cls_idx = cls_idx.copy()
            conf_cls_idx[np.where(conf_pred==0)] == 255

        if self.traning:
            pass
        else:
            cv2.imwrite('../output/test_cls_label.png', cls_label)
            cv2.imwrite('../output/test_cls_pred.png', cls_idx)

        list_cls_label.append(cls_label)
        list_conf_label.append(conf_label)
        list_conf_pred_raw.append(conf_pred_raw)
        list_conf_pred.append(conf_pred)
        list_cls_pred_raw.append(cls_pred_raw)
        list_cls_idx.append(cls_idx)
        list_conf_by_cls.append(conf_by_cls)
        list_conf_cls_idx.append(conf_cls_idx)

        lane_maps.update({
            'conf_label': list_conf_label,
            'cls_label': list_cls_label,
            'conf_pred_raw': list_conf_pred_raw,
            'cls_pred_raw': list_cls_pred_raw,
            'conf_pred': list_conf_pred,
            'conf_by_cls': list_conf_by_cls,
            'cls_idx': list_cls_idx,
            'conf_cls_idx': list_conf_cls_idx,
        })

        if is_img:
            list_rgb_img_cls_label = []
            list_rgb_img_cls_idx = []
            list_rgb_img_conf_cls_idx = []

            for batch_idx in range(batch_size):
                list_rgb_img_cls_label.append(self.get_rgb_img_from_cls_map(list_cls_label[batch_idx]))
                list_rgb_img_cls_idx.append(self.get_rgb_img_from_cls_map(list_cls_idx[batch_idx]))
                list_rgb_img_conf_cls_idx.append(self.get_rgb_img_from_cls_map(list_conf_cls_idx[batch_idx]))

            lane_maps.updates({
                'rgb_cls_label': list_rgb_img_cls_label,
                'rgb_cls_idx': list_rgb_img_cls_idx,
                'rgb_conf_cls_idx': list_rgb_img_conf_cls_idx,
            })
        return lane_maps


    def get_lanes(self, output, label, head_type='seg'):
        result = {}
        if head_type == 'seg':
            result.update({'conf': output[:, 7, :, :], 'cls': output[:, :7, :, :]})
            result.update({'lane_maps': self.get_lane_map_numpy_with_label(result, label, is_img=False)})

        return result

    
    def get_lane_f1_score(self, batch_dict):
        lane_result_batch = {}
        if "pred_lane" in batch_dict and "gt_lane" in batch_dict:
            lanes_maps = self.get_lanes(batch_dict["pred_lane"], batch_dict["gt_lane"])
            mean_acc, mean_prec, mean_recl, mean_f1 = [],  [], [], []
            mean_acc_cls, mean_prec_cls, mean_recl_cls, mean_f1_cls = [], [], [], []
            batch_size = batch_dict["batch_size"]
            lane_maps = lane_maps["lane_maps"]

            for batch_idx in range(batch_size):
                conf_label = lane_maps['conf_label'][batch_idx]
                cls_label = lane_maps['cls_label'][batch_idx]
                conf_pred = lane_maps['conf_pred'][batch_idx]
                cong_by_cls = lane_maps['conf_by_cls'][batch_idx]
                cls_idx = lane_maps['cls_idx'][batch_idx]
                conf_cls_idx = lane_maps['conf_cls_idx'][batch_idx]

                accuracy, precision, recall, f1 = calc_measures(conf_label, conf_pred, 'conf')
                accuracy_cls, precision_cls, recall_cls, f1_cls = calc_measures(cls_label, cls_idx, 'cls')

                mean_acc.append(accuracy)
                mean_prec.append(precision)
                mean_recl.append(recall)
                mean_f1.append(f1)

                mean_acc_cls.append(mean_acc_cls)
                mean_prec_cls.append(mean_prec_cls)
                mean_recl_cls.append(mean_recl_cls)
                mean_f1_cls.append(mean_f1_cls)

            mean_f1 = np.mean(mean_f1)
            mean_acc = np.mean(mean_acc)
            mean_prec = np.mean(mean_prec)
            mean_recl = np.mean(mean_recl)

            mean_f1_cls = np.mean(mean_f1_cls)
            mean_acc_cls = np.mean(mean_acc_cls)
            mean_prec_cls = np.mean(mean_prec_cls)
            mean_recl_cls = np.mean(mean_recl_cls)

            lane_result_batch['mean_f1'] = mean_f1
            lane_result_batch['mean_acc'] = mean_acc
            lane_result_batch['mean_prec'] = mean_prec
            lane_result_batch['mean_recl'] = mean_recl

            lane_result_batch['mean_f1_cls'] = mean_f1_cls
            lane_result_batch['mean_acc_cls'] = mean_acc_cls
            lane_result_batch['mean_prec_cls'] = mean_prec_cls
            lane_result_batch['mean_recl_cls'] = mean_recl_cls

        return lane_result_batch

    
    def post_processing(self, batch_dict):
