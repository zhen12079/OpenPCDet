import os
import torch
import torch.nn as nn
 
from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils.spconv_utils import find_all_spconv_keys
from ...utils.metric_utils import calc_measures
from .. import backbones_2d, backbones_3d, dense_haeds, roi_heads
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
