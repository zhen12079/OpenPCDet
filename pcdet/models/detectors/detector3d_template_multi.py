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
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_size', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index
            
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds= box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arrange(1, self.num_class+1, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores = cur_cls_preds, box_preds=cur_box_preds,
                        nms_config = post_process_cfg.NMS_CONFIG,
                        score_thresh= post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                finsl_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.calss_agnostic_nms(
                    cls_scores = cls_preds, box_preds=box_preds,
                    nms_config = post_process_cfg.NMS_CONFIG,
                    score_thresh= post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAQ_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds = final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict = recall_dict, batch_index = index, data_dict = batch_dict,
                thresh_list = post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict ={ 'pred_boxes': final_boxes, 'pred_scores': final_scores, 'pred_labels': final_labels}
            pred_dicts.append(record_dict)

        ############################################SEG###############################################################
        hist_list = []
        if "grid_ind" in batch_dict and "labels_ori" in batch_dict:
            pred_seg = batch_dict["pred_seg"]
            val_grid_ind = batch_dict["grid_ind"]
            val_pt_labs = batch_dict["labels_ori"]
            predict_labels = torch.argmax(pred_seg, dim=1)
            predict_labels =predict_labels.cpu(),detach().numpy()

            for count, i_val_grid in enumerate(val_grid_ind):
                hist_list.append(self.fast_hist_crop(
                    predict_labels[count, i_val_grid[:, 1], i_val_grid[:, 0], i_val_grid[:, 2]], val_pt_labs[count], self.unique_label
                )) 

        
        ##########################################LEAN#####################################################################
        lane_result_batch = {}
        if "pred_lane" in batch_dict and "gt_lane" in batch_dict:
            lane_maps = self.get_lanes(batch_dict["pred_lane"], batch_dict["gt_lane"])
            mean_acc, mean_prec, mean_recl, mean_f1 = [],  [], [], []
            mean_acc_cls, mean_prec_cls, mean_recl_cls, mean_f1_cls = [], [], [], []
            batch_size = batch_dict["batch_size"]
            lane_maps = lane_maps["lane_maps"]
            pred_output = []
            for batch_idx in range(batch_size):
                conf_label = lane_maps['conf_label'][batch_idx]
                cls_label = lane_maps['cls_label'][batch_idx]
                conf_pred = lane_maps['conf_pred'][batch_idx]
                cong_by_cls = lane_maps['conf_by_cls'][batch_idx]
                cls_idx = lane_maps['cls_idx'][batch_idx]
                conf_cls_idx = lane_maps['conf_cls_idx'][batch_idx]

                accuracy, precision, recall, f1 = calc_measures(conf_label, conf_pred, 'conf')
                accuracy_cls, precision_cls, recall_cls, f1_cls = calc_measures(cls_label, cls_idx, 'cls')
                pred_output_dict = {}
                pred_output_dict['cls_pred'] = [batch_idx, batch_dict['frame_id'][batch_idx], cls_label, cls_idx, f1_cls, precision_cls, recall_cls]
                pred_output_dict['conf_pred'] = [batch_idx, batch_dict['frame_id'][batch_idx], conf_label, conf_pred, f1, precision, recall]
                pred_output.append(pred_output_dict)

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

            lane_result_batch["pred_output"] = pred_output

        return pred_dicts, recall_dict, hist_list, lane_result_batch, batch_dict

    
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0
            
        cur_gt = gt_boxes
        k = cur.gt__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k+1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else: 
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled
                
            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        stste_dict = self.state_dict()
        spconv_keys = find_all_spconv_keys(self)
        update_model_state = {}
        for key, val in model_state_disk.item():
            if key in spconv_keys and key in model_state and state_dict[key].shape != val.shape:
                val_native = val.transpose(-1, -2)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
            
        if strict:
            self.load_state_dict(update_model_state)
        else:
            self.load_state_dict(model_state_disk, strict=False)
        return state_dict, update_model_state
    

    def load_params_from_file(self, filename, logger, to_cpu=False, filename_base=None):
        if not os.path.isfile(filename):
            print(filename)
            raise FileExistsError
        
        logger.info('===> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else "cuda:0"
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if filename_base and os.path.exists(filename_base):
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename_base, 'CPU' if to_cpu else 'GPU'))
            base_checkpoint =  torch.load(filename_base,map_location=loc_type)
            base_model_state_disk = base_checkpoint['model_state']

            for k, v in base_model_state_disk.items():
                if 'dense_head' in k:
                    print (k, v.shape)
                else:
                    model_state_disk[k] = v

        version = checkpoint.get('version', None)
        if version is not None:
            logger.info('===> Checkpoint trained from version: %s' % version)
            
        
        if self.training:
            no_head_state_dict = {}
            for k, v in model_state_disk.item():
                if 'lane_head' not in k:
                    no_head_state_dict[k] = v
            state_dict, update_model_state = self._load_state_dict(no_head_state_dict, strict=False)
        else:
            state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s:%s' % (key, str(state_dict[key].shape)))
        logger.info('===>Done (loaaded%d/%d)' % (len(update_model_state),len(state_dict)))



    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename,map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        if self.training:
            no_head_state_dict = {}
            for k, v in checkpoint['model_state'].items():
                if 'lane_head' not in k:
                    no_head_state_dict[k] = v
            self._load_state_dict(no_head_state_dict, strict=False)
        else:
            self._load_state_dict(checkpoint['model_state'], strict=False)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('===> Loading optimizer parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state']) 
        
        if 'version' in checkpoint:
            print('====> checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('===>Done')
        return it, epoch


    def load_params_with_det_lane(self, det_filename=None, lane_ckpt=None, to_cpu=False, optimizer=None, logger=None):
        epoch = 0
        it = 0.0
        if os.path.isfile(lane_ckpt):
            logger.info('====> Loading parameters from checkpoint %s to %s' % (lane_ckpt, 'CPU' if to_cpu else 'GPU'))
            loc_type = torch.device('cpu') if to_cpu else None
            lane_checkpoint = torch.load(lane_checkpoint, map_location=loc_type)
            epoch = lane_checkpoint.get('epoch' , -1)
            it = lane_checkpoint.get('it', 0.0)
            self._load_state_dict(lane_checkpoint['model_state'], strict=False)

        if not os.path.isfile(det_filename):
            raise FileNotFoundError

        logger.info('====> Loading parameters from checkpoint %s to %s' % (det_filename, 'CPU' if to_cpu else 'GPU'))
        loc_type =  torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(det_filename, map_location=loc_type)
        if self.trainging:
            no_head_state_dict = {}
            for k, v in checkpoint['model_state'].items():
                if 'lane_head' not in k:
                    no_head_state_dict[k] = v
            self._load_state_dict(no_head_state_dict, strict=False)
        else:
            self._load_state_dict(checkpoint['model_state'], strict=False)

        if optimizer is not None:
            if 'optimizer_state' in lane_checkpoint and lane_checkpoint['optimizer_state'] is not None:
                logger.info('===> Loading optimizer parameters from checkpoint %s to %s' % (lane_ckpt, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(lane_checkpoint['optimizer_state'])
            else:
                assert lane_ckpt[-4] == '.', lane_ckpt
                src_file, ext = lane_ckpt[:-4], lane_ckpt[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state']) 
        
        if 'version' in checkpoint:
            print('====> checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('===>Done')
        return it, epoch





















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
        self.unique_label = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19])


        # self.module_topology = [
        #     'vfe','vfe_lane', 'backbone_3d', 'map_to_bev_module','map_to_bev_module_lane', 'pfe',
        #     'backbone_2d', 'dense_head', 'point_head', 'roi_head', 'segment_head', 'lane_head'
        # ]
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head', 'segment_head', 'lane_head'
        ]

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
        # print(model_info_dict['num_rawpoint_features'])
        # print(model_info_dict['grid_size'])
        # print(model_info_dict['point_cloud_range'])
        # print(model_info_dict['voxel_size'])
        # print(model_info_dict['depth_downsample_factor'])
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            grid_size=model_info_dict['grid_size'],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_vfe_lane(self, model_info_dict):
        if self.model_cfg.get('VFE_LANE', None) is None:
            return None, model_info_dict

        vfe_module_lane = vfe.__all__[self.model_cfg.VFE_LANE.NAME](
            model_cfg=self.model_cfg.VFE_LANE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=[0.0, -11.52, -2.5, 69.12, 11.52, 0.5],
            voxel_size=[0.48,0.16,3],
            grid_size=[144,144,1],
            depth_downsample_factor=model_info_dict['depth_downsample_factor']
        )
        model_info_dict['num_point_features_lane'] = vfe_module_lane.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module_lane)
        return vfe_module_lane, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        model_info_dict['backbone_channels'] = backbone_3d_module.backbone_channels \
            if hasattr(backbone_3d_module, 'backbone_channels') else None
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )

        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_map_to_bev_module_lane(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV_LANE', None) is None:
            return None, model_info_dict

        map_to_bev_module_lane = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV_LANE.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=[144,144,1]
        )

        model_info_dict['module_list'].append(map_to_bev_module_lane)
        model_info_dict['num_bev_features_lane'] = map_to_bev_module_lane.num_bev_features
        return map_to_bev_module_lane, model_info_dict


    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            return None, model_info_dict

        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
        )

        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False),
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_segment_head(self, model_info_dict):
        if self.model_cfg.get('SEGMENT_HEAD', None) is None:
            return None, model_info_dict
        segment_head_module = SegmentHead_multitask(
            model_cfg=self.model_cfg.SEGMENT_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict.get('voxel_size', False)
        )
        model_info_dict['module_list'].append(segment_head_module)
        return segment_head_module, model_info_dict

    def build_lane_head(self, model_info_dict):
        if self.model_cfg.get('LANE_HEAD', None) is None:
            return None, model_info_dict
        lane_head_module = LaneHead_multitask(
            model_cfg=self.model_cfg.LANE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict.get('voxel_size', False)
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
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            backbone_channels=model_info_dict['backbone_channels'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def fast_hist(self,pred, label, n):
        k = (label >= 0) & (label < n)
        bin_count = np.bincount(
            n * label[k].astype(int) + pred[k], minlength=n ** 2)
        return bin_count[:n ** 2].reshape(n, n)
    def fast_hist_crop(self,output, target, unique_label):
        hist = self.fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 1)
        hist = hist[unique_label, :]
        hist = hist[:, unique_label]
        return hist
    def get_lane_map_numpy_with_label(self, output, label, is_flip=True, is_img=False):
        lane_maps = dict()

        # for batch
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
            conf_label = np.where(cls_label == 255, 0, 1)

            conf_pred_raw = output['conf'][batch_idx].cpu().detach().numpy()
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0),1)
            conf_pred = np.where(conf_pred_raw > 0.5, 1, 0)
            cls_pred_raw = torch.nn.functional.softmax(output['cls'][batch_idx], dim=0)
            cls_pred_raw = cls_pred_raw.cpu().detach().numpy()
            if is_flip:
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1),2)
            cls_idx = np.argmax(cls_pred_raw, axis=0)
            cls_idx[np.where(cls_idx==6)] = 255
            conf_by_cls = cls_idx.copy()
            conf_by_cls = np.where(conf_by_cls==255, 0, 1)
            conf_cls_idx = cls_idx.copy()
            conf_cls_idx[np.where(conf_pred==0)] = 255

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
            'conf_pred': list_conf_pred, #背景分头结果转换为0-1
            'conf_by_cls': list_conf_by_cls, #分类头结果转换为0-1
            'cls_idx': list_cls_idx, #分类头结果，0~5
            'conf_cls_idx': list_conf_cls_idx, #分类头结果同时应用背景分头结果，0~5
        })

        if is_img:
            list_rgb_img_cls_label = []
            list_rgb_img_cls_idx = []
            list_rgb_img_conf_cls_idx = []

            for batch_idx in range(batch_size):
                list_rgb_img_cls_label.append(
                    self.get_rgb_img_from_cls_map(list_cls_label[batch_idx]))
                list_rgb_img_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_cls_idx[batch_idx]))
                list_rgb_img_conf_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_conf_cls_idx[batch_idx]))
            
            lane_maps.update({
                'rgb_cls_label': list_rgb_img_cls_label,
                'rgb_cls_idx': list_rgb_img_cls_idx,
                'rgb_conf_cls_idx': list_rgb_img_conf_cls_idx,
            })
        return lane_maps
    def get_lanes(self, output, label,head_type='seg'):
        result = {}       
        if head_type == 'seg':
            result.update({'conf': output[:,7,:,:], 'cls': output[:,:7,:,:]})
            result.update({
                'lane_maps': self.get_lane_map_numpy_with_label(
                                    result, label, is_img=False)})
        # elif head_type == 'row':
        #     result.update(self.heads.get_conf_and_cls_dict(output))
        #     result.update({
        #         'lane_maps': self.get_lane_map_numpy_with_label(
        #                             result, label, is_img=self.cfg.view)})

        return result
    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:
        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                # import pdb;pdb.set_trace()
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class + 1, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:

                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)
################################################################# SEG ###############################################################
        # val_pt_labs =batch_dict["gt_seg"]#[4, 496, 432, 32]
        
        hist_list = []
        if "grid_ind" in batch_dict and "labels_ori" in batch_dict:
            pred_seg = batch_dict["pred_seg"]  # [4, 20, 496, 432, 32]
            val_grid_ind = batch_dict["grid_ind"]#[26997, 4])
            val_pt_labs = batch_dict["labels_ori"]#[26997]
            predict_labels = torch.argmax(pred_seg, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            for count, i_val_grid in enumerate(val_grid_ind):
                # i_val_grid = i_val_grid[:val_num_points[count]]
                # i_val_grid=torch.from_numpy(i_val_grid).cuda(device=predict_labels.device)
                hist_list.append(self.fast_hist_crop(
                    predict_labels[count, i_val_grid[:, 1], i_val_grid[:, 0], i_val_grid[:, 2]], val_pt_labs[count],
                    self.unique_label))

################################################################# LANE ###############################################################
        # print()
        lane_result_batch = {}
        if "pred_lane" in batch_dict and "gt_lane" in batch_dict:
            lane_maps = self.get_lanes(batch_dict["pred_lane"], batch_dict["gt_lane"])
            mean_acc, mean_prec, mean_recl, mean_f1 = [], [], [], []
            mean_acc_cls, mean_prec_cls, mean_recl_cls, mean_f1_cls = [], [], [], []
            batch_size = batch_dict["batch_size"]
            lane_maps = lane_maps['lane_maps']
            for batch_idx in range(batch_size):
                conf_label = lane_maps['conf_label'][batch_idx]
                cls_label = lane_maps['cls_label'][batch_idx]
                conf_pred = lane_maps['conf_pred'][batch_idx]
                conf_by_cls = lane_maps['conf_by_cls'][batch_idx]
                cls_idx = lane_maps['cls_idx'][batch_idx]
                conf_cls_idx = lane_maps['conf_cls_idx'][batch_idx]

                accuracy, precision, recall, f1 = calc_measures(conf_label, conf_pred, 'conf')
                accuracy_cls, precision_cls, recall_cls, f1_cls = calc_measures(cls_label, cls_idx, 'cls')
                
                #shutil.copy("/dahuafs"+batch_dict['frame_id'][0],"/dahuafs/groupdata/share/openset/leap_data/LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/LANE_DATA/HESAI/v1_hesai_multi_task_lane_only_val/")
                # print("argmax cls_pred_raw",np.argmax(lane_maps["cls_pred_raw"][0],axis=0))
                # for i in np.argmax(lane_maps["cls_pred_raw"][0],axis=0):
                #     print(i)

                # print("frame_id",batch_dict['frame_id'][0])
                # conf_pred[conf_pred==0]=255
                # conf_label[conf_label==0]=255
                # conf_by_cls[conf_by_cls==0]=255
                # cls_idx[cls_idx==5]=210
                # cls_idx[cls_idx==4]=170
                # cls_idx[cls_idx==3]=130
                # cls_idx[cls_idx==2]=90
                # cls_idx[cls_idx==1]=50

                # cls_label[cls_label==5]=210
                # cls_label[cls_label==4]=170
                # cls_label[cls_label==3]=130
                # cls_label[cls_label==2]=90
                # cls_label[cls_label==1]=50

                # cv2.imwrite("/dahuafs/userdata/31289/tmps/cls_idx.png",cls_idx)
                # cv2.imwrite("/dahuafs/userdata/31289/tmps/conf_label.png",conf_label)
                # cv2.imwrite("/dahuafs/userdata/31289/tmps/conf_pred.png",conf_pred)
                # cv2.imwrite("/dahuafs/userdata/31289/tmps/cls_label.png",cls_label)
                # import pdb;pdb.set_trace()

                mean_acc.append(accuracy)
                mean_prec.append(precision)
                mean_recl.append(recall)
                mean_f1.append(f1)

                mean_acc_cls.append(accuracy_cls)
                mean_prec_cls.append(precision_cls)
                mean_recl_cls.append(recall_cls)
                mean_f1_cls.append(f1_cls)

            mean_f1 = np.mean(mean_f1)
            mean_acc = np.mean(mean_acc)
            mean_prec = np.mean(mean_prec)
            mean_recl = np.mean(mean_recl)

            mean_f1_cls = np.mean(mean_f1_cls)
            mean_acc_cls = np.mean(mean_acc_cls)
            mean_prec_cls = np.mean(mean_prec_cls)
            mean_recl_cls = np.mean(mean_recl_cls)

            lane_result_batch["mean_f1"]=mean_f1
            lane_result_batch["mean_acc"]=mean_acc
            lane_result_batch["mean_prec"]=mean_prec
            lane_result_batch["mean_recl"]=mean_recl

            lane_result_batch["mean_f1_cls"]=mean_f1_cls
            lane_result_batch["mean_acc_cls"]=mean_acc_cls
            lane_result_batch["mean_prec_cls"]=mean_prec_cls
            lane_result_batch["mean_recl_cls"]=mean_recl_cls

        # import pdb;
        # pdb.set_trace()
        return pred_dicts, recall_dict, hist_list, lane_result_batch, batch_dict

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()  # local cache of state_dict

        spconv_keys = find_all_spconv_keys(self)

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()
            # print(key)
            # for k,v in state_dict.items():
            #     print(k)
            # import pdb;pdb.set_trace()
            if key in state_dict and state_dict[key].shape == val.shape:
                # if "quant" in key:
                # print(key)
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            # state_dict.update(update_model_state)
            # self.load_state_dict(state_dict)
            self.load_state_dict(model_state_disk, strict=False)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False, filename_base=None):
        # if not os.path.isfile(filename):
        #     raise FileNotFoundError
        loc_type = torch.device('cpu') if to_cpu else "cuda:0"

        if filename and os.path.exists(filename):
            logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
            checkpoint = torch.load(filename, map_location=loc_type)
            model_state_disk = checkpoint['model_state']
        else:
            model_state_disk = {}
        if filename_base and os.path.exists(filename_base):
            logger.info('==> Loading parameters from base checkpoint %s to %s' % (filename_base, 'CPU' if to_cpu else 'GPU'))
            base_checkpoint = torch.load(filename_base, map_location=loc_type)
            base_model_state_disk = base_checkpoint['model_state']
            for k, v in base_model_state_disk.items():
                model_state_disk[k] = v
                
        # version = checkpoint.get("version", None)
        # if version is not None:
        #     logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        # state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)
        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self._load_state_dict(checkpoint['model_state'], strict=True)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
