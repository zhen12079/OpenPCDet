import copy
import pickle
import json
import numba as nb
import numpy as np
# import ipdb
import sys, os
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ...datasets.dataset import DatasetTemplate
import torch
from .pypcd import PointCloud
import cv2
# CLS_NAME = ['Car', 'barrier', 0, 'Truck', 'Unknown', 'Pedestrian', 'Bus', 'non-motor', 'Other']
# det_cls = {'Car': 0, 'Pedestrian': 1, 'non-motor': 2}


class LeapDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, version='v2.0_split',
                 pcd2json_path=None, only_infer=False,add_offset=False):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.only_infer=only_infer
        self.grid_size_seg = [432, 496, 32]
        self.voxel_size_seg = [0.2505, 0.2505, 0.251]
        self.version = version
        self.pcd2json_path = pcd2json_path
        self.add_offset = add_offset
        self.cat2id = {name: i for i, name in enumerate(self.class_names)}
        self.leap_infos = []
        mode = 'train' if self.training else 'test'
        if self.logger is not None:
            self.logger.info('Loading Leap dataset')
        info_path = self.dataset_cfg.INFO_PATH[mode][0]
        if self.pcd2json_path:
            info_path = Path(self.pcd2json_path).parent.parent / info_path
        else:
            info_path = self.root_path / info_path
        self.logger.info("load from:" + str(info_path))
        if not info_path.exists():
            assert 'NOT Found ' + str(info_path)
        else:
            with open(info_path, 'rb') as f:
                self.leap_infos.extend(pickle.load(f))

        self.IoU_Threshold = {'Car': 0.5, 'Pedestrian': 0.3, 'non-motor': 0.3, 'Bus': 0.5, 'Truck': 0.5, 'barrier': 0.3}
        self.distance = [0,80]
        self.POINT_CLOUD_RANGE = dataset_cfg.POINT_CLOUD_RANGE
        # self.learning_map = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 15: 12,
        #                      16: 13, 17: 14, 18: 15, 19: 16}
        self.learning_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 10,
                             13: 11, 14: 0, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 20: 0, 21: 0}
        # self.learning_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
        #                      13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20}
        if "PCD_EXIST_PATHS" not in dataset_cfg:
            self.PCD_EXIST_PATHS = [
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_yintai1/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_yintai0/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_jianghong_-2/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_jianghong_-1/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_07/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_06/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_05/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_04/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_03/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_02/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_00/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_01/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_tunnel/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_Q/pcd",

                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_garage/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_00/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_01/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_02/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_03/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_04/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_05/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_06/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_07/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_08/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_09/pcd",

                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/HESAI_seg_only_1.1w/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/HESAI_seg_only_val_3k/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/multi_task_train_6w/pcd",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/multi_task_val_3k/pcd"]

            self.SEG_LABEL_EXIST_PATHS = [
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_yintai1/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_yintai0/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_jianghong_-2/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_jianghong_-1/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_07/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_06/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_05/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_04/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_03/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_02/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_00/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_01/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_tunnel/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/train_Q/label",

                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_garage/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_00/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_01/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_02/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_03/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_04/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_05/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_06/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_07/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_08/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/valid_09/label",

                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/HESAI_seg_only_val_3k/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/HESAI_seg_only_1.1w/label",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/multi_task_train_6w/label_seg",
                self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/SEG_DATA/multi_task_val_3k/label_seg"]
        # self.seg_label_path =self.root_path / "LEAP_MULTITASKS_DATASET_FACTORY/LEAP_DATA/multi_task_train_6w/label_seg"
        else:
            self.PCD_EXIST_PATHS = dataset_cfg.PCD_EXIST_PATHS
            self.SEG_LABEL_EXIST_PATHS = dataset_cfg.SEG_LABEL_EXIST_PATHS
        self.PCD_EXIST_PATHS = [Path(i) for i in self.PCD_EXIST_PATHS]
        self.SEG_LABEL_EXIST_PATHS = [Path(i) for i in self.SEG_LABEL_EXIST_PATHS]
        if self.pcd2json_path:
            self.PCD_EXIST_PATHS.append(Path(self.pcd2json_path))
        DATASET_PATH_ROOT = Path("/dahuafs/groupdata/share/openset/leap_data/LEAP_MULTITASKS_DATASET_FACTORY")
        DATASET_PATH_WHITE_DATA = DATASET_PATH_ROOT / "WHITE_DATA"
        DATASET_PATH_OPEN_DATA = DATASET_PATH_ROOT / "OPEN_DATA"
        DATASET_PATH_WRONG_DATA = DATASET_PATH_ROOT / "WRONG_DATA"
        self.small_num_multi_fold_path_list = [i for i in DATASET_PATH_WHITE_DATA.rglob("*.pcd")] + \
                                              [i for i in DATASET_PATH_OPEN_DATA.rglob("*.pcd")] + \
                                              [i for i in DATASET_PATH_WRONG_DATA.rglob("*.pcd")]
        self.small_num_multi_fold_name_list = [i.name for i in self.small_num_multi_fold_path_list]
        # self.__getitem__(1)

    def get_cat_ids(self, idx):
        info = self.leap_infos[idx]
        gt_names = set(info['annos']['name'])
        cat_ids = []
        for name in gt_names:
            if name in self.class_names:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def read_pcd(self, lidar_file):
        with open(lidar_file, 'rb') as f:
            lines = f.readlines()
        point = []
        for line in lines:
            line = line.strip().split()
            if len(line) == 6:
                line = line[:4]
            if len(line) == 4:
                p_4 = list(map(float, line))
                point.append(p_4)
        point = np.array(point, dtype=np.float32)
        point[:, 3] /= 255
        return point

    def get_lidar(self, idx):
        # print(idx)
        if "/" in idx:
            if os.path.exists(idx):
                lidar_file = Path(idx)
            else:
                lidar_file = Path("/dahuafs"+idx)
                if not lidar_file.exists():
                    lidar_file = Path(str(lidar_file).replace("/dahuafs",""))
        else:
            if idx + '.pcd' in self.small_num_multi_fold_name_list:
                lidar_file = self.small_num_multi_fold_path_list[self.small_num_multi_fold_name_list.index(idx + ".pcd")]
            else:
                for path in self.PCD_EXIST_PATHS:
                    lidar_file = path / (idx + '.pcd')
                    lidar_file_no_dahua = Path(str(lidar_file).replace("/dahuafs",""))
                    if lidar_file.exists():
                        break
                    if lidar_file_no_dahua.exists():
                        lidar_file = lidar_file_no_dahua
                        break
        # print(lidar_file)
        assert lidar_file.exists()
        # points = self.read_pcd(lidar_file)
        points=np.asarray(PointCloud.from_path(lidar_file).pc_data.tolist(), dtype=np.float32)[:, :4]
###################################################################################坐标系变换测试########################
        if self.add_offset:
            points[:, 0] += 1.75
            points[:, 2] += 1.25
###################################################################################坐标系变换测试########################
        points[:, 3] /= 255
        return points

    def get_lidar_lane_full_path(self, lidar_file):
        # assert lidar_file.exists()
        if "HESAI" in str(lidar_file):
            points=np.asarray(PointCloud.from_path(lidar_file).pc_data.tolist(), dtype=np.float32)[:, :4]
            points[:, 3] /= 255
        else:
            points = self.read_pcd(lidar_file)
        return points

    def get_road_planes(self, idx):
        para_file = self.root_path / 'ground_para' / (idx + '.txt')
        assert para_file.exists()
        with open(para_file, 'r') as fid:
            para_info = fid.readlines()
        para_str = para_info[0].split(':')
        A = float(para_str[1][:-1])
        B = float(para_str[2][:-1])
        C = float(para_str[3][:-1])
        D = float(para_str[4])
        road_planes = [A, B, C, D]
        return road_planes

    def if_clean_this_data(self, cls, position, scale, heading):
        headings = np.array(heading, dtype=np.float32)
        positions = np.array(position, dtype=np.float32)
        scales = np.array(scale, dtype=np.float32)
        flag1 = False
        flag2 = False
        flag1 = flag1 or scales[0] <= 0 or scales[1] <= 0 or scales[2] <= 0
        # if cls=='Car':
        #     flag2 =flag2 or scales[0] > 10 or scales[1] > 5 or scales[2] > 5
        #     flag2 =flag2 or scales[0] < 1 or scales[1] < 1 or scales[2]  < 1
        if cls == 'Pedestrian':
            flag2 = flag2 or scales[0] > 3 or scales[1] > 3 or scales[2] > 3
            flag2 = flag2 or scales[0] < 0.2 or scales[1] < 0.2 or scales[2] < 0.2
        if cls == 'non-motor':
            flag2 = flag2 or scales[0] > 5 or scales[1] > 5 or scales[2] > 5
            flag2 = flag2 or scales[0] < 0.2 or scales[1] < 0.2 or scales[2] < 0.2
        # clean by val nan
        flag3 = np.isnan(headings).any() or np.isnan(positions).any() or np.isnan(scales).any()
        # clean by position
        flag4 = positions[0] > 300 or positions[0] < -10
        flag5 = positions[1] > 200 or positions[1] < -200
        flag6 = positions[2] > 20 or positions[2] < -5

        flag = flag1 or flag2 or flag3 or flag4 or flag5 or flag6
        return flag

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    def evaluation(self, det_annos, class_names, fp, **kwargs):
        if 'annos' not in self.leap_infos[0].keys():
            return None, {}
        # from .compute_precision_recall_3D_multiprocess_linux import evaluation, gt_hash, filter_by_point_range, \
        #     divide_by_distance, filter_by_score, count_nums, filter_by_other
        from .compute_precision_recall_3D_multiprocess_linux_per_score import evaluation_per_score, gt_hash, \
            filter_by_point_range, \
            divide_by_distance, filter_by_score, count_nums, filter_by_other

        eval_det_annos = copy.deepcopy(det_annos)
        eval_det_annos = gt_hash(eval_det_annos, if_gt=False)

        eval_gt_annos = copy.deepcopy(self.leap_infos)
        eval_gt_annos = gt_hash(eval_gt_annos, if_gt=True, add_offset=self.add_offset)
        filter_by_point_range(self.POINT_CLOUD_RANGE, eval_gt_annos, eval_det_annos)
        # import pdb;pdb.set_trace()
        filter_by_other(eval_gt_annos, eval_det_annos)

        # fp.write('\n' + str(kwargs['output_path']) + ' eval !!!\n')
        fp.write('IoU_Threshold: {}\n'.format(self.IoU_Threshold))
        for dis in range(1, len(self.distance)):
            fp.write(
                '\n---------------------------{}m --- {}m---------------------------\n'.format(self.distance[dis - 1],
                                                                                               self.distance[dis]))
            print('\n---------------------------{}m --- {}m---------------------------\n'.format(self.distance[dis - 1],
                                                                                                 self.distance[dis]))
            # print('\n Thresh {}:\n'.format(self.Score_Threshold))
            # sub_gt, sub_pr = divide_by_distance(self.distance[dis - 1], self.distance[dis], eval_gt_annos,
            #                                     eval_det_annos)
            # print("##################gt_info#######################")
            # count_nums(sub_gt)

            # for T in range(1,10,1):
            #     fp.write('\n Thresh{}:\n'.format(T*0.1))
            #     print('\n Thresh{}:\n'.format(T*0.1))
            #     filter_by_score(sub_pr, T*0.1)
            #     count_nums(sub_pr)
            #     evaluation(sub_gt, sub_pr, class_names,fp,iou_th=self.IoU_Threshold,if_3DIoU=True,num_workers=kwargs["num_workers"])

            print("##################gt_info#######################")
            sub_gt_tmp, sub_pr_tmp = divide_by_distance(self.distance[dis - 1], self.distance[dis], eval_gt_annos,
                                                        eval_det_annos)
            print("gt info:")
            count_nums(sub_gt_tmp)
            # fp.write('\n Thresh {}:\n'.format(self.Score_Threshold))
            # import pdb;pdb.set_trace()
            # filter_by_score(sub_pr_tmp, self.Score_Threshold)
            print("pred info:")
            count_nums(sub_pr_tmp)
            # evaluation(sub_gt_tmp, sub_pr_tmp, class_names, fp, iou_th=self.IoU_Threshold, if_3DIoU=True,
            #            num_workers=kwargs["num_workers"])
            evaluation_per_score(sub_gt_tmp, sub_pr_tmp, class_names, fp, iou_th=self.IoU_Threshold, if_3DIoU=True,
                                 num_workers=kwargs["num_workers"])

        fp.close()
        ap_result_str, ap_dict = None, None

        return ap_result_str, ap_dict

    def __len__(self):
        return len(self.leap_infos)

    def read_label(self, label_path, dtype=np.uint8):
        labels = []
        with open(str(label_path), 'r') as f:
            for line in f.readlines():
                labels.append(dtype(line.strip().split('\x00')[-1]))
        labels = np.array(labels).reshape(-1, 1)
        return labels

    def nb_process_label(self, processed_label, sorted_label_voxel_pair):
        label_size = 256
        counter = np.zeros((label_size,), dtype=np.uint16)
        counter[sorted_label_voxel_pair[0, 3]] = 1
        cur_sear_ind = sorted_label_voxel_pair[0, :3]
        for i in range(1, sorted_label_voxel_pair.shape[0]):
            cur_ind = sorted_label_voxel_pair[i, :3]
            if not np.all(np.equal(cur_ind, cur_sear_ind)):
                processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
                counter = np.zeros((label_size,), dtype=np.uint16)
                cur_sear_ind = cur_ind
            counter[sorted_label_voxel_pair[i, 3]] += 1
        processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
        return processed_label

    def SemKITTI2train_single(self, label):
        return label - 1  # uint8 trick

    def SemKITTI2train(self, label):
        if isinstance(label, list):
            return [self.SemKITTI2train_single(a) for a in label]
        else:
            return self.SemKITTI2train_single(label)

    def swap(self, pt1, pt2, start_angle, end_angle, label1, label2):
        # calculate horizontal angle for each point
        yaw1 = -np.arctan2(pt1[:, 1], pt1[:, 0])
        yaw2 = -np.arctan2(pt2[:, 1], pt2[:, 0])

        # select points in sector
        idx1 = np.where((yaw1 > start_angle) & (yaw1 < end_angle))
        idx2 = np.where((yaw2 > start_angle) & (yaw2 < end_angle))

        # swap
        pt1_out = np.delete(pt1, idx1, axis=0)
        pt1_out = np.concatenate((pt1_out, pt2[idx2]))
        pt2_out = np.delete(pt2, idx2, axis=0)
        pt2_out = np.concatenate((pt2_out, pt1[idx1]))

        label1_out = np.delete(label1, idx1)
        label1_out = np.concatenate((label1_out, label2[idx2]))
        label2_out = np.delete(label2, idx2)
        label2_out = np.concatenate((label2_out, label1[idx1]))
        assert pt1_out.shape[0] == label1_out.shape[0]
        assert pt2_out.shape[0] == label2_out.shape[0]

        return pt1_out, pt2_out, label1_out, label2_out

    def rotate_copy(self, pts, labels, instance_classes, Omega):
        # extract instance points
        pts_inst, labels_inst = [], []
        for s_class in instance_classes:
            pt_idx = np.where((labels == s_class))
            pts_inst.append(pts[pt_idx])
            labels_inst.append(labels[pt_idx])
        pts_inst = np.concatenate(pts_inst, axis=0)
        labels_inst = np.concatenate(labels_inst, axis=0)

        # rotate-copy
        if len(pts_inst) == 0:
            # print("No instance in pts, reutrn")
            return None, None
        pts_copy = [pts_inst]
        labels_copy = [labels_inst]
        for omega_j in Omega:
            rot_mat = np.array([[np.cos(omega_j),
                                 np.sin(omega_j), 0],
                                [-np.sin(omega_j),
                                 np.cos(omega_j), 0], [0, 0, 1]])
            new_pt = np.zeros_like(pts_inst)
            new_pt[:, :3] = np.dot(pts_inst[:, :3], rot_mat)
            new_pt[:, 3] = pts_inst[:, 3]
            pts_copy.append(new_pt)
            labels_copy.append(labels_inst)
        pts_copy = np.concatenate(pts_copy, axis=0)
        labels_copy = np.concatenate(labels_copy, axis=0)
        return pts_copy, labels_copy

    def polarmix(self, pts1, labels1, pts2, labels2, alpha, beta, Omega, instance_classes=[6, 7, 8, 9, 10, 11]):
        pts_out, labels_out = pts1, labels1
        # swapping
        if np.random.random() < 0.5:
            pts_out, _, labels_out, _ = self.swap(pts1, pts2, start_angle=alpha, end_angle=beta, label1=labels1,
                                                  label2=labels2)

        # rotate-pasting
        if np.random.random() < 1.0:
            # rotate-copy
            pts_copy, labels_copy = self.rotate_copy(pts2, labels2, instance_classes, Omega)
            # paste
            if pts_copy is not None:
                pts_out = np.concatenate((pts_out, pts_copy), axis=0)
                labels_out = np.concatenate((labels_out, labels_copy), axis=0)

        return pts_out, labels_out
    def get_seg_label_one_pcd(self, points1, sample_idx):

        # for seg_path in self.SEG_LABEL_EXIST_PATHS:
        if "/" not in sample_idx:
            for seg_path in self.SEG_LABEL_EXIST_PATHS:
                seg_label_path = seg_path / (sample_idx + ".label")
                if seg_label_path.exists():
                    break
        else:
            seg_label_path = sample_idx

        # if seg_label_path.exists() and not self.training:
        
        labels1 = self.read_label(str(seg_label_path), dtype=np.uint8)
        labels1 = np.vectorize(self.learning_map.__getitem__)(labels1)
        labels_ori = labels1.astype(np.uint8)
        labels1 = labels_ori.reshape(-1)
        labels1 = labels1[:, np.newaxis]

        grid_ind = (np.floor((np.clip(points1[:, :3], self.point_cloud_range[:3],
                                        self.point_cloud_range[3:]) - self.point_cloud_range[
                                                                    :3]) / self.voxel_size_seg)).astype(
            np.int64)
        voxel_labels = np.ones(self.grid_size_seg, dtype=np.uint8) * 255
        label_voxel_pair = np.concatenate([grid_ind, labels1], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # voxel_labels = self.nb_process_label(np.copy(voxel_labels), label_voxel_pair)
        voxel_labels = nb_process_label(np.copy(voxel_labels), label_voxel_pair)
        voxel_labels = voxel_labels.transpose(1, 0, 2)
        # voxel_labels = self.SemKITTI2train(voxel_labels)  # .type(torch.LongTensor)
        # labels_ori = self.SemKITTI2train(labels_ori)
        # points1[:,3] /= 255
        return voxel_labels, grid_ind, labels_ori

    def get_seg_label(self, points1, points2, sample_idx, sample_idx2):

        # for seg_path in self.SEG_LABEL_EXIST_PATHS:
        for seg_path in self.SEG_LABEL_EXIST_PATHS:
            seg_label_path = seg_path / (sample_idx + ".label")
            if seg_label_path.exists():
                break
        if sample_idx2 !=None:
            for seg_path in self.SEG_LABEL_EXIST_PATHS:
                seg_label_path2 = seg_path / (sample_idx2 + ".label")
                if seg_label_path2.exists():
                    break
        if seg_label_path.exists() and not self.training:
            labels1 = self.read_label(str(seg_label_path), dtype=np.uint8)
            labels1 = np.vectorize(self.learning_map.__getitem__)(labels1)
            labels_ori = labels1.astype(np.uint8)
            labels1 = labels_ori.reshape(-1)
            labels1 = labels1[:, np.newaxis]

            grid_ind = (np.floor((np.clip(points1[:, :3], self.point_cloud_range[:3],
                                            self.point_cloud_range[3:]) - self.point_cloud_range[
                                                                        :3]) / self.voxel_size_seg)).astype(
                np.int64)
            voxel_labels = np.ones(self.grid_size_seg, dtype=np.uint8) * 255
            label_voxel_pair = np.concatenate([grid_ind, labels1], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            # voxel_labels = self.nb_process_label(np.copy(voxel_labels), label_voxel_pair)
            voxel_labels = nb_process_label(np.copy(voxel_labels), label_voxel_pair)
            voxel_labels = voxel_labels.transpose(1, 0, 2)
            # voxel_labels = self.SemKITTI2train(voxel_labels)  # .type(torch.LongTensor)
            # labels_ori = self.SemKITTI2train(labels_ori)
            points1[:,3] /= 255
            return points1, voxel_labels, grid_ind, labels_ori
        elif seg_label_path.exists() and seg_label_path2.exists() and self.training:
            labels1 = self.read_label(str(seg_label_path), dtype=np.uint8)
            labels1 = np.vectorize(self.learning_map.__getitem__)(labels1)
            labels1 = labels1.astype(np.uint8).reshape(-1)
            # import pdb;pdb.set_trace()
            # labels1 = labels_ori.reshape(-1)
            labels2 = self.read_label(str(seg_label_path2), dtype=np.uint8)
            labels2 = np.vectorize(self.learning_map.__getitem__)(labels2)
            labels2 = labels2.astype(np.uint8).reshape(-1)

            alpha = (np.random.random() - 1) * np.pi / 3
            beta = alpha + np.pi / 3
            points, labels = self.polarmix(points1, labels1, points2, labels2, alpha=alpha, beta=beta, Omega=[])
            labels = labels.reshape(-1, 1)

            # labels1 = labels1[:,np.newaxis]
            random_ind = np.random.permutation(points.shape[0])
            points = points[random_ind, :]
            labels = labels[random_ind, :]

            # rotate
            rotate_rad = np.deg2rad((np.random.random() * 2 - 1) * 30)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            points[:, :2] = np.dot(points[:, :2], j)
            # flip
            flip_type = np.random.choice(2, 1)
            if flip_type == 1:
                points[:, 1] = -points[:, 1]

            grid_ind = (np.floor((np.clip(points[:, :3], self.point_cloud_range[:3],
                                            self.point_cloud_range[3:]) - self.point_cloud_range[
                                                                        :3]) / self.voxel_size_seg)).astype(
                np.int64)
            voxel_labels = np.ones(self.grid_size_seg, dtype=np.uint8) * 255
            label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
            label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
            voxel_labels = self.nb_process_label(np.copy(voxel_labels), label_voxel_pair)
            voxel_labels = voxel_labels.transpose(1, 0, 2)
            # voxel_labels = self.SemKITTI2train(voxel_labels)  # .type(torch.LongTensor)
            # labels_ori = self.SemKITTI2train(labels)
            # print(labels.shape, voxel_labels.shape, grid_ind.shape, labels_ori.shape)
            points[:,3] /= 255
            return points, voxel_labels, grid_ind, labels
            # else:
            #     continue
        if not seg_label_path.exists():
            print(seg_label_path)
            raise
            # return None

    def get_lane_label(self,label_path):
        # print(label_path)
        # try:
        if os.path.splitext(label_path)[-1] == ".png":
            bev_tensor_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        else:
            bev_tensor_label = np.load(label_path)
        bev_tensor_label = bev_tensor_label[:,:144]
        # except Exception as e:
        #     print(label_path)
        #     print("some problem in file:" + self.leap_infos[index]['point_cloud_info']['lidar_idx'])
        #     print(str(e))
        # for i in bev_tensor_label:
        #     print(i)
        # print(np.min(bev_tensor_label[:,:144]))
        # bev_tensor_label_resized = cv2.resize(bev_tensor_label,(92,276),interpolation=cv2.INTER_AREA)
        # cv2.imwrite("/dahuafs/userdata/31289/tmps/bev_tensor_label_resized.png",bev_tensor_label_resized)
        # # import pdb;pdb.set_trace()
        # print("saved")
        # print(bev_tensor_label_resized.shape)
        # for i in bev_tensor_label_resized:
        #     print(i)
        return bev_tensor_label
    def __getitem__(self, index):
        def index2dict(index):
            info = copy.deepcopy(self.leap_infos[index])
            # import pdb;pdb.set_trace()
            sample_idx = info['timestamp']
            # sample_idx = info['point_cloud_info']['lidar_idx']
            get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
            input_dict = {
                'frame_id': sample_idx,#info['point_cloud_info']['pc'],#sample_idx,#
            }
            input_dict['gt_seg']=np.array([])
            input_dict['gt_boxes']=np.array([])
            input_dict['gt_names']=np.array([])

            if 'annos' in info:
                annos = info['annos']

                annos = common_utils.drop_info_with_name(annos, name='Unknown')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_lidar = np.concatenate([loc, dims, rots], axis=1).astype(np.float32)

                input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar
                })
                if "gt_boxes2d" in get_item_list:
                    input_dict['gt_boxes2d'] = annos["bbox"]

            if "points" in get_item_list:
                
                # input_dict['points'] = points
                # if "lane" in get_item_list and 'pc' in info['point_cloud_info']:
                #     points = self.get_lidar_lane_full_path("/dahuafs"+info['point_cloud_info']['pc'])
                if "lane" in get_item_list:
                    points = self.get_lidar_lane_full_path(info['pcd'])
                else:
                    points = self.get_lidar(sample_idx)

                    # points = self.get_lidar(sample_idx)
                    
                # if "seg" in get_item_list and not self.only_infer:
                #     points = self.get_lidar(sample_idx)
                #     if self.training:
                #         sample_idx2 = self.leap_infos[np.random.choice(len(self.leap_infos))]['point_cloud_info']['lidar_idx']
                #         points2 = self.get_lidar(sample_idx2)
                #     else:
                #         points2 = None
                #         sample_idx2 = None

                # elif "seg" in get_item_list and self.only_infer:
                #     # points = self.get_lidar(sample_idx)          
                #     # points[:, 3] /= 255
                #     points_num = points.shape[0]
                #     grid_ind = (np.floor((np.clip(points[:, :3], self.point_cloud_range[:3],
                #                 self.point_cloud_range[3:]) - self.point_cloud_range[
                #                                             :3]) / self.voxel_size_seg)).astype(np.int64)
                #     input_dict['grid_ind'] = grid_ind
                #     input_dict['points_num'] = points_num
                #     input_dict['points_ori'] = points
                
                # if "lane" not in get_item_list and "seg" not in get_item_list:
                #     points = self.get_lidar(sample_idx)

                input_dict['points'] = points

            if "road_planes" in get_item_list:
                road_planes = self.get_road_planes(sample_idx)
                input_dict['road_planes'] = road_planes

            if "seg" in get_item_list and not self.only_infer:
                if "seg_label_path" in info['point_cloud_info']:
                    seg_label_path = "/dahuafs"+info['point_cloud_info']['seg_label_path']
                    seg_label, grid_ind, labels_ori = self.get_seg_label_one_pcd(points, seg_label_path)
                else:
                    seg_label, grid_ind, labels_ori = self.get_seg_label_one_pcd(points, sample_idx)
                input_dict['gt_seg'] = seg_label
                input_dict['grid_ind'] = grid_ind
                input_dict['labels_ori'] = labels_ori
                    # for seg_path in self.SEG_LABEL_EXIST_PATHS:
                    #     seg_label_path = seg_path / (sample_idx + ".label")
                    #     if seg_label_path.exists():
                    #         break
                    # seg_label, grid_ind, labels_ori = self.get_seg_label_one_pcd(points, seg_label_path)
                    # input_dict['gt_seg'] = seg_label
                    # input_dict['grid_ind'] = grid_ind
                    # input_dict['labels_ori'] = labels_ori

            if "lane" in get_item_list:
                lane_label = self.get_lane_label(info['label'])
                input_dict['gt_lane'] = lane_label
            
            # if "lane" in get_item_list:
            #     if "lane_label_path" in info['point_cloud_info']:
            #         lane_label = self.get_lane_label("/dahuafs"+info['point_cloud_info']['lane_label_path'])
            #         input_dict['gt_lane'] = lane_label
            #     elif "bev_tensor_label" in info['point_cloud_info']:
            #         lane_label = self.get_lane_label("/dahuafs"+info['point_cloud_info']['bev_tensor_label'])
            #         input_dict['gt_lane'] = lane_label


            return input_dict
        def index2dict_safe(index):
            while True:
                try:
                    input_dict = index2dict(index)
                    assert len(input_dict['points'].shape)==2
                    return input_dict
                except Exception as e:
                    print("some problem in file:" + self.leap_infos[index]['pcd'])
                    # print("some problem in file:" + self.leap_infos[index]['point_cloud_info']['lidar_idx'])
                    print(str(e))
                    print("change one")
                    index = np.random.choice(len(self.leap_infos))
            raise RuntimeError("error in data read")
        # import pdb;pdb.set_trace()
        input_dict = index2dict_safe(index)
        # print(input_dict)
        data_dict = self.prepare_data(data_dict=input_dict)
        # print("points.shape",data_dict["points"].shape)
        # print("voxels.shape",data_dict["voxels"].shape)
        # print("gt_lane.shape",data_dict["gt_lane"].shape)
        # import pdb;pdb.set_trace()
        return data_dict

@nb.jit('u1[:, :, :](u1[:, :, :], i8[:, :])',
        nopython=True,
        cache=True,
        parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
