from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from ..utils import box_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        if hasattr(self.dataset_cfg.DATA_AUGMENTOR, "AUG_CONFIG_LIST") and self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST!=None:
            self.data_augmentor = DataAugmentor(
                self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
            ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range,
            training=self.training, num_point_features=self.point_feature_encoder.num_point_features
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    # def MASIC(self, data_dict):
    #     x_line = 54
    #     y_line = 0
    #     gt_boxes_part_lb, gt_boxes_part_rb, gt_boxes_part_lt, gt_boxes_part_rt = [], [], [], []
    #     points_part_lb, points_part_rb, points_part_lt, points_part_rt = [], [], [], []
    #     names_part_lb, names_part_rb, names_part_lt, names_part_rt = [], [], [], []
    #     for data in data_dict:
    #         points = data["points"]
    #         if len(data["gt_boxes"]) == 0:
    #             gt_boxes_part_lb_, gt_boxes_part_rb_, gt_boxes_part_lt_, gt_boxes_part_rt_ = [], [], [], []
    #             points_part_lb_, points_part_rb_, points_part_lt_, points_part_rt_ = [], [], [], []
    #             names_part_lb_, names_part_rb_, names_part_lt_, names_part_rt_ = [], [], [], []
    #             for point in points:
    #                 if np.isnan(point).any(): continue
    #                 if point[0] <= x_line and point[1] <= y_line:
    #                     points_part_lb_.append(point)
    #                 elif point[0] < x_line and point[1] > y_line:
    #                     points_part_rb_.append(point)
    #                 elif point[0] > x_line and point[1] <= y_line:
    #                     points_part_lt_.append(point)
    #                 else:
    #                     points_part_rt_.append(point)
    #
    #             points_part_lb.append(points_part_lb_)
    #             points_part_rb.append(points_part_rb_)
    #             points_part_lt.append(points_part_lt_)
    #             points_part_rt.append(points_part_rt_)
    #             gt_boxes_part_lb.append(gt_boxes_part_lb_)
    #             gt_boxes_part_rb.append(gt_boxes_part_rb_)
    #             gt_boxes_part_lt.append(gt_boxes_part_lt_)
    #             gt_boxes_part_rt.append(gt_boxes_part_rt_)
    #             names_part_lb.append(names_part_lb_)
    #             names_part_rb.append(names_part_rb_)
    #             names_part_lt.append(names_part_lt_)
    #             names_part_rt.append(names_part_rt_)
    #             continue
    #         boxes_corners = box_utils.boxes_to_corners_3d(data["gt_boxes"])
    #         x_min = np.min(boxes_corners[:, :4, 0], axis=1)
    #         x_max = np.max(boxes_corners[:, :4, 0], axis=1)
    #         y_min = np.min(boxes_corners[:, :4, 1], axis=1)
    #         y_max = np.max(boxes_corners[:, :4, 1], axis=1)
    #         find_y_line = False
    #         find_x_line = False
    #         for i in range(10):
    #             for xmin, xmax in zip(x_min, x_max):
    #                 if xmin < x_line and xmax > x_line and min(xmax - x_line, x_line - xmin) > 0.33:
    #                     if xmax - x_line > x_line - xmin:
    #                         x_line -= (xmax - xmin) / 5
    #                     else:
    #                         x_line += (xmax - xmin) / 5
    #                     break
    #                 if x_min[-1] == xmin and x_max[-1] == xmax:
    #                     find_x_line = True
    #             if find_x_line:
    #                 break
    #         if find_x_line:
    #             for i in range(10):
    #                 for ymin, ymax in zip(y_min, y_max):
    #                     if ymin < y_line and ymax > y_line and min(ymax - y_line, y_line - ymin) > 0.33:
    #                         if ymax - y_line > y_line - ymin:
    #                             y_line -= (ymax - ymin) / 5
    #                         else:
    #                             y_line += (ymax - ymin) / 5
    #                         break
    #                     if y_min[-1] == ymin and y_max[-1] == ymax:
    #                         find_y_line = True
    #                 if find_y_line:
    #                     break
    #         if find_x_line and find_y_line:
    #             gt_boxes_part_lb_, gt_boxes_part_rb_, gt_boxes_part_lt_, gt_boxes_part_rt_ = [], [], [], []
    #             points_part_lb_, points_part_rb_, points_part_lt_, points_part_rt_ = [], [], [], []
    #             names_part_lb_, names_part_rb_, names_part_lt_, names_part_rt_ = [], [], [], []
    #             for point in points:
    #                 if np.isnan(point).any(): continue
    #                 if point[0] <= x_line and point[1] <= y_line:
    #                     points_part_lb_.append(point)
    #                 elif point[0] < x_line and point[1] > y_line:
    #                     points_part_rb_.append(point)
    #                 elif point[0] > x_line and point[1] <= y_line:
    #                     points_part_lt_.append(point)
    #                 else:
    #                     points_part_rt_.append(point)
    #             for box, name in zip(data["gt_boxes"], data["gt_names"]):
    #                 if box[0] <= x_line and box[1] <= y_line:
    #                     gt_boxes_part_lb_.append(box)
    #                     names_part_lb_.append(name)
    #                 elif box[0] < x_line and box[1] > y_line:
    #                     gt_boxes_part_rb_.append(box)
    #                     names_part_rb_.append(name)
    #                 elif box[0] > x_line and box[1] <= y_line:
    #                     gt_boxes_part_lt_.append(box)
    #                     names_part_lt_.append(name)
    #                 else:
    #                     gt_boxes_part_rt_.append(box)
    #                     names_part_rt_.append(name)
    #
    #             points_part_lb.append(points_part_lb_)
    #             points_part_rb.append(points_part_rb_)
    #             points_part_lt.append(points_part_lt_)
    #             points_part_rt.append(points_part_rt_)
    #             gt_boxes_part_lb.append(gt_boxes_part_lb_)
    #             gt_boxes_part_rb.append(gt_boxes_part_rb_)
    #             gt_boxes_part_lt.append(gt_boxes_part_lt_)
    #             gt_boxes_part_rt.append(gt_boxes_part_rt_)
    #             names_part_lb.append(names_part_lb_)
    #             names_part_rb.append(names_part_rb_)
    #             names_part_lt.append(names_part_lt_)
    #             names_part_rt.append(names_part_rt_)
    #     find_num = len(gt_boxes_part_lb)
    #     if find_num != 0:
    #         lb_index = np.random.choice(find_num)
    #         rb_index = np.random.choice(find_num)
    #         lt_index = np.random.choice(find_num)
    #         rt_index = np.random.choice(find_num)
    #         data_dict[0]["gt_boxes"] = np.array(gt_boxes_part_lb[lb_index] + gt_boxes_part_rb[rb_index] + \
    #                                             gt_boxes_part_lt[lt_index] + gt_boxes_part_rt[rt_index])
    #         data_dict[0]["points"] = np.array(points_part_lb[lb_index] + points_part_rb[rb_index] + \
    #                                           points_part_lt[lt_index] + points_part_rt[rt_index])
    #         data_dict[0]["gt_names"] = np.array(names_part_lb[lb_index] + names_part_rb[rb_index] + \
    #                                             names_part_lt[lt_index] + names_part_rt[rt_index])
    #
    #     data_dict = data_dict[0]
    #     return data_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:

            # if isinstance(data_dict, list):
            #     data_dict = self.MASIC(data_dict)
            # assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            if 'gt_boxes' in data_dict:
                gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)
                if hasattr(self.dataset_cfg.DATA_AUGMENTOR,
                        "AUG_CONFIG_LIST") and self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST != None:

                    data_dict = self.data_augmentor.forward(
                        data_dict={
                            **data_dict,
                            'gt_boxes_mask': gt_boxes_mask
                        }
                    )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # if len(data_dict['gt_names'])==0:
            #     print(data_dict)
            if len(data_dict['gt_names']) != 0:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
                gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes'] = gt_boxes
            # if len(data_dict['gt_boxes'])==0:
            #     print(gt_classes,gt_boxes,data_dict['gt_names'])

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        # if data_dict.get('points', None) is not None:
        #     data_dict = self.point_feature_encoder.forward(data_dict) #当前检测中这一步没有任何作用，可去除
        data_dict['use_lead_xyz'] = True

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        # if self.training and len(data_dict['gt_boxes']) == 0:
        #     new_index = np.random.randint(self.__len__())
        #     return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        # if batch_size != 8:
        #     print(batch_size,batch_list)

        ret = {}
        points_len = 0
        for key, val in data_dict.items():
            # print(key,val)
            try:
                if key in ['voxels', 'voxel_num_points','voxels_lane', 'voxel_num_points_lane']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords','voxel_coords_lane']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                        if key == 'points':
                            points_len += 1
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, 8), dtype=np.float32)
                    for k in range(batch_size):
                        if len(val[k]) == 0: continue
                        # print("-----------------",val[k].shape,val[k].__len__(),val[0].shape[-1])
                        # if val[0].shape[-1]==1:
                        #     print(key,val)
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_seg']:
                    voxel_labels=[]
                    for d in val:
                        voxel_labels.append(d)
                    voxel_labels = np.vstack(voxel_labels).reshape((batch_size,496,432,-1))
                    ret[key] = voxel_labels
                elif key in ['gt_lane']:
                    voxel_labels=[]
                    for d in val:
                        voxel_labels.append(d)
                    voxel_labels = np.vstack(voxel_labels).reshape((batch_size,144,144))
                    ret[key] = voxel_labels
                elif key in ["grid_ind"]:
                    voxel_labels = []
                    for d in val:
                        voxel_labels.append(d)
                    # voxel_labels = np.vstack(voxel_labels).reshape((batch_size, 496, 432, -1))
                    ret[key] = voxel_labels
                elif key in ["labels_ori"]:
                    voxel_labels = []
                    for d in val:
                        voxel_labels.append(d)
                    # voxel_labels = np.vstack(voxel_labels).reshape((batch_size, 496, 432, -1))
                    ret[key] = voxel_labels
                    # print(voxel_labels)
                    # print(voxel_labels.shape)
                    # import pdb;pdb.set_trace()
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        # import pdb;pdb.set_trace()
        ret['batch_size'] = batch_size
        # points_len=len(set(ret['points'][:,0].tolist()))
        # if batch_size != points_len or points_len != ret['gt_boxes'].shape[0]:
        #     print('points_len',points_len)
        #     print('batch_size',batch_size)
        #     return None
        return ret
