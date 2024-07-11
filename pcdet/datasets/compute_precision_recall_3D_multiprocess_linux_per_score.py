"""
#####  Huang Gang ########
###### 2021 -09-20########
计算3D 检测模型前向结果的AP和Recall
Args:
result_pkl_path： 结果文件路径
pix ： 可以是pkl、也可以是txt，但是都有固定格式，格式参考kitti
"""
from tqdm import tqdm
import glob, os
import copy
from pathlib import Path
import pickle as pkl
import numpy as np
from scipy.spatial import ConvexHull
import threading
import math, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
from multiprocessing import Process, Queue
from copy import deepcopy


def get_3d_box(box_size, heading_angle, center):
    ''' Calculate 3D bounding box corners from its parameterization.

    Input:
        box_size: tuple of (l,w,h)
        heading_angle: rad scalar, clockwise from pos x axis
        center: tuple of (x,y,z)
    Output:
        corners_3d: numpy array of shape (8,3) for 3D box cornders
    '''

    def rotz(t):
        ''' Rotation about the z-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    R = rotz(heading_angle)
    l, w, h = box_size
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return round((cp2[0] - cp1[0]) * (p[1] - cp1[1]), 9) > round((cp2[1] - cp1[1]) * (p[0] - cp1[0]), 9)

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


# def poly_area(x,y):
#     """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
#     return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
#
# def convex_hull_intersection(p1, p2):
#     """ Compute area of two convex hull's intersection area.
#         p1,p2 are a list of (x,y) tuples of hull vertices.
#         return a list of (x,y) for the intersection and its volume
#     """
#     inter_p = polygon_clip(p1,p2)
#     if inter_p is not None:
#         hull_inter = ConvexHull(inter_p)
#         return inter_p, hull_inter.volume
#     else:
#         return None, 0.0

def get_2d_bbox(cx, cy, w, l, heading_angle):
    def rotz(t):
        ''' Rotation about the z-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s],
                         [s, c]])

    x_corners = [w / 2, w / 2, -w / 2, -w / 2]
    y_corners = [l / 2, -l / 2, -l / 2, l / 2]
    R = rotz(heading_angle)
    corners_2d = np.dot(R, np.vstack([x_corners, y_corners]))
    corners_2d[0, :] = corners_2d[0, :] + cx
    corners_2d[1, :] = corners_2d[1, :] + cy
    corners_2d = np.transpose(corners_2d)
    return corners_2d


def get_2d_iou(rect1, rect2):
    rect1 = [(rect1[i, 0], rect1[i, 1]) for i in range(3, -1, -1)]
    rect2 = [(rect2[i, 0], rect2[i, 1]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area / (area1 + area2 - inter_area)
    return iou_2d


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0, :] - corners[1, :]) ** 2))
    b = np.sqrt(np.sum((corners[1, :] - corners[2, :]) ** 2))
    c = np.sqrt(np.sum((corners[0, :] - corners[4, :]) ** 2))
    return a * b * c


def box3d_iou(corners1, corners2):
    # corner points are in counter clockwise order
    rect1 = [(corners1[i, 0], corners1[i, 1]) for i in range(3, -1, -1)]
    rect2 = [(corners2[i, 0], corners2[i, 1]) for i in range(3, -1, -1)]
    area1 = poly_area(np.array(rect1)[:, 0], np.array(rect1)[:, 1])
    area2 = poly_area(np.array(rect2)[:, 0], np.array(rect2)[:, 1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    # iou_2d = inter_area / (area1 + area2 - inter_area)
    ymax = min(corners1[0, 2], corners2[0, 2])
    ymin = max(corners1[4, 2], corners2[4, 2])
    inter_vol = inter_area * max(0.0, ymax - ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    iou_2d = inter_vol / (vol1)
    return iou, iou_2d


def poly_area(x, y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1, p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


class IoU():
    def __init__(self, if_use_3d_iou):
        self.if_use_3d_iou = if_use_3d_iou

    def Calculate_IOU(self, g_box, p_box):
        cornersp = get_3d_box(p_box[3:6], p_box[6], p_box[0:3])
        cornersg = get_3d_box(g_box[3:6], g_box[6], g_box[0:3])
        iou_3d, iou_2d = box3d_iou(cornersp, cornersg)
        if self.if_use_3d_iou: return iou_3d
        return iou_2d


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


def generate_prediction_dicts(idx, Calib, box_dict, class_names, output_path=None):
    """
    Args:
        batch_dict:
            frame_id:
        pred_dicts: list of pred_dicts
            pred_boxes: (N, 7), Tensor
            pred_scores: (N), Tensor
            pred_labels: (N), Tensor
        class_names:
        output_path:

    Returns:

    """

    def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
        """
        :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        :param calib:
        :return:
            boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        """
        boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
        xyz_lidar = boxes3d_lidar_copy[:, 0:3]
        l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
        r = boxes3d_lidar_copy[:, 6:7]

        xyz_lidar[:, 2] -= h.reshape(-1) / 2
        xyz_cam = calib.lidar_to_rect(xyz_lidar)
        # xyz_cam[:, 1] += h.reshape(-1) / 2
        r = -r - np.pi / 2
        return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

    def get_template_prediction(num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def generate_single_sample_dict(box_dict, Calib):
        pred_scores = box_dict['pred_scores']
        pred_boxes = box_dict['pred_boxes']
        pred_labels = box_dict['pred_labels']
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        calib = Calib
        image_shape = np.array([375, 1242])
        pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        # pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
        #     pred_boxes_camera, calib, image_shape=image_shape
        # )

        pred_dict['name'] = np.array(class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        # pred_dict['bbox'] = pred_boxes_img
        # pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        # pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['boxes_camera'] = pred_boxes_camera[:, 0:7]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes

        return pred_dict

    single_pred_dict = generate_single_sample_dict(box_dict, Calib)
    single_pred_dict['frame_id'] = idx
    annos = [single_pred_dict]

    return annos


def cal_IOU(if_use_3d_iou, iou_th):
    def _nms(gt_label_catogs, pred_label_catogs, cls):
        gt_num = gt_label_catogs.shape[0]
        pr_num = pred_label_catogs.shape[0]
        ntp = 0
        iou_pred_data_mean = 0
        heading_pred_data_mean = 0
        # print("------------------------------------------------------1")
        # print("gt_num",gt_num,"pr_num",pr_num)
        if gt_num > 0 and pr_num > 0:
            flg_pred = [0 for _ in range(pr_num)]
            iou_pred_data = [0 for _ in range(pr_num)]
            heading_pred_data = [0 for _ in range(pr_num)]
            for id_gt in range(gt_num):
                gt_box = gt_label_catogs[id_gt]
                iou_pred = [0 for _ in range(pr_num)]
                heading_pred = [0 for _ in range(pr_num)]
                for idx_pred in range(pr_num):
                    pred_box = pred_label_catogs[idx_pred]

                    cornersp = get_3d_box(pred_box[3:6], pred_box[6], pred_box[0:3])
                    cornersg = get_3d_box(gt_box[3:6], gt_box[6], gt_box[0:3])

                    iou_3d, iou_2d = box3d_iou(cornersp, cornersg)
                    iou_val = iou_3d if if_use_3d_iou else iou_2d
                    if iou_val > iou_pred[idx_pred] and iou_val >= iou_th[cls]:
                        iou_pred[idx_pred] = iou_val
                        # print("pred_box",pred_box)
                        # print("gt_box", gt_box)
                        # print("pred_box[6]",pred_box[6],gt_box[6])
                        # print(heading_pred)
                        heading_abs = np.abs(pred_box[6] - gt_box[6]) if gt_box[6] > 0 else np.abs(
                            pred_box[6] - (gt_box[6] + np.pi * 2))
                        heading_pred[idx_pred] = heading_abs if heading_abs <= np.pi else np.abs(
                            np.pi * 2 - heading_abs)
                max_pred_id = iou_pred.index(max(iou_pred))

                if iou_pred[max_pred_id] >= iou_th[cls]:
                    if iou_pred[max_pred_id] > iou_pred_data[max_pred_id]:
                        iou_pred_data[max_pred_id] = iou_pred[max_pred_id]
                        heading_pred_data[max_pred_id] = heading_pred[max_pred_id]
                    flg_pred[max_pred_id] = 1
            ntp = sum(flg_pred)
            need_iou_pred_data=[i for i in iou_pred_data if i != 0]
            need_heading_pred_data = [i for i in heading_pred_data if i != 0]
            iou_pred_data_mean = np.mean(need_iou_pred_data) if len(need_iou_pred_data)!=0 else 0
            heading_pred_data_mean = np.mean(need_heading_pred_data) if len(need_heading_pred_data)!=0 else 0
            # print("------------------------------------------------------1")
            # print("iou_pred_data_mean",iou_pred_data_mean,"heading_pred_data_mean",heading_pred_data_mean)
            # print("iou_pred_data",iou_pred_data)
            # print("heading_pred_data", heading_pred_data)
            # print("------------------------------------------------------2")

        # else:
        #     print("exist 0 sample",ntp,gt_num,pr_num)
        return ntp, gt_num, pr_num, iou_pred_data_mean, heading_pred_data_mean

    return _nms


def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).

    Args:
        recalls (np.ndarray): Recalls with shape of (num_scales, num_dets)
            or (num_dets, ).
        precisions (np.ndarray): Precisions with shape of
            (num_scales, num_dets) or (num_dets, ).
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]

    Returns:
        float or np.ndarray: Calculated average precision.
    """
    if recalls.ndim == 1:
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]

    assert recalls.shape == precisions.shape
    assert recalls.ndim == 2

    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    return ap


def eval_worker(data_queue, result_queue, iou_th, if_3DIoU):
    while True:
        k, pcd_idx, cls, g, p = data_queue.get()
        nms = cal_IOU(if_3DIoU, iou_th)
        ntp, gt_num, pr_num, iou_pred_data_mean, heading_pred_data_mean = nms(g, p, cls)
        result_queue.put((k, pcd_idx, ntp, gt_num, pr_num, iou_pred_data_mean, heading_pred_data_mean))

def evaluation_per_score(gts, det_annos, class_names, fid, iou_th=0.1, if_3DIoU=True, num_workers=2, **kwargs):
    Score_Threshold_min = {'Car': 0.3, 'Pedestrian': 0.2, 'non-motor': 0.3, 'Bus': 0.3, 'Truck': 0.3, 'barrier': 0.2}
    Score_Threshold_max = {'Car': 0.5, 'Pedestrian': 0.4, 'non-motor': 0.5, 'Bus': 0.5, 'Truck': 0.5, 'barrier': 0.4}
    Score_Threshold_stride = {}
    result = []
    split_len = 10
    for cls, score_min in Score_Threshold_min.items():
        stride = (Score_Threshold_max[cls] - Score_Threshold_min[cls]) / split_len
        Score_Threshold_stride[cls] = stride
    Score_Threshold = Score_Threshold_min.copy()
    for i in range(split_len + 1):
        det_annos_ = deepcopy(det_annos)
        sences = '\nScore Threshold:' + str(Score_Threshold)
        fid.write(sences + "\n")
        print(sences)
        filter_by_score(det_annos_, Score_Threshold)
        cur_score_total_result = evaluation_(gts, det_annos_, class_names, fid, iou_th=iou_th, if_3DIoU=if_3DIoU,
                                             num_workers=num_workers, **kwargs)
        # print(Score_Threshold_min)
        result.append(cur_score_total_result)
        for cls, score_min in Score_Threshold.items():
            Score_Threshold[cls] += Score_Threshold_stride[cls]

    aps = {}
    best_pr = {}
    iou_mul_th_mean = []
    heading_mul_th_mean = []
    sences = "NEW AP RESULT:"
    fid.write("\n" + sences + '\n')
    print(sences)
    for cls in class_names:
        p, r, ap, f1, iou_mul_th, heading_mul_th = [], [], [], [], [], []
        for re in result:
            p.append(re[cls]['precision'])
            r.append(re[cls]['recall'])
            ap.append(re[cls]['ap'])
            f1.append(re[cls]['F1'])
            iou_mul_th.append(re[cls]['iou'])
            heading_mul_th.append(re[cls]['heading'])
        best_ap_index = np.argmax(np.array(f1))
        best_precision, best_recall, best_f1, best_ap = p[best_ap_index], r[best_ap_index], f1[best_ap_index], ap[
            best_ap_index]
        bset_score = Score_Threshold_min[cls] + Score_Threshold_stride[cls] * best_ap_index
        best_pr[cls] = [best_precision, best_recall, best_f1, best_ap, bset_score]
        ap = average_precision(np.array(r), np.array(p))
        sences = "catogory {}  , AP_pr_per_thres = {:.3f}, iou_per_thres = {:.3f}, heading_per_thres = {:.3f}".format(
            cls, ap[0] * 100, np.mean(iou_mul_th), np.mean(heading_mul_th))
        fid.write(sences + '\n')
        print(sences)
        aps[cls] = ap[0]
        iou_mul_th_mean.append(np.mean(iou_mul_th))
        heading_mul_th_mean.append(np.mean(heading_mul_th))
        # import pdb;pdb.set_trace()
        # print(cls,ap)
    aps_ = []
    for k, v in aps.items():
        aps_.append(v)
    mean_ap = np.mean(aps_)
    aps["total"] = mean_ap
    sences = "mean AP_pr_per_thres = {:.3f}, mean iou_per_thres = {:.3f}, mean heading_per_thres = {:.3f}".format(
        mean_ap * 100, np.mean(iou_mul_th_mean), np.mean(heading_mul_th_mean))
    fid.write(sences + '\n')
    print(sences)

    sences = "\nbset performance:"
    fid.write(sences + '\n')
    print('\n' + sences)
    Score_Threshold_best = {}
    for k, v in best_pr.items():
        # sences = "catogory {}  , precision = {:.3f}, recall = {:.3f} F1 = {:.3f} AP = {:.3f}".format(k, v[0],v[1],v[2],v[3])
        # fid.write(sences+'\n')
        # print(sences)
        Score_Threshold_best[k] = v[-1]
    sences = "bset score:" + str(Score_Threshold_best)
    fid.write(sences + '\n')
    print(sences)
    filter_by_score(det_annos, Score_Threshold_best)
    cur_score_total_result_best = evaluation_(gts, det_annos, class_names, fid, iou_th=iou_th, if_3DIoU=if_3DIoU,
                                              num_workers=num_workers, **kwargs)

    Score_Threshold_tmp = {'Car': 0.37, 'Pedestrian': 0.12, 'non-motor': 0.4, 'Bus': 0.4, 'Truck': 0.4, 'barrier': 0.3}
    sences = "\nsample score:" + str(Score_Threshold_tmp)
    fid.write(sences + '\n')
    print(sences)
    filter_by_score(det_annos, Score_Threshold_tmp)
    cur_score_total_result_tmp = evaluation_(gts, det_annos, class_names, fid, iou_th=iou_th, if_3DIoU=if_3DIoU,
                                             num_workers=num_workers, **kwargs)

    print("main result for excel (r|p|a):")
    fid.write("\nmain result for excel (r|p|a):")
    sences = ""
    sences_name = ""
    for k, v in cur_score_total_result_best.items():
        sences_name += f"{k} "
    print(sences_name)
    fid.write("\n" + sences_name)
    for k, v in cur_score_total_result_best.items():
        sences += "{:.2f} {:.2f} {:.3f} {:.3f} {:.2f} ".format(v["recall"] * 100, v["precision"] * 100, v["iou"],
                                                              v["heading"], aps[k] * 100)
    print(sences)
    fid.write("\n" + sences+"\n")
    print("\n")

def evaluation_(gts, det_annos, class_names, fid, iou_th=0.1, if_3DIoU=True, num_workers=2, **kwargs):
    # gts: 7860 {'file_name':{'name':['Car',...},'gt_boxes':[[x,y,z,l,w,h,heading],...]}
    # det_annos: 7860 {'file_name':{'name':['Car',...},'score':[0.95346,...],'boxes_lidar':[[x,y,z,l,w,h,heading],...]}
    data_queue = Queue()
    result_queue = Queue()
    workers = []
    print("eval_num_workers", num_workers)
    # num_workers = 1

    for _ in range(num_workers):
        workers.append(Process(target=eval_worker, args=(data_queue, result_queue, iou_th, if_3DIoU)))
    for w in tqdm(workers):
        w.daemon = True
        w.start()

    iou = IoU(if_3DIoU)
    cls = len(class_names) + 1
    tp_dict, npre_dict, ngt_dict, iou_dict, heading_dict = {}, {}, {}, {}, {}
    for i in range(1, cls):
        tp_dict[i] = {}
        npre_dict[i] = {}
        ngt_dict[i] = {}
        iou_dict[i] = {}
        heading_dict[i] = {}
    pcd_idxs = list(gts.keys())
    pbar = tqdm(pcd_idxs)
    pcd_idxs_2 = []
    for pcd_idx in pbar:
        gt_boxes = gts[pcd_idx]
        if pcd_idx not in det_annos:
            continue
        pcd_idxs_2.append(pcd_idx)
        pred_boxes = det_annos[pcd_idx]
        gt_cls = np.array(gt_boxes['name'])
        # print(pcd_idx, gt_cls)
        for k, cls in enumerate(class_names):
            gt_indx = gt_cls == cls
            gt_label_catogs = gt_boxes['gt_boxes'][gt_indx, :]
            # print(type(gt_boxes['gt_boxes'][gt_indx, :]))

            pred_box = pred_boxes['name']
            pre_indx = pred_box == cls
            pred_label_catogs = pred_boxes['boxes_lidar'][pre_indx, :]  # [x,y,z,l,w,h,headings]
            data_queue.put((k, pcd_idx, cls, gt_label_catogs, pred_label_catogs))
    # import pdb;pdb.set_trace()
    for i in tqdm(pcd_idxs_2):
        for j in range(len(class_names)):
            k, pcd_idx, ntp, gt_num, pr_num, iou_pred_data_mean, heading_pred_data_mean = result_queue.get()
            tp_dict[k + 1][pcd_idx] = ntp
            npre_dict[k + 1][pcd_idx] = pr_num
            ngt_dict[k + 1][pcd_idx] = gt_num
            iou_dict[k + 1][pcd_idx] = iou_pred_data_mean
            heading_dict[k + 1][pcd_idx] = heading_pred_data_mean
    cur_score_total_result = {}
    tp, pre, gt = 0, 0, 0
    iou_all, heading_all = [], []
    for k, cls in enumerate(class_names):
        cur_score_total_result[cls] = {}
        # import pdb;pdb.set_trace()
        ntp, npre, ngt = 0, 0, 0
        tp_cat, pre_cat, gt_cat, iou_cat, heading_cat = tp_dict[k + 1], npre_dict[k + 1], ngt_dict[k + 1], iou_dict[
            k + 1], heading_dict[k + 1]
        # ntp = sum([v for k, v in tp_cat.items()])
        # npr = sum([v for k, v in pre_cat.items()])
        # ngt = sum([v for k, v in gt_cat.items()])
        # ntp = np.cumsum(np.array([v for k, v in tp_cat.items()]))
        # npr = np.cumsum(np.array([v for k, v in pre_cat.items()]))
        # ngt = np.cumsum(np.array([v for k, v in gt_cat.items()]))
        tp_temp, pre_temp, gt_temp, iou_temp, heading_temp = [], [], [], [], []
        for idx in pcd_idxs_2:
            tp_temp.append(tp_cat[idx])
            pre_temp.append(pre_cat[idx])
            gt_temp.append(gt_cat[idx])
            if iou_cat[idx]!=0:
                iou_temp.append(iou_cat[idx])
            if heading_cat[idx]!=0:
                heading_temp.append(heading_cat[idx])
        if len(iou_temp)==0:#barrier 80-120 often happend
            iou_temp=[0]
        if len(heading_temp)==0:#barrier 80-120 often happend
            heading_temp=[0]
        iou_mean = np.mean(iou_temp)
        heading_mean = np.mean(heading_temp)
        iou_all += iou_temp
        heading_all += heading_temp
        tp_temp = np.array(tp_temp)
        pre_temp = np.array(pre_temp)
        gt_temp = np.array(gt_temp)
        ntp = np.cumsum(tp_temp)
        npr = np.cumsum(pre_temp)
        ngt = np.cumsum(gt_temp)
        tp = tp + ntp
        pre = pre + npr
        gt = gt + ngt
        # npr = 1 if npr[-1] == 0 else npr[-1]
        # ngt = 1 if ngt[-1] == 0 else ngt[-1]
        recall = ntp / (ngt + 0.000001)
        precision = ntp / (npr + 0.000001)
        # import pdb;pdb.set_trace()
        F1 = 2 * (recall[-1] * precision[-1]) / (precision[-1] + recall[-1] + 0.000001)
        ap = average_precision(recall, precision)
        cur_score_total_result[cls]['precision'] = precision[-1]
        cur_score_total_result[cls]['recall'] = recall[-1]
        cur_score_total_result[cls]['F1'] = F1
        cur_score_total_result[cls]['ap'] = ap[0]
        cur_score_total_result[cls]['iou'] = iou_mean
        cur_score_total_result[cls]['heading'] = heading_mean
        sences = "catogory {}  , precision = {:.3f}, recall = {:.3f}, iou = {:.3f}, heading = {:.3f} F1 = {:.3f} AP = {:.3f}".format(
            cls, precision[
                     -1] * 100, recall[-1] * 100, iou_mean, heading_mean, F1 * 100, ap[0] * 100)
        fid.write(sences + '\n')
        print(sences)

    recall_all = tp / (gt + 0.000001)
    precesion_all = tp / (pre + 0.000001)
    F1_all = 2 * (recall_all[-1] * precesion_all[-1]) / (precesion_all[-1] + recall_all[-1] + 0.000001)
    ap_all = average_precision(recall_all, precesion_all)
    iou_all_mean = np.mean(iou_all)
    heading_all_mean = np.mean(heading_all)
    sences = "over all, precision = {:.3f}, recall = {:.3f}, iou = {:.3f}, heading = {:.3f} F1 = {:.3f} AP={:.3f}".format(
        precesion_all[-1] * 100,
        recall_all[-1] * 100, iou_all_mean, heading_all_mean,
        F1_all * 100, ap_all[0] * 100)
    cur_score_total_result['total'] = {}
    cur_score_total_result['total']['precision'] = precesion_all[-1]
    cur_score_total_result['total']['recall'] = recall_all[-1]
    cur_score_total_result['total']['F1'] = F1_all
    cur_score_total_result['total']['iou'] = iou_all_mean
    cur_score_total_result['total']['heading'] = heading_all_mean
    print(sences + '\n')
    fid.write(sences + '\n')
    return cur_score_total_result


def get_calib(idx, calib_file):
    p = calib_file + str(idx) + '.txt'
    calib_file = Path(p)
    assert calib_file.exists(), calib_file
    return Calibration(calib_file)


# def gt_hash(gt_lits, if_gt=True):
#     ret = {}
#     if if_gt:
#         for l in gt_lits:
#             key = l['point_cloud_info']['lidar_idx']
#             ret[key] = {}
#             ret[key]['name'] = np.array(l['annos']['name'])
#             ret[key]['gt_boxes'] = l['annos']['gt_boxes']
#     else:
#         for l in gt_lits:
#             key = l['frame_id']
#             ret[key] = l
#     return ret

def gt_hash(gt_lits, if_gt=True, add_offset=False):
    ret = {}
    if if_gt:
        for l in gt_lits:
            key = l['point_cloud_info']['lidar_idx']
            ret[key] = {}
            ret[key]['name'] = np.array(l['annos']['name'])
            ret[key]['gt_boxes'] = l['annos']['gt_boxes']
            if add_offset and len(ret[key]['gt_boxes'])!=0 and len(ret[key]['gt_boxes'][0])!=0:
                # print(ret[key]['gt_boxes'])
                ret[key]['gt_boxes'][:,0] += 1.75
                ret[key]['gt_boxes'][:,2] += 1.25
    else:
        for l in gt_lits:
            key = l['frame_id']
            ret[key] = l
    return ret

def get_pre_by_pkl(det_annos, result_path, class_names, gt_pkl):
    # det_annos ={}
    calib_dir = ''.join(gt_pkl.split('object')[:-1]) + 'object\\training\\calib\\'
    with open(result_path, 'rb') as f:
        pres = pkl.load(f)
    for val in pres:
        idx = val['frame_id']
        val['boxes_camera'] = np.hstack([val['location'], val['dimensions'], val['rotation_y'][:, np.newaxis]])
        det_annos[idx] = val
        print('yes!')


def filter_by_point_range(range, gt, pre):
    # print(len(gt),len((pre)))
    for k, v in gt.items():
        # print(len(v['name']))
        if len(v['name']) == 0: continue
        box_mask_x = (v['gt_boxes'][:, 0] >= range[0]) * (
                v['gt_boxes'][:, 0] <= range[3])
        box_mask_y = (v['gt_boxes'][:, 1] >= range[1]) * (
                v['gt_boxes'][:, 1] <= range[4])
        box_mask_z = (v['gt_boxes'][:, 2] >= range[2]) * (
                v['gt_boxes'][:, 2] <= range[5])
        mask = box_mask_x * box_mask_y * box_mask_z
        for h, g in v.items():
            if h == 'name':
                gt[k][h] = g[mask]
            else:
                gt[k][h] = gt[k][h][mask, :]
    for k, v in pre.items():
        if len(v['name']) == 0: continue
        box_mask_x = (v['boxes_lidar'][:, 0] >= range[0]) * (
                v['boxes_lidar'][:, 0] <= range[3])
        box_mask_y = (v['boxes_lidar'][:, 1] >= range[1]) * (
                v['boxes_lidar'][:, 1] <= range[4])
        box_mask_z = (v['boxes_lidar'][:, 2] >= range[2]) * (
                v['boxes_lidar'][:, 2] <= range[5])
        mask = box_mask_x * box_mask_y * box_mask_z
        for h, g in v.items():
            if h == 'name' or h == 'score':
                pre[k][h] = pre[k][h][mask]
            if h == 'boxes_lidar':
                pre[k][h] = pre[k][h][mask, :]


def filter_by_other(gt, pre):
    for k, v in gt.items():
        red_gt_boxes = v['gt_boxes']
        if k not in pre: continue
        pred_boxes = pre[k]['boxes_lidar']
        mask = np.ones(red_gt_boxes.shape[0])
        mask_1 = np.ones(pred_boxes.shape[0])
        if v['name'].shape[0] == 0: continue
        # print(v['name'].shape[0])
        for m in range(v['name'].shape[0]):
            if v['name'][m] != 'Other':
                continue
            else:
                other_boxes = v['gt_boxes'][m, :]
                cornersg = get_3d_box(other_boxes[3:6], other_boxes[6], other_boxes[0:3])
                for i in range(red_gt_boxes.shape[0]):
                    if i == m:
                        mask[i] = True
                        continue
                    import pdb;
                    # pdb.set_trace()
                    cornersp = get_3d_box(red_gt_boxes[i, 3:6], red_gt_boxes[i, 6], red_gt_boxes[i, 0:3])
                    iou_3d, iou_2d = box3d_iou(cornersp, cornersg)
                    # print("####1",iou_2d)
                    if iou_2d < 0.9:
                        mask[i] = True
                    else:
                        mask[i] = False

                for i in range(pred_boxes.shape[0]):
                    cornersp = get_3d_box(pred_boxes[i, 3:6], pred_boxes[i, 6], pred_boxes[i, 0:3])
                    iou_3d, iou_2d = box3d_iou(cornersp, cornersg)
                    # print("####2",iou_2d)
                    if iou_2d < 0.9:
                        mask_1[i] = True
                    else:
                        mask_1[i] = False
        mask = np.array(mask, dtype=bool)
        for h, g in v.items():
            if h == 'name':
                gt[k][h] = g[mask]
            else:
                gt[k][h] = gt[k][h][mask, :]
        mask_1 = np.array(mask_1, dtype=bool)
        for h, g in pre[k].items():
            if h == 'name' or h == 'score':
                pre[k][h] = g[mask_1]
            elif h == 'boxes_lidar':
                pre[k][h] = pre[k][h][mask_1, :]


def divide_by_distance(dis1, dis2, gt, pre):
    if dis1 >= dis2:
        k = dis1
        dis1 = dis2
        dis2 = k
    a = -dis1 * dis1 if dis1 < 0 else dis1 * dis1
    b = dis2 * dis2
    sub_gt, sub_pr = {}, {}
    for k, v in gt.items():
        sub_gt[k] = {}
        if len(v['name']) == 0:
            sub_gt[k]['name'] = np.array([])
            sub_gt[k]['gt_boxes'] = np.array([])
            continue
        box_dis = np.power(v['gt_boxes'][:, 0], 2) + np.power(v['gt_boxes'][:, 1], 2)
        mask1 = box_dis <= b
        mask2 = box_dis > a
        mask = mask1 * mask2
        sub_gt[k]['name'] = v['name'][mask]
        sub_gt[k]['gt_boxes'] = v['gt_boxes'][mask, :]

    for k, v in pre.items():
        sub_pr[k] = {}
        if len(v['name']) == 0:
            sub_pr[k]['name'] = np.array([])
            sub_pr[k]['score'] = np.array([])
            sub_pr[k]['boxes_lidar'] = np.array([])
            continue
        box_dis = np.power(v['boxes_lidar'][:, 0], 2) + np.power(v['boxes_lidar'][:, 1], 2)
        mask1 = box_dis <= b
        mask2 = box_dis > a
        mask = mask1 * mask2
        sub_pr[k]['name'] = pre[k]['name'][mask]
        sub_pr[k]['score'] = pre[k]['score'][mask]
        sub_pr[k]['boxes_lidar'] = pre[k]['boxes_lidar'][mask, :]
    return sub_gt, sub_pr


def count_nums(boxes):
    cc, nc, pc, bc, tc, bac = 0, 0, 0, 0, 0, 0
    for k, v in boxes.items():
        cls = list(v['name'])
        cc += cls.count('Car')
        nc += cls.count('non-motor')
        pc += cls.count('Pedestrian')
        bc += cls.count('Bus')
        tc += cls.count('Truck')
        bac += cls.count('barrier')
    print("car:{}, non-motor:{}, Pedestrian:{} ,Bus:{}, truck:{}, barrier:{}".format(cc, nc, pc, bc, tc, bac))


def filter_by_score(pred_boxes, Threshold):
    for k, v in pred_boxes.items():
        if v['name'].shape[0] == 0:
            continue
        if isinstance(Threshold, dict):
            score_mask = []
            for name, score in zip(v['name'], v['score']):
                if score >= Threshold[name]:
                    score_mask.append(True)
                else:
                    score_mask.append(False)
        else:
            score_mask = v['score'] >= Threshold
        score_mask = np.array(score_mask)
        pred_boxes[k]['score'] = v['score'][score_mask]
        pred_boxes[k]['name'] = v['name'][score_mask]
        pred_boxes[k]['boxes_lidar'] = v['boxes_lidar'][score_mask, :]


if __name__ == '__main__':
    class_names = ['Car', 'Pedestrian', 'non-motor', 'Bus', 'Truck']
    IoU_Threshold = {'Car': 0.1, 'Pedestrian': 0.1, 'non-motor': 0.1, 'Bus': 0.1, 'Truck': 0.1}
    Score_Threshold = 0.1
    distance = [0, 120]
    if_3DIoU = True
    eval_by_pkl = True  # False
    POINT_CLOUD_RANGE = [0, -62, -6, 108, 62, 10]
    use_multiprocess = False
    ###如过前向结果为txt文件保存，需指明如下路径，并且eval_by_pkl设置为False
    pix = "/*.txt"
    pre_txt = r"E:\workspace\openpcdet\data\test_kitti_200\py_infer2"
    pre_txt = r"E:\workspace\py_infer2"
    result_dir = Path('E:\workspace\openpcdet\data\test_kitti_200\testing\kitti_label')

    ###如过前向结果为pkl文件保存，需指明如下路径，并且eval_by_pkl设置为True
    result_pkl_path = "/home/workspace/openpcdet_mmdet/outputs/2022_2_17/model/checkpoint_epoch_80/result_ori_pth.pkl"
    output = result_pkl_path.replace('pkl', 'txt')
    ###加载GT
    gt_pkl = r'/home/data/leap_data/v2.0_split_leap_infos_val_clean.pkl'
    assert os.path.exists(gt_pkl), "GT pkl file not exists, Please check gt path!!!"
    with open(gt_pkl, 'rb') as f:
        gt = pkl.load(f)
        gt = gt_hash(gt)

    ####set iou class end numworker
    num_worker = 0
    iou = IoU(if_3DIoU)
    cls = len(class_names) + 1

    if eval_by_pkl:
        with open(result_pkl_path, 'rb') as fid:
            det_annos = pkl.load(fid)
            det_annos = gt_hash(det_annos, if_gt=False)

    print("starting to do eval!")
    filter_by_point_range(POINT_CLOUD_RANGE, gt, det_annos)
    fid = open(output, 'a')
    fid.write('\n' + result_pkl_path + ' eval !!!\n')
    fid.write('IoU_Threshold: {}\n'.format(IoU_Threshold))
    fid.close()
    for dis in range(1, len(distance)):
        fid = open(output, 'a')
        fid.write('\n{}m --- {}m:\n'.format(distance[dis - 1], distance[dis]))
        fid.close()
        print('{}m --- {}m,eval ...'.format(distance[dis - 1], distance[dis]))
        sub_gt, sub_pr = divide_by_distance(distance[dis - 1], distance[dis], gt, det_annos)
        filter_by_score(sub_pr, Score_Threshold)
        count_nums(sub_gt)
        count_nums(sub_pr)
        s = time.time()
        evaluation(sub_gt, sub_pr, class_names, output_path=output, iou_th=IoU_Threshold, if_3DIoU=if_3DIoU)
        e = time.time()
        print(e - s)
