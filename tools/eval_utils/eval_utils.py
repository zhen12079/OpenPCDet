import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass

















import json
import os.path
import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

# unique_label_str = ['Driveable', 'Other-ground', 'Sidewalk', 'Lane', 'Road-edge', 'Fence', 'Car', 'Truck', 'Bus',
#                     'Pedestrian', 'Rider', 'Other-people', 'Non-motor', 'Building', 'Traffic light', 'Pole',
#                     'Traffic sign', 'Barrier', 'Vegetation', 'Tree trunk']

unique_label_str = ['Void','Road', 'Sidewalk', 'Lane', 'Road-edge', 'Fence', 'Car', 'Truck', 'Bus',
                    'Pedestrian', 'Rider', 'Non-motor', 'Traffic light', 'Pole',
                    'Traffic sign', 'Barrier', 'Vegetation']
def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
            metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def seg_val_result(hist_list, unique_label_str, logger,fp):
    iou = per_class_iu(sum(hist_list))
    # print('Validation per class iou: ')
    logger.info('Seg validation per class iou: ')
    fp.write('Seg validation per class iou: \n')
    total=""
    for class_name, class_iou in zip(unique_label_str, iou):
        # print('%s : %.2f%%' % (class_name, class_iou * 100))
        logger.info('%s : %.2f%%' % (class_name, class_iou * 100))
        fp.write('%s : %.2f%% \n' % (class_name, class_iou * 100))
        total+="%.2f "%(class_iou * 100)
    val_miou = np.nanmean(iou) * 100
    logger.info('Current val miou is %.3f' % val_miou)
    fp.write('Current val miou is %.3f \n' % val_miou)
    total +="%.3f "%val_miou
    logger.info("For excel paste:"+total)
    fp.write("For excel paste:"+total+"\n")

def lane_val_result(lane_result_total,logger,fp):
    lane_result_total["mean_f1"] = np.mean(lane_result_total["mean_f1"])
    lane_result_total["mean_prec"] = np.mean(lane_result_total["mean_prec"])
    lane_result_total["mean_recl"] = np.mean(lane_result_total["mean_recl"])
    lane_result_total["mean_acc"] = np.mean(lane_result_total["mean_acc"])

    lane_result_total["mean_f1_cls"] = np.mean(lane_result_total["mean_f1_cls"])
    lane_result_total["mean_prec_cls"] = np.mean(lane_result_total["mean_prec_cls"])
    lane_result_total["mean_recl_cls"] = np.mean(lane_result_total["mean_recl_cls"])
    lane_result_total["mean_acc_cls"] = np.mean(lane_result_total["mean_acc_cls"])

    logger.info('lane result of multiTask: ')
    fp.write('lane result of multiTask: \n')
    total=""

    for k,v in lane_result_total.items():
        logger.info('%s : %.3f%%' % (k, v*100))
        fp.write('%s : %.3f%% \n' % (k, v*100))
        total+="%.3f "%(v*100)

    logger.info("For excel paste:"+total)
    fp.write("For excel paste:"+total+"\n")

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None,
                   num_workers=4, save_json_path=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0
    result_txt_dir = result_dir / 'final_result'
    if not result_txt_dir.exists():
        result_txt_dir.mkdir(parents=True, exist_ok=True)
    fp = open(str(result_txt_dir / 'result.txt'), 'w')
    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()
    
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    from pathlib import Path
    # import pdb;pdb.set_trace()
    hist_list = []
    lane_result_total = {"mean_f1":[],"mean_acc":[],"mean_prec":[],"mean_recl":[]
                        ,"mean_f1_cls":[],"mean_acc_cls":[],"mean_prec_cls":[],"mean_recl_cls":[]}
    if not Path(result_dir / 'result.pkl').exists():
        for i, batch_dict in enumerate(dataloader):
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, ret_dict, hist_list_batch, lane_result_batch, batch_dict_total = model(batch_dict)
            disp_dict = {}
            statistics_info(cfg, ret_dict, metric, disp_dict)
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, class_names,
                output_path=final_output_dir if save_to_file else None
            )
            det_annos += annos
            if cfg.LOCAL_RANK == 0:
                progress_bar.set_postfix(disp_dict)
                progress_bar.update()
            if hist_list_batch != []:
                hist_list += hist_list_batch
            if lane_result_batch != {}:
                lane_result_total["mean_f1"].append(lane_result_batch["mean_f1"])
                lane_result_total["mean_acc"].append(lane_result_batch["mean_acc"])
                lane_result_total["mean_prec"].append(lane_result_batch["mean_prec"])
                lane_result_total["mean_recl"].append(lane_result_batch["mean_recl"])
                lane_result_total["mean_f1_cls"].append(lane_result_batch["mean_f1_cls"])
                lane_result_total["mean_acc_cls"].append(lane_result_batch["mean_acc_cls"])
                lane_result_total["mean_prec_cls"].append(lane_result_batch["mean_prec_cls"])
                lane_result_total["mean_recl_cls"].append(lane_result_batch["mean_recl_cls"])

        if hist_list != []:
            seg_val_result(hist_list, unique_label_str, logger, fp=fp)

        if lane_result_total["mean_f1"] != []:
            lane_val_result(lane_result_total, logger, fp=fp)

        if cfg.LOCAL_RANK == 0:
            progress_bar.close()

        if dist_test:
            rank, world_size = common_utils.get_dist_info()
            det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
            metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

        logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
        sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
        logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

        if cfg.LOCAL_RANK != 0:
            return {}

        ret_dict = {}
        if dist_test:
            for key, val in metric[0].items():
                for k in range(1, world_size):
                    metric[0][key] += metric[k][key]
            metric = metric[0]

        gt_num_cnt = metric['gt_num']
        for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
            logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
            ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
            ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

        total_pred_objects = 0
        for anno in det_annos:
            total_pred_objects += anno['name'].__len__()
        logger.info('Average predicted number of objects(%d samples): %.3f'
                    % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
        if save_json_path:
            if not os.path.exists(save_json_path):
                os.makedirs(save_json_path)
            print("save json...")
            for ann in tqdm.tqdm(det_annos):
                save_json_name = ann["frame_id"] + ".json"
                save_json = []
                save_json.append({})
                save_json[0]["pcd_name"] = ann["frame_id"] + ".pcb"
                save_json[0]["img_name"] = ""
                save_json[0]["object"] = []
                for i, cls in enumerate(ann["name"]):
                    object = {}
                    object["psr"] = {}
                    object["psr"]["position"] = {}
                    object["psr"]["position"]["x"] = float(ann["boxes_lidar"][i, 0])
                    object["psr"]["position"]["y"] = float(ann["boxes_lidar"][i, 1])
                    object["psr"]["position"]["z"] = float(ann["boxes_lidar"][i, 2])
                    object["psr"]["scale"] = {}
                    object["psr"]["scale"]["x"] = float(ann["boxes_lidar"][i, 3])
                    object["psr"]["scale"]["y"] = float(ann["boxes_lidar"][i, 4])
                    object["psr"]["scale"]["z"] = float(ann["boxes_lidar"][i, 5])
                    object["psr"]["rotation"] = {}
                    object["psr"]["rotation"]["x"] = 0.0
                    object["psr"]["rotation"]["y"] = 0.0
                    object["psr"]["rotation"]["z"] = float(ann["boxes_lidar"][i, 6])
                    object["obj_type"] = cls
                    object["obj_id"] = int(ann["score"][i] * 100)
                    save_json[0]["object"].append(object)
                with open(os.path.join(save_json_path, save_json_name), "w") as f:
                    json.dump(save_json, f)
            with open(result_dir / 'result.pkl', 'wb') as f:
                pickle.dump(det_annos, f)
        else:
            with open(result_dir / 'result.pkl', 'wb') as f:
                pickle.dump(det_annos, f)

    else:
        ret_dict = {}
        print("using exist result.pkl")
        with open(Path(result_dir / 'result.pkl'), 'rb') as f:
            det_annos = pickle.load(f)

    if not save_json_path:
        result_str, result_dict = dataset.evaluation(
            det_annos, class_names,
            eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
            fp=fp, num_workers=num_workers
        )

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass

