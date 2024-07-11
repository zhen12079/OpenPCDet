from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
















from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # import pdb;pdb.set_trace()
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            if loss is None:
                return None, None, None

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts, hist_list_batch, lane_result_batch, all_result_dict = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, hist_list_batch, lane_result_batch, all_result_dict
        # return batch_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_all = 0
        tb_dict = {}

        if self.model_cfg.get('DENSE_HEAD', False) and "dense_head" not in self.model_cfg.NOT_TRAIN_IN_MULTI_TASK:
            loss_detect, tb_dict_detect = self.dense_head.get_loss()
            # print("loss_detect:",loss_detect)
            loss_all += loss_detect
            tb_dict.update(tb_dict_detect)

        if self.model_cfg.get('SEGMENT_HEAD', False) and "segment_head" not in self.model_cfg.NOT_TRAIN_IN_MULTI_TASK:
            loss_segment, tb_dict_segment = self.segment_head.get_loss()
            # print("loss_segment:",loss_segment)
            loss_all += loss_segment
            tb_dict.update(tb_dict_segment)

        if self.model_cfg.get('LANE_HEAD', False) and "lane_head" not in self.model_cfg.NOT_TRAIN_IN_MULTI_TASK:
            loss_lane, tb_dict_lane = self.lane_head.get_loss()
            # print("loss_lane:",loss_lane)
            loss_all += loss_lane
            tb_dict.update(tb_dict_lane)
        # print("loss_all:",loss_all.item(),"loss_detect:",loss_detect.item(),"loss_segment:",loss_segment.item(),"loss_lane:",loss_lane.item())
        tb_dict = {
            'total_loss': loss_all.item(),
            **tb_dict
        }
        loss = loss_all
        return loss, tb_dict, disp_dict
