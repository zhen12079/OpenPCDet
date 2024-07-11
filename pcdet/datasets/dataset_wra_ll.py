a_ll.py# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.utils.data as torch_data

class CBGSDataset(object):
    """A wrapper of class sampled dataset with ann_file path. Implementation of
    paper `Class-balanced Grouping and Sampling for Point Cloud 3D Object
    Detection <https://arxiv.org/abs/1908.09492.>`_.

    Balance the number of scenes under different classes.

    Args:
        dataset (:obj:`CustomDataset`): The dataset to be class sampled.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.CLASSES = dataset.class_names
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.sample_indices = self._get_sample_indices()
        # self.dataset.data_infos = self.data_infos
        if hasattr(self.dataset, 'flag'):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        sample_indices = []
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}  #class_sample_idxs save all class id appear in pcd
        for idx in range(len(self.dataset)):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            if len(sample_cat_ids)==0:
                sample_indices.append(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])   #sum of all class
        class_distribution = {k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()}
        # print(len(sample_indices))
        # print(class_distribution)
        # left=len(sample_indices)%len(self.CLASSES)
        # if left!=0:
        #     le = len(sample_indices)//len(self.CLASSES)
        # else:
        #     le = len(sample_indices) // len(self.CLASSES)+1
        #     sample_indices += np.random.choice(sample_indices,left).tolist()
        # print(le,left)
        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        # print(self.cat2id.keys())
        # print(class_sample_idxs.keys())
        # print(frac,ratios)
        # import pdb;
        #pdb.set_trace()
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            np.random.shuffle(cls_inds)
            if ratio <= 1:
                sample_indices += cls_inds[:int(len(cls_inds) * ratio)]
            else:
                sample_indices += cls_inds * int(ratio//1) + cls_inds[:int(len(cls_inds) * (ratio%1))]
            # sample_indices += np.random.choice(cls_inds,
            #                                    int(len(cls_inds) *
            #                                        ratio)).tolist()
        np.random.shuffle(sample_indices)
        # print(len(self.dataset))
        # print(len(sample_indices))
        return sample_indices

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        # import ipdb
        # ipdb.set_trace()
        ori_idx = self.sample_indices[idx]
        return self.dataset[ori_idx]

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.sample_indices)
