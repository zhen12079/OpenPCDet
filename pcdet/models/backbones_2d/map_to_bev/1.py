import torch
import torch.nn as nn


class PointPillarScatter_lane(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        if self.model_cfg.get('WITH_MAX_VOXEL',False):
            self.nz_tmp = grid_size[2]
            self.nz == 1
        else:
            assert self.nz == 1

    def forward(self, batch_dict, **kwargs):
        #import pdb;pdb.set_trace()
        pillar_features, coords = batch_dict['pillar_features_lane'], batch_dict['voxel_coords_lane']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            if self.model_cfg.get('WITH_MASK',False):
                #import pdb;pdb.set_trace()
                mask = torch.zeros(
                    1,
                    self.nz * self.nx * self.ny,
                    dtype=pillar_features.dtype,
                    device=pillar_features.device)
                voxel_num_point = batch_dict['voxel_num_points_lane']
                voxel_num_point = voxel_num_point[batch_mask]
                mask[:,indices] = 1.0*voxel_num_point/48
                spatial_feature = spatial_feature * mask
                #mask[:, indices] = 1.0
                #spatial_feature = torch.cat((spatial_feature,mask))
                #self.num_bev_features +=1

            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        if not self.model_cfg.get('WITH_MAX_VOXEL',False):
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        else:
            batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features, self.nz_tmp,self.ny,
                                                                 self.nx)
            #batch_spatial_features = torch.max(batch_spatial_features,dim=2)[0]
            batch_spatial_features = torch.mean(batch_spatial_features, dim=2)
        batch_dict['spatial_features_lane'] = batch_spatial_features #[4, 64, 144, 144]
        # print(batch_spatial_features.shape)
        #import pdb;pdb.set_trace()
        return batch_dict
