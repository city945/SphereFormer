import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import unittest
import yaml, pu4c

from util import config
from util.data_util import collate_fn_limit, collation_fn_voxelmean
from functools import partial
import torch
from util.semantic_kitti import SemanticKITTI
import numpy as np
import spconv.pytorch as spconv
from util.common_util import load_params_from_file
from pprint import pprint

class TestModel(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        pu4c.nprandom.seed(123)

    # @unittest.skip("")
    def test_sphereformer_unet(self):
        cfg_file = os.path.join('config', 'semantic_kitti', 'semantic_kitti_unet32_spherical_transformer.yaml')
        cfg = config.load_cfg_from_cfg_file(cfg_file)
        ckpt = "model_zoo/download/model_semantic_kitti.pth"

        from model.unet_spherical_transformer import Semantic as Model
        cfg.patch_size = np.array([cfg.voxel_size[i] * cfg.patch_size for i in range(3)]).astype(np.float32)
        window_size = cfg.patch_size * cfg.window_size
        window_size_sphere = np.array(cfg.window_size_sphere)
        model = Model(input_c=cfg.input_c, 
            m=cfg.m,
            classes=cfg.classes, 
            block_reps=cfg.block_reps, 
            block_residual=cfg.block_residual, 
            layers=cfg.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / cfg.quant_size_scale, 
            quant_size_sphere=window_size_sphere / cfg.quant_size_scale, 
            rel_query=cfg.rel_query, 
            rel_key=cfg.rel_key, 
            rel_value=cfg.rel_value, 
            drop_path_rate=cfg.drop_path_rate, 
            window_size_scale=cfg.window_size_scale, 
            grad_checkpoint_layers=cfg.grad_checkpoint_layers, 
            sphere_layers=cfg.sphere_layers,
            a=cfg.a,
        )
        if os.path.isfile(ckpt):
            load_params_from_file(model, ckpt)
        model.cuda().eval()

        print(f"\n-------------------- sphereformer_unet val --------------------")
        val_set = SemanticKITTI(
            cfg.data_root, 
            voxel_size=cfg.voxel_size, 
            split='val', 
            rotate_aug=False, 
            flip_aug=False, 
            scale_aug=False, 
            transform_aug=False, 
            xyz_norm=cfg.xyz_norm, 
            pc_range=cfg.get("pc_range", None), 
            use_tta=False,
            vote_num=cfg.vote_num,
        )
        val_loader = torch.utils.data.DataLoader(val_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=cfg.workers,
            pin_memory=True, 
            sampler=None, 
            collate_fn=collation_fn_voxelmean
        )

        batch_data = next(iter(val_loader))
        coord, xyz, feat, target, offset, inds_reverse, info = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, cfg.batch_size)
        output = model(sinput, xyz, batch)
        output = output[inds_reverse, :]
        output = output.max(1)[1]
        metric_dict = {
            "acc": (output == target).cpu().numpy().sum() / target.shape[0],
        }
        pprint(metric_dict)



if __name__ == '__main__':
    unittest.main()

"""
        # # train_loader
        # train_set = SemanticKITTI(
        #     cfg.data_root, 
        #     voxel_size=cfg.voxel_size, 
        #     split='train', 
        #     return_ref=True, 
        #     label_mapping=cfg.label_mapping, 
        #     rotate_aug=True, 
        #     flip_aug=True, 
        #     scale_aug=True, 
        #     scale_params=[0.95,1.05], 
        #     transform_aug=True, 
        #     trans_std=[0.1, 0.1, 0.1],
        #     elastic_aug=False, 
        #     elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
        #     ignore_label=cfg.ignore_label, 
        #     voxel_max=cfg.voxel_max, 
        #     xyz_norm=cfg.xyz_norm,
        #     pc_range=cfg.get("pc_range", None), 
        #     use_tta=False,
        #     vote_num=cfg.vote_num,
        # )
        # collate_fn = partial(collate_fn_limit, max_batch_points=cfg.max_batch_points, logger=None)
        # train_loader = torch.utils.data.DataLoader(train_set, 
        #     batch_size=cfg.batch_size, 
        #     shuffle=False, 
        #     num_workers=cfg.workers,
        #     pin_memory=True, 
        #     sampler=None, 
        #     drop_last=True, 
        #     collate_fn=collate_fn
        # )

        # # train
        # batch_data = next(iter(train_loader))
        # coord, xyz, feat, target, offset, info = batch_data
        # offset_ = offset.clone()
        # offset_[1:] = offset_[1:] - offset_[:-1]
        # batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        # coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        # # coord[:, 1:] += (torch.rand(3) * 2).type_as(coord)  
        # spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        # coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        # batch = batch.cuda(non_blocking=True)
        # sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, cfg.batch_size)
        # output = model(sinput, xyz, batch)
        # output = output.max(1)[1]
"""