import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import unittest
import yaml, pu4c

from util import config
from util.data_util import collate_fn_limit, collation_fn_voxelmean
from functools import partial
import torch
from pprint import pprint

class TestDataset(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        pu4c.nprandom.seed(123)

    # @unittest.skip("")
    def test_semantic_kitti(self):
        from util.semantic_kitti import SemanticKITTI
        cfg_file = os.path.join('config', 'semantic_kitti', 'semantic_kitti_unet32_spherical_transformer.yaml')
        cfg = config.load_cfg_from_cfg_file(cfg_file)
        rand_idx = pu4c.nprandom.randint(1000)


        print(f"\n-------------------- semantic_kitti dataset test --------------------")
        train_set = SemanticKITTI(
            cfg.data_root, 
            voxel_size=cfg.voxel_size, 
            split='train', 
            return_ref=True, 
            label_mapping=cfg.label_mapping, 
            rotate_aug=True, 
            flip_aug=True, 
            scale_aug=True, 
            scale_params=[0.95,1.05], 
            transform_aug=True, 
            trans_std=[0.1, 0.1, 0.1],
            elastic_aug=False, 
            elastic_params=[[0.12, 0.4], [0.8, 3.2]], 
            ignore_label=cfg.ignore_label, 
            voxel_max=cfg.voxel_max, 
            xyz_norm=cfg.xyz_norm,
            pc_range=cfg.get("pc_range", None), 
            use_tta=False,
            vote_num=cfg.vote_num,
        )
        sample_data = train_set.__getitem__(index=pu4c.nprandom.randint(len(train_set)))
        coord, xyz, feat, label, info = sample_data
        metric_dict = {
            "voxel_coords": coord[rand_idx],
            "coords": xyz[rand_idx],
            "voxels": feat[rand_idx],
            "label": label[rand_idx],
        }
        pprint(metric_dict)


        print(f"\n-------------------- semantic_kitti train_loader test --------------------")
        collate_fn = partial(collate_fn_limit, max_batch_points=cfg.max_batch_points, logger=None)
        train_loader = torch.utils.data.DataLoader(train_set, 
            batch_size=2, 
            shuffle=False, 
            num_workers=cfg.workers,
            pin_memory=True, 
            sampler=None, 
            drop_last=True, 
            collate_fn=collate_fn
        )
        batch_data = next(iter(train_loader))
        coord, xyz, feat, label, offset, info = batch_data
        metric_dict = {
            "voxel_coords": coord[rand_idx],
            "coords": xyz[rand_idx],
            "voxels": feat[rand_idx],
            "label": label[rand_idx],
        }
        pprint(metric_dict)


        print(f"\n-------------------- semantic_kitti val_loader test --------------------")
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
            batch_size=2, 
            shuffle=False, 
            num_workers=cfg.workers,
            pin_memory=True, 
            sampler=None, 
            collate_fn=collation_fn_voxelmean
        )
        batch_data = next(iter(val_loader))
        coord, xyz, feat, label, offset, inds_reverse, info = batch_data
        metric_dict = {
            "voxel_coords": coord[rand_idx],
            "coords": xyz[rand_idx],
            "voxels": feat[rand_idx],
            "label": label[rand_idx],
            "inds_reverse": inds_reverse[rand_idx],
            "frame_id": info['frame_id'],
        }
        pprint(metric_dict)



if __name__ == '__main__':
    unittest.main()