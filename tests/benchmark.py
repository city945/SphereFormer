# copy from https://github.com/open-mmlab/mmdetection3d/blob/main/tools/analysis_tools/benchmark.py
"""
python tests/benchmark.py config/semantic_kitti/semantic_kitti_unet32_spherical_transformer.yaml runs/semantic_kitti_unet32_spherical_transformer/20240301-075449/model/model_best.pth
python tests/benchmark.py config/nuscenes/nuscenes_unet32_spherical_transformer.yaml runs/nuscenes_unet32_spherical_transformer/20240306-120258/model/model_best.pth
python tests/benchmark.py config/waymo/waymo_unet32_spherical_transformer.yaml runs/waymo_unet32_spherical_transformer/20240303-043616/model/model_best.pth
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import argparse
import time

import torch

from util import config
from util.nuscenes import nuScenes
from util.semantic_kitti import SemanticKITTI
from util.waymo import Waymo
import numpy as np
from util.data_util import collation_fn_voxelmean
import spconv.pytorch as spconv
from util.common_util import load_params_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='benchmark a model')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--samples', default=2000, help='samples to benchmark')
    parser.add_argument('--log-interval', default=50, help='interval of logging')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    # 1. config
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.batch_size_val = 1
    cfg.workers = 0 # 似乎不影响

    # 2. dataset
    if cfg.data_name == 'nuscenes':
        val_data = nuScenes(data_path=cfg.data_root, 
            info_path_list=['nuscenes_seg_infos_1sweeps_val.pkl'], 
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
    elif cfg.data_name == 'semantic_kitti':
        val_data = SemanticKITTI(data_path=cfg.data_root, 
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
    elif cfg.data_name == 'waymo':
        val_data = Waymo(data_path=cfg.data_root, 
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
    else:
        raise ValueError("The dataset {} is not supported.".format(cfg.data_name))

    val_loader = torch.utils.data.DataLoader(val_data, 
        batch_size=cfg.batch_size_val, 
        shuffle=False, 
        num_workers=cfg.workers,
        pin_memory=True, 
        sampler=None, 
        collate_fn=collation_fn_voxelmean
    )

    # 3. model
    if cfg.arch == 'unet_spherical_transformer':
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
    else:
        raise Exception('architecture {} not supported yet'.format(cfg.arch))

    load_params_from_file(model, args.checkpoint)
    model.cuda()

    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0

    # benchmark with several samples and take the average
    for i, batch_data in enumerate(val_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        (coord, xyz, feat, target, offset, inds_reverse, info) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)
    
        coord, xyz, feat, target, offset = coord.cuda(non_blocking=True), xyz.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(feat, coord.int(), spatial_shape, cfg.batch_size_val)

        assert batch.shape[0] == feat.shape[0]
        
        with torch.no_grad():
            output = model(sinput, xyz, batch)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                      f'fps: {fps:.1f} sample / s')

        if (i + 1) == args.samples:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.1f} sample / s')
            break


if __name__ == '__main__':
    main()
