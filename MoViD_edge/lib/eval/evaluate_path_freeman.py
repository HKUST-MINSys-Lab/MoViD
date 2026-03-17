import os
import time
import os.path as osp
from glob import glob
from collections import defaultdict

import torch
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir
from lib.utils.utils import prepare_batch
from lib.utils.imutils import avg_preds

try:
    from lib.vis.renderer import Renderer
    _render = True
except:
    print("PyTorch3D is not properly installed! Cannot render the SMPL mesh")
    _render = False

m2mm = 1e3
@torch.no_grad()
def main(cfg, args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Dataloaders ========= #
    eval_loader = setup_eval_dataloader(cfg, 'freeman', 'test_c03', cfg.MODEL.BACKBONE)
    logger.info(f'Dataset loaded')
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Build SMPL models with each gender
    smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ['male', 'female', 'neutral']}
    
    pelvis_idxs = [2, 3]
    
    accumulator = defaultdict(list)
    bar = Bar('Inference', fill='#', max=len(eval_loader))
    with torch.no_grad():
        for i in range(len(eval_loader)):
            # Original batch
            batch = eval_loader.dataset.load_data(i, False)
            x, inits, features, kwargs, gt = prepare_batch(batch, cfg.DEVICE, cfg.TRAIN.STAGE=='stage2' or cfg.TRAIN.STAGE=='stage3')
            
            if cfg.FLIP_EVAL:
                flipped_batch = eval_loader.dataset.load_data(i, True)
                f_x, f_inits, f_features, f_kwargs, _ = prepare_batch(flipped_batch, cfg.DEVICE, cfg.TRAIN.STAGE=='stage2' or cfg.TRAIN.STAGE=='stage3')
            
                # Forward pass with flipped input
                flipped_pred = network(f_x, f_inits, f_features, **f_kwargs)
            
            # Forward pass with normal input
            pred = network(x, inits, features, **kwargs)
            
            if cfg.FLIP_EVAL:
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)

                # Update network predictions with merged results
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                pred = network.forward_smpl(**kwargs)

            # <======= Build predicted local motion
            pred_cam = smpl['neutral'](
                body_pose=pred['poses_body'],
                global_orient=pred['poses_root_cam'],
                betas=pred['betas'].squeeze(0),
                pose2rot=False
            )
            pred_verts_cam = pred_cam.vertices
            pred_j3d_cam = pred_cam.joints[:, :24]
            # =======>

            # <======= Build target local motion
            target_cam = smpl[batch['gender']](
                body_pose=transforms.rotation_6d_to_matrix(gt['pose'][0, :, 1:]),
                global_orient=transforms.rotation_6d_to_matrix(gt['pose'][0, :, :1]),
                betas=pred['betas'].squeeze(0),
                pose2rot=False
            )
            target_verts_cam = target_cam.vertices
            target_j3d_cam = target_cam.joints[:, :24]
            # =======>
            
            # <======= Evaluation on the local motion
            pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
                [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs
            )
            S1_hat = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
            pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
            mpjpe = torch.sqrt(((pred_j3d_cam - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
            pve = torch.sqrt(((pred_verts_cam - target_verts_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm
            accel = compute_error_accel(joints_pred=pred_j3d_cam.cpu(), joints_gt=target_j3d_cam.cpu())[1:-1]
            accel = accel * (30 ** 2)       # per frame^s to per s^2
            # =======>
            
            summary_string = f'{batch["vid"]} | PA-MPJPE: {pa_mpjpe.mean():.1f}   MPJPE: {mpjpe.mean():.1f}   PVE: {pve.mean():.1f}'
            bar.suffix = summary_string
            bar.next()
            
            # <======= Accumulate metrics
            accumulator['pa_mpjpe'].append(pa_mpjpe)
            accumulator['mpjpe'].append(mpjpe)
            accumulator['pve'].append(pve)
            accumulator['accel'].append(accel)
            # =======>
            
            # <======= (Optional) Render the prediction
            if _render and args.render:
                # Save path
                viz_pth = os.path.join('output', 'visualization')
                os.makedirs(viz_pth, exist_ok=True)
                # Build Renderer
                width, height = batch['cam_intrinsics'][0][0, :2, -1].cpu().numpy() * 2
                focal_length = batch['cam_intrinsics'][0][0, 0, 0].item()
                renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl['neutral'].faces)

                frame_list = batch['frame_id'][0].numpy()
                imname_list = sorted(glob(os.path.join(_C.PATHS.FREEMAN_DATASET_PATH, 'imageFiles', batch["vid"], '*.jpg')))
                writer = imageio.get_writer(os.path.join(viz_pth, f'{batch["vid"]}.mp4'), 
                                            mode='I', format='FFMPEG', fps=30, macro_block_size=1)

                for i, frame in enumerate(frame_list):
                    image = imageio.imread(imname_list[frame])
                    vertices = pred['verts_cam'][i] + pred['trans_cam'][[i]]
                    image = renderer.render_mesh(vertices, image)
                    writer.append_data(image)
                writer.close()
            # =======>
            
    for k, v in accumulator.items():
        accumulator[k] = np.concatenate(v).mean()

    print('')
    log_str = 'Evaluation on Freeman, '
    log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
    logger.info(log_str)
            
if __name__ == '__main__':
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    
    main(cfg, args)