from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import sys
import torch
import numpy as np
import cv2
from collections import defaultdict
from loguru import logger
from progress.bar import Bar

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import constants as _C
from lib.models.smpl import SMPL
from lib.models.preproc.extractor import FeatureExtractor
from lib.utils import transforms
from lib.utils.imutils import flip_kp, flip_bbox
from lib.models.preproc.detector import DetectionModel
dataset = defaultdict(list)

def normalize_image(img,width=256,height=256):
    IMG_NORM_MEAN = [0.485, 0.456, 0.406]
    IMG_NORM_STD = [0.229, 0.224, 0.225]
    img = cv2.resize(img, (width, height))
    img = img / 255.
    mean = np.array(IMG_NORM_MEAN, dtype=np.float32)
    std = np.array(IMG_NORM_STD, dtype=np.float32)
    norm_img = (img - mean) / std
    norm_img = np.transpose(norm_img, (2, 0, 1))
    return norm_img

def compute_bbox_from_keypoints(X, do_augment=False, mask=None):
    X = torch.tensor(X)
    def smooth_bbox(bb):
        import scipy.signal as signal
        smoothed = np.array([signal.medfilt(param, int(30 / 2)) for param in bb])
        return smoothed
    
    def do_augmentation(scale_factor=0.2, trans_factor=0.05):
        _scaleFactor = np.random.uniform(1.0 - scale_factor, 1.2 + scale_factor)
        _trans_x = np.random.uniform(-trans_factor, trans_factor)
        _trans_y = np.random.uniform(-trans_factor, trans_factor)
        
        return _scaleFactor, _trans_x, _trans_y
    
    if do_augment:
        scaleFactor, trans_x, trans_y = do_augmentation()
    else:
        scaleFactor, trans_x, trans_y = 1.5, 0.0, 0.0
    
    if mask is None:
        bbox = [X[:, :, 0].min(-1)[0], X[:, :, 1].min(-1)[0],
                X[:, :, 0].max(-1)[0], X[:, :, 1].max(-1)[0]]
    else:
        bbox = []
        for x, _mask in zip(X, mask):
            if _mask.sum() > 10: 
                _mask[:] = False
            _bbox = [x[~_mask, 0].min(-1)[0], x[~_mask, 1].min(-1)[0],
                    x[~_mask, 0].max(-1)[0], x[~_mask, 1].max(-1)[0]]
            bbox.append(_bbox)
        bbox = torch.tensor(bbox).T
    
    cx, cy = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
    bbox_w = bbox[2] - bbox[0]
    bbox_h = bbox[3] - bbox[1]
    bbox_size = torch.stack((bbox_w, bbox_h)).max(0)[0]*1.3
    scale = bbox_size * scaleFactor
    bbox = torch.stack((cx + trans_x * scale, cy + trans_y * scale, scale))
    
    if do_augment:
        bbox = torch.from_numpy(smooth_bbox(bbox.numpy()))
    
    return bbox.T

def get_sequence_data(humman_loader, session_name):
    """Helper function to get all data for a HuMMan sequence
    Args:
        humman_loader: HuMMan dataset loader instance
        session_name: name of the session
        view_id: camera view index (0-7)
    """
    camera_params = humman_loader.load_cameras(session_name)
    smpl_poses,global_orient,betas,transl = humman_loader.load_smpl_of_all_frame(session_name)
    smpl_poses = np.concatenate((global_orient,smpl_poses),1)
    
    if smpl_poses is None:
        return None
    
    return {
        'camera_params': camera_params,
        'smpl_poses': smpl_poses,
        'global_orient': global_orient,
        'smpl_trans': transl,
        'betas': betas
    }

@torch.no_grad()
def preprocess_humman(humman_loader, split, view_id, batch_size):
    """Preprocess HuMMan dataset for a specific view
    Args:
        humman_loader: HuMMan dataset loader instance
        split: dataset split ('train', 'validation', 'test')
        view_id: camera view index (0-7)
        batch_size: batch size for feature extraction
    """
    tt = lambda x: torch.from_numpy(x).float()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = osp.join(_C.PATHS.PARSED_DATA, f'humman_{split}_{view_id}_small_vit.pth')
    extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=batch_size)
    detector = DetectionModel('cuda')
    
    smpl = {
        'neutral': SMPL(model_path=_C.BMODEL.FLDR),
        'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'),
        'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
    }
    
    for session_name in humman_loader.session_list:
        detector.initialize_tracking()
        logger.info(f'Processing {session_name}...')
        
        # Get all data for this sequence and view
        seq_data = get_sequence_data(humman_loader, session_name)
        if seq_data is None:
            logger.warning(f'No data found for session {session_name} view {view_id}')
            continue
        if seq_data['smpl_poses'].shape[0] < 15:
            logger.warning(f'Not enough frames for session {session_name} view {view_id}')
            continue
        
        # Get camera parameters
        cam_param = seq_data['camera_params']
        camera_name=f"kinect_color_{view_id:03d}"
        extrinsics = np.eye(4)
        extrinsics[:3, :3] =np.array(cam_param[camera_name]['R'])
        extrinsics[:3, 3] = np.array(cam_param[camera_name]['T'])
        intrinsics = np.array(cam_param[camera_name]['K'])
        
        
        # Extract features
        patch_list, frame_ids = [], []
        fps=15
        length = len(seq_data["smpl_poses"])
        bar = Bar(f'Loading frames from {session_name}', fill='#', max=length)
        images = humman_loader.load_color_image_all_frame(session_name,view_id)
        width, height = images[0].shape[1], images[0].shape[0]
        for frame_idx, kp2d in enumerate(seq_data["smpl_poses"]):

                # Process image
            detector.track(images[frame_idx], fps, length)
            norm_img = normalize_image(images[frame_idx])
            patch_list.append(torch.from_numpy(norm_img).unsqueeze(0).float())
            frame_ids.append(frame_idx)
                
            bar.next()
        if not frame_ids:  # Skip if no valid frames
            continue
        tracking_results = detector.process(fps)
        bboxes = tt(tracking_results[0]['bbox'])
        # Process features in batches
        patch_list = torch.split(torch.cat(patch_list), batch_size)
        features, flipped_features = [], []
        
        for i, patch in enumerate(patch_list):
            bbox = bboxes[i*batch_size:min((i+1)*batch_size, len(frame_ids))].float().cuda()
            bbox_center = bbox[:, :2]
            bbox_scale = bbox[:, 2] / 200

            feature = extractor.model(patch.cuda(), encode=True)
            features.append(feature.cpu())
            
            flipped_feature = extractor.model(torch.flip(patch, (3,)).cuda(), encode=True)
            flipped_features.append(flipped_feature.cpu())
            
            if i == 0:
                init_patch = patch[[0]].clone()
        
        features = torch.cat(features)
        flipped_features = torch.cat(flipped_features)
        
        # Save data
        frame_ids = torch.from_numpy(np.array(frame_ids))
        poses = tt(seq_data['smpl_poses'])[frame_ids]
        bboxes = tt(tracking_results[0]['bbox'])
        dataset['vid'].append(f'{session_name}_{view_id}')
        dataset['res'].append(torch.tensor([[width, height]]).repeat(len(frame_ids), 1).float())
        dataset['pose'].append(poses)
        dataset['frame_id'].append(frame_ids)
        dataset['cam_poses'].append(tt(extrinsics).unsqueeze(0).repeat(len(frame_ids), 1, 1))
        dataset['intrinsics'].append(tt(intrinsics).unsqueeze(0).repeat(len(frame_ids), 1, 1))
        dataset['features'].append(features)
        dataset['flipped_features'].append(flipped_features)
        dataset['betas'].append(tt(seq_data['betas'])[frame_ids])
        dataset['gender'].append('neutral')
        dataset['kp2d'].append(tt(tracking_results[0]['keypoints']))
        dataset['bbox'].append(bboxes)
        # Flipped data
        dataset['flipped_bbox'].append(
            torch.from_numpy(flip_bbox(bboxes.clone().numpy(), width, height)).float()
        )
        dataset['flipped_kp2d'].append(
            torch.from_numpy(flip_kp(dataset['kp2d'][-1].clone().numpy(), width)).float()
        )
        
        # Initial predictions
        bbox = bboxes[:1].clone().cuda()
        bbox_center = bbox[:, :2].clone()
        bbox_scale = bbox[:, 2].clone() / 200
        kwargs = {
            'img_w': torch.tensor(width).repeat(1).float().cuda(),
            'img_h': torch.tensor(height).repeat(1).float().cuda(),
            'bbox_center': bbox_center,
            'bbox_scale': bbox_scale
        }

        pred_global_orient, pred_pose, pred_shape, _ = extractor.model(init_patch.cuda(), **kwargs)
        pred_output = smpl['neutral'].get_output(
            global_orient=pred_global_orient.cpu(),
            body_pose=pred_pose.cpu(),
            betas=pred_shape.cpu(),
            pose2rot=False
        )
        
        init_kp3d = pred_output.joints
        init_pose = transforms.matrix_to_axis_angle(torch.cat((pred_global_orient, pred_pose), dim=1))
        
        dataset['init_kp3d'].append(init_kp3d)
        dataset['init_pose'].append(init_pose.cpu())
        
        # Flipped initial predictions
        bbox_center[:, 0] = width - bbox_center[:, 0]
        pred_global_orient, pred_pose, pred_shape, _ = extractor.model(torch.flip(init_patch, (3,)).cuda(), **kwargs)
        pred_output = smpl['neutral'].get_output(
            global_orient=pred_global_orient.cpu(),
            body_pose=pred_pose.cpu(),
            betas=pred_shape.cpu(),
            pose2rot=False
        )
        
        init_kp3d = pred_output.joints
        init_pose = transforms.matrix_to_axis_angle(torch.cat((pred_global_orient, pred_pose), dim=1))
        
        dataset['flipped_init_kp3d'].append(init_kp3d)
        dataset['flipped_init_pose'].append(init_pose.cpu())

    # Save processed data
    torch.save(dataset, save_pth)
    logger.info(f'==> Saved preprocessed data to {save_pth}')


if __name__ == '__main__':
    import argparse
    from lib.eval.HuMMan_loader import HuMManLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, choices=['train', 'validation', 'test'], help='Dataset split')
    parser.add_argument('-v', '--view', type=int, choices=range(10), help='Camera view index (0-7)')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--base_dir', type=str, required=True, help='Path to HuMMan dataset')
    args = parser.parse_args()
    
    humman_loader = HuMManLoader(args.base_dir)
    preprocess_humman(humman_loader, args.split, args.view, args.batch_size)
