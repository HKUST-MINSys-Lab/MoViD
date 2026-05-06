import os
import argparse
import os.path as osp
from glob import glob
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from progress.bar import Bar

from configs.config import get_cfg_defaults, resolve_cfg_paths
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

REPO_ROOT = Path(__file__).resolve().parent


def _resolve_cli_path(path_value):
    if not path_value:
        return path_value

    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    for root in (Path.cwd(), REPO_ROOT):
        candidate = (root / path).resolve()
        if candidate.exists():
            return str(candidate)

    return path_value


def _prepare_runtime_cfg(cfg):
    cfg = resolve_cfg_paths(cfg)
    if str(cfg.DEVICE).startswith('cuda') and not torch.cuda.is_available():
        cfg = cfg.clone()
        logger.warning('CUDA was requested but is not available. Falling back to CPU.')
        cfg.DEVICE = 'cpu'
    return cfg


def _log_device_info(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        device_index = 0
        if ':' in str(device):
            try:
                device_index = int(str(device).split(':', 1)[1])
            except ValueError:
                device_index = 0
        logger.info(f'GPU name -> {torch.cuda.get_device_name(device_index)}')
        logger.info(f'GPU feat -> {torch.cuda.get_device_properties(device_index)}')
    else:
        logger.info(f'Running on device -> {device}')


def _extract_state_dict(checkpoint):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint.get('model', checkpoint)
    return {k: v for k, v in state_dict.items() if not k.startswith('smpl.')}


def _load_network_checkpoint(network, checkpoint_path, device, label):
    checkpoint_path = _resolve_cli_path(checkpoint_path)
    if not checkpoint_path or not osp.exists(checkpoint_path):
        raise FileNotFoundError(f'{label} checkpoint not found: {checkpoint_path}')

    logger.info(f'Loading {label} model from: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    model_state_dict = network.state_dict()
    compatible_state_dict = {}
    skipped_shape_keys = []
    for k, v in state_dict.items():
        if k not in model_state_dict:
            continue
        if v.shape == model_state_dict[k].shape:
            compatible_state_dict[k] = v
        else:
            skipped_shape_keys.append(k)

    missing_keys, unexpected_keys = network.load_state_dict(compatible_state_dict, strict=False)
    if skipped_shape_keys:
        logger.warning(f'{label} checkpoint skipped {len(skipped_shape_keys)} shape-mismatched keys')
        logger.debug(f'{label} first shape-mismatched keys: {skipped_shape_keys[:5]}')
    if missing_keys:
        logger.warning(f'{label} checkpoint missing {len(missing_keys)} keys')
    if unexpected_keys:
        logger.warning(f'{label} checkpoint has {len(unexpected_keys)} unexpected keys')
    network.eval()
    logger.info(f'Loaded {len(compatible_state_dict)} compatible parameters for {label} model')
    return checkpoint_path


def align_by_pelvis(joints, pelvis_idxs=[2, 3]):
    """
    Align joints by pelvis (root alignment)
    Args:
        joints: (N, J, 3) joints
        pelvis_idxs: indices of pelvis joints
    Returns:
        aligned_joints: (N, J, 3) pelvis-aligned joints
    """
    pelvis = joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    return joints - pelvis


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (N, 3) closest to a set of 3D points S2 (N, 3),
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S1.shape[0] == S2.shape[0], (S1.shape, S2.shape)

    # 1. Remove mean
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform S1
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_mpjpe(pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Mean Per Joint Position Error (MPJPE) after pelvis alignment
    Args:
        pred_joints: (N, J, 3) predicted 3D joints
        gt_joints: (N, J, 3) ground truth 3D joints
        pelvis_idxs: indices of pelvis joints for alignment
    Returns:
        mpjpe: mean per joint position error in mm
    """
    assert pred_joints.shape == gt_joints.shape
    
    # Align by pelvis
    pred_aligned = align_by_pelvis(pred_joints, pelvis_idxs)
    gt_aligned = align_by_pelvis(gt_joints, pelvis_idxs)
    
    # Compute MPJPE
    error = np.sqrt(np.sum((pred_aligned - gt_aligned) ** 2, axis=-1))  # (N, J)
    mpjpe = np.mean(error) * 1000  # convert to mm
    return mpjpe


def compute_pa_mpjpe(pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE)
    Args:
        pred_joints: (N, J, 3) predicted 3D joints
        gt_joints: (N, J, 3) ground truth 3D joints
        pelvis_idxs: indices of pelvis joints for alignment
    Returns:
        pa_mpjpe: procrustes aligned mean per joint position error in mm
    """
    assert pred_joints.shape == gt_joints.shape
    
    # Align by pelvis first
    pred_aligned = align_by_pelvis(pred_joints, pelvis_idxs)
    gt_aligned = align_by_pelvis(gt_joints, pelvis_idxs)
    
    N, J, _ = pred_aligned.shape
    errors = []
    
    for i in range(N):
        # Apply Procrustes alignment for each frame
        pred_proc = compute_similarity_transform(pred_aligned[i], gt_aligned[i])
        
        # Compute error
        error = np.sqrt(np.sum((pred_proc - gt_aligned[i]) ** 2, axis=-1))
        errors.append(np.mean(error))
    
    pa_mpjpe = np.mean(errors) * 1000  # convert to mm
    return pa_mpjpe


def compute_pve(pred_verts, gt_verts, pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Per Vertex Error (PVE) after pelvis alignment
    Args:
        pred_verts: (N, V, 3) predicted vertices
        gt_verts: (N, V, 3) ground truth vertices
        pred_joints: (N, J, 3) predicted joints (for pelvis alignment)
        gt_joints: (N, J, 3) ground truth joints (for pelvis alignment)
        pelvis_idxs: indices of pelvis joints
    Returns:
        pve: mean per vertex error in mm
    """
    assert pred_verts.shape == gt_verts.shape
    assert pred_joints.shape == gt_joints.shape
    
    # Compute pelvis position from joints (same as MPJPE)
    pred_pelvis = pred_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    gt_pelvis = gt_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    
    # Align vertices using pelvis from joints
    pred_verts_aligned = pred_verts - pred_pelvis
    gt_verts_aligned = gt_verts - gt_pelvis
    
    # Compute PVE
    error = np.sqrt(np.sum((pred_verts_aligned - gt_verts_aligned) ** 2, axis=-1))  # (N, V)
    pve = np.mean(error) * 1000  # convert to mm
    return pve


def run_inference(cfg,
                  video,
                  output_pth,
                  network,
                  dataset,
                  model_name='model',
                  run_smplify=False):
    """
    Run inference on a single model
    """
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    logger.info(f'Running inference with {model_name}...')
    
    for subj in range(n_subjs):
        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, None, inits, features, mask=mask, init_root=init_root, 
                                      cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, None, inits, features, mask=mask, init_root=init_root, 
                              cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Merge two predictions
                flipped_pose, flipped_shape = flipped_pred['pose'].squeeze(0), flipped_pred['betas'].squeeze(0)
                pose, shape = pred['pose'].squeeze(0), pred['betas'].squeeze(0)
                flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
                avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                avg_pose = avg_pose.reshape(-1, 144)
                avg_contact = (flipped_pred['contact'][..., [2, 3, 0, 1]] + pred['contact']) / 2
                
                # Refine trajectory with merged prediction
                network.pred_pose = avg_pose.view_as(network.pred_pose)
                network.pred_shape = avg_shape.view_as(network.pred_shape)
                network.pred_contact = avg_contact.view_as(network.pred_contact)
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, None, inits, features, mask=mask, init_root=init_root, 
                              cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        if run_smplify:
            smplify = TemporalSMPLify(network.smpl, img_w=dataset.width, img_h=dataset.height, device=cfg.DEVICE)
            input_keypoints = dataset.tracking_results[_id]['keypoints']
            pred = smplify.fit(pred, input_keypoints, **kwargs)
            
            with torch.no_grad():
                network.pred_pose = pred['pose']
                network.pred_shape = pred['betas']
                network.pred_cam = pred['cam']
                output = network.forward_smpl(**kwargs)
                pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        
        # ========= Store results ========= #
        pred_body_pose = matrix_to_axis_angle(pred['poses_body']).cpu().numpy().reshape(-1, 69)
        pred_root = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)
        pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()
        
        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['pose_world'] = pred_pose_world
        results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
        results[_id]['joints2d'] = pred['joints2d'].cpu().numpy()
        results[_id]['joints3d'] = pred['joints3d'].cpu().numpy()
    
    return results


def evaluate_models(gt_results, pred_results):
    """
    Evaluate prediction against ground truth
    """
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    all_mpjpe = []
    all_pa_mpjpe = []
    all_pve = []
    
    # Pelvis joint indices
    pelvis_idxs = [2, 3]
    
    for subj_id in gt_results.keys():
        if subj_id not in pred_results:
            logger.warning(f"Subject {subj_id} not found in predictions, skipping...")
            continue
        
        gt_joints = gt_results[subj_id]['joints3d']
        pred_joints = pred_results[subj_id]['joints3d']
        
        gt_verts = gt_results[subj_id]['verts']
        pred_verts = pred_results[subj_id]['verts']
        
        # Ensure same number of frames
        min_frames = min(len(gt_joints), len(pred_joints))
        gt_joints = gt_joints[:min_frames]
        pred_joints = pred_joints[:min_frames]
        gt_verts = gt_verts[:min_frames]
        pred_verts = pred_verts[:min_frames]
        
        # Compute metrics with consistent pelvis alignment
        mpjpe = compute_mpjpe(pred_joints, gt_joints, pelvis_idxs)
        pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints, pelvis_idxs)
        pve = compute_pve(pred_verts, gt_verts, pred_joints, gt_joints, pelvis_idxs)
        
        all_mpjpe.append(mpjpe)
        all_pa_mpjpe.append(pa_mpjpe)
        all_pve.append(pve)
        
        logger.info(f"\nSubject {subj_id}:")
        logger.info(f"  MPJPE:    {mpjpe:.2f} mm")
        logger.info(f"  PA-MPJPE: {pa_mpjpe:.2f} mm")
        logger.info(f"  PVE:      {pve:.2f} mm")
    
    # Overall metrics
    logger.info("\n" + "="*50)
    logger.info("OVERALL METRICS:")
    logger.info(f"  Average MPJPE:    {np.mean(all_mpjpe):.2f} mm")
    logger.info(f"  Average PA-MPJPE: {np.mean(all_pa_mpjpe):.2f} mm")
    logger.info(f"  Average PVE:      {np.mean(all_pve):.2f} mm")
    logger.info("="*50 + "\n")
    
    return {
        'mpjpe': np.mean(all_mpjpe),
        'pa_mpjpe': np.mean(all_pa_mpjpe),
        'pve': np.mean(all_pve),
        'per_subject': {
            'mpjpe': all_mpjpe,
            'pa_mpjpe': all_pa_mpjpe,
            'pve': all_pve
        }
    }


def run_single_video(cfg,
                     video,
                     output_pth,
                     network_gt,
                     network_pred,
                     calib=None,
                     run_global=True,
                     save_pkl=False,
                     visualize=False,
                     run_smplify=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Failed to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Preprocess
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and 
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            
            if run_global: 
                slam = SLAMModel(video, output_pth, width, height, calib)
            else: 
                slam = None
            
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break
                
                # 2D detection and tracking
                detector.track(img, fps, length)
                
                # SLAM
                if slam is not None: 
                    slam.track()
                
                bar.next()

            tracking_results = detector.process(fps)
            
            if slam is not None: 
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion
        
            # Extract image features
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        
        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth}! Load the data.')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # ========= Run GT Model ========= #
    logger.info("\n" + "="*50)
    logger.info("Running Ground Truth Model...")
    logger.info("="*50)
    gt_results = run_inference(cfg, video, output_pth, network_gt, dataset, 
                                model_name='Ground Truth', run_smplify=run_smplify)
    
    # ========= Run Prediction Model ========= #
    logger.info("\n" + "="*50)
    logger.info("Running Prediction Model...")
    logger.info("="*50)
    pred_results = run_inference(cfg, video, output_pth, network_pred, dataset, 
                                  model_name='Prediction', run_smplify=run_smplify)
    
    # ========= Evaluate ========= #
    metrics = evaluate_models(gt_results, pred_results)
    
    # ========= Save Results ========= #
    if save_pkl:
        joblib.dump(gt_results, osp.join(output_pth, "gt_output.pkl"))
        joblib.dump(pred_results, osp.join(output_pth, "pred_output.pkl"))
        joblib.dump(metrics, osp.join(output_pth, "evaluation_metrics.pkl"))
        logger.info(f'Saved results to {output_pth}')
     
    # Visualize
    if visualize:
        from lib.vis.run_vis import run_vis_on_demo, run_skeleton_vis
        
        # Create visualization directories
        gt_vis_path = osp.join(output_pth, 'gt_vis')
        pred_vis_path = osp.join(output_pth, 'pred_vis')
        os.makedirs(gt_vis_path, exist_ok=True)
        os.makedirs(pred_vis_path, exist_ok=True)
        
        logger.info("Visualizing Ground Truth model results...")
        run_skeleton_vis(cfg, video, gt_results, gt_vis_path, 
                        network_gt.smpl, vis_global=run_global)
        
        logger.info("Visualizing Prediction model results...")
        run_skeleton_vis(cfg, video, pred_results, pred_vis_path, 
                        network_pred.smpl, vis_global=run_global)
        
    return gt_results, pred_results, metrics


def find_videos(folder_path, extensions=['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']):
    """
    Find all video files in a folder
    """
    video_files = []
    for ext in extensions:
        video_files.extend(glob(osp.join(folder_path, f'*{ext}')))
    return sorted(video_files)


def process_folders(cfg, folders, output_base, gt_checkpoint, pred_checkpoint, 
                   network_gt, network_pred, args):
    """
    Process all videos in multiple folders
    """
    all_results = []
    summary_metrics = {
        'folder': [],
        'video_name': [],
        'mpjpe': [],
        'pa_mpjpe': [],
        'pve': [],
        'status': []
    }
    
    for folder in folders:
        folder_name = osp.basename(folder.rstrip('/'))
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing folder: {folder_name}")
        logger.info(f"{'='*60}")
        
        # Find all videos in this folder
        videos = find_videos(folder)
        
        if len(videos) == 0:
            logger.warning(f"No videos found in {folder}")
            continue
        
        logger.info(f"Found {len(videos)} videos in {folder_name}")
        
        # Process each video
        for idx, video_path in enumerate(videos, 1):
            video_name = osp.basename(video_path)
            video_name_no_ext = osp.splitext(video_name)[0]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{idx}/{len(videos)}] Processing: {video_name}")
            logger.info(f"{'='*60}")
            
            # Create output folder for this video
            output_pth = osp.join(output_base, folder_name, video_name_no_ext)
            os.makedirs(output_pth, exist_ok=True)
            
            try:
                # Run evaluation
                gt_results, pred_results, metrics = run_single_video(
                    cfg,
                    video_path,
                    output_pth,
                    network_gt,
                    network_pred,
                    args.calib,
                    run_global=not args.estimate_local_only,
                    save_pkl=True,  # Always save pkl
                    visualize=args.visualize,
                    run_smplify=args.run_smplify
                )
                
                # Save metrics to JSON for easy reading
                metrics_json = {
                    'folder': folder_name,
                    'video': video_name,
                    'mpjpe': float(metrics['mpjpe']),
                    'pa_mpjpe': float(metrics['pa_mpjpe']),
                    'pve': float(metrics['pve']),
                    'per_subject_mpjpe': [float(x) for x in metrics['per_subject']['mpjpe']],
                    'per_subject_pa_mpjpe': [float(x) for x in metrics['per_subject']['pa_mpjpe']],
                    'per_subject_pve': [float(x) for x in metrics['per_subject']['pve']]
                }
                
                with open(osp.join(output_pth, 'metrics.json'), 'w') as f:
                    json.dump(metrics_json, f, indent=4)
                
                # Add to summary
                summary_metrics['folder'].append(folder_name)
                summary_metrics['video_name'].append(video_name)
                summary_metrics['mpjpe'].append(metrics['mpjpe'])
                summary_metrics['pa_mpjpe'].append(metrics['pa_mpjpe'])
                summary_metrics['pve'].append(metrics['pve'])
                summary_metrics['status'].append('Success')
                
                logger.info(f"✓ Successfully processed {video_name}")
                logger.info(f"  MPJPE: {metrics['mpjpe']:.2f} mm")
                logger.info(f"  PA-MPJPE: {metrics['pa_mpjpe']:.2f} mm")
                logger.info(f"  PVE: {metrics['pve']:.2f} mm")
                
            except Exception as e:
                logger.error(f"✗ Failed to process {video_name}: {str(e)}")
                import traceback
                error_msg = traceback.format_exc()
                
                summary_metrics['folder'].append(folder_name)
                summary_metrics['video_name'].append(video_name)
                summary_metrics['mpjpe'].append(np.nan)
                summary_metrics['pa_mpjpe'].append(np.nan)
                summary_metrics['pve'].append(np.nan)
                summary_metrics['status'].append(f'Failed: {str(e)}')
                
                # Save error log
                with open(osp.join(output_pth, 'error.log'), 'w') as f:
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(f"Full traceback:\n{error_msg}\n")
                continue
    
    return summary_metrics


def save_summary(summary_metrics, output_base):
    """
    Save summary of all evaluations
    """
    # Create DataFrame
    df = pd.DataFrame(summary_metrics)
    
    # Save to CSV
    csv_path = osp.join(output_base, 'evaluation_summary.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved summary to: {csv_path}")
    
    # Calculate and display statistics
    logger.info("\n" + "="*60)
    logger.info("OVERALL SUMMARY")
    logger.info("="*60)
    
    # Statistics by folder
    for folder in df['folder'].unique():
        folder_df = df[df['folder'] == folder]
        success_df = folder_df[folder_df['status'] == 'Success']
        
        logger.info(f"\nFolder: {folder}")
        logger.info(f"  Total videos: {len(folder_df)}")
        logger.info(f"  Successful: {len(success_df)}")
        logger.info(f"  Failed: {len(folder_df) - len(success_df)}")
        
        if len(success_df) > 0:
            logger.info(f"  Average MPJPE: {success_df['mpjpe'].mean():.2f} mm")
            logger.info(f"  Average PA-MPJPE: {success_df['pa_mpjpe'].mean():.2f} mm")
            logger.info(f"  Average PVE: {success_df['pve'].mean():.2f} mm")
    
    # Overall statistics
    success_df = df[df['status'] == 'Success']
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL STATISTICS (All Folders)")
    logger.info("="*60)
    logger.info(f"Total videos processed: {len(df)}")
    logger.info(f"Successful: {len(success_df)}")
    logger.info(f"Failed: {len(df) - len(success_df)}")
    
    if len(success_df) > 0:
        logger.info(f"\nAverage Metrics Across All Videos:")
        logger.info(f"  MPJPE:    {success_df['mpjpe'].mean():.2f} ± {success_df['mpjpe'].std():.2f} mm")
        logger.info(f"  PA-MPJPE: {success_df['pa_mpjpe'].mean():.2f} ± {success_df['pa_mpjpe'].std():.2f} mm")
        logger.info(f"  PVE:      {success_df['pve'].mean():.2f} ± {success_df['pve'].std():.2f} mm")
    
    # Save statistics to JSON
    stats = {
        'total_videos': len(df),
        'successful': len(success_df),
        'failed': len(df) - len(success_df),
        'overall_metrics': {
            'mpjpe_mean': float(success_df['mpjpe'].mean()) if len(success_df) > 0 else None,
            'mpjpe_std': float(success_df['mpjpe'].std()) if len(success_df) > 0 else None,
            'pa_mpjpe_mean': float(success_df['pa_mpjpe'].mean()) if len(success_df) > 0 else None,
            'pa_mpjpe_std': float(success_df['pa_mpjpe'].std()) if len(success_df) > 0 else None,
            'pve_mean': float(success_df['pve'].mean()) if len(success_df) > 0 else None,
            'pve_std': float(success_df['pve'].std()) if len(success_df) > 0 else None,
        },
        'by_folder': {}
    }
    
    for folder in df['folder'].unique():
        folder_df = df[df['folder'] == folder]
        success_folder_df = folder_df[folder_df['status'] == 'Success']
        stats['by_folder'][folder] = {
            'total': len(folder_df),
            'successful': len(success_folder_df),
            'failed': len(folder_df) - len(success_folder_df),
            'mpjpe_mean': float(success_folder_df['mpjpe'].mean()) if len(success_folder_df) > 0 else None,
            'pa_mpjpe_mean': float(success_folder_df['pa_mpjpe'].mean()) if len(success_folder_df) > 0 else None,
            'pve_mean': float(success_folder_df['pve'].mean()) if len(success_folder_df) > 0 else None,
        }
    
    with open(osp.join(output_base, 'summary_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    logger.info(f"\nSaved statistics to: {osp.join(output_base, 'summary_statistics.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--folders', type=str, nargs='+', required=True,
                        help='List of folders containing videos to process')

    parser.add_argument('--output_base', type=str, default='output/batch_eval',
                        help='Base output folder for all results')
    
    parser.add_argument('--gt_checkpoint', type=str, required=True,
                        help='Ground truth model checkpoint path')
    
    parser.add_argument('--pred_checkpoint', type=str, required=True,
                        help='Prediction model checkpoint path')
    
    parser.add_argument('--calib', type=str, default=None,
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')

    args = parser.parse_args()
    args.folders = [_resolve_cli_path(folder) for folder in args.folders]
    args.gt_checkpoint = _resolve_cli_path(args.gt_checkpoint)
    args.pred_checkpoint = _resolve_cli_path(args.pred_checkpoint)
    args.calib = _resolve_cli_path(args.calib)

    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(REPO_ROOT / 'configs' / 'yamls' / 'demo.yaml'))
    cfg = _prepare_runtime_cfg(cfg)
    
    _log_device_info(cfg.DEVICE)
    
    # Create base output directory
    os.makedirs(args.output_base, exist_ok=True)
    
    # ========= Load MoViD Models ========= #
    logger.info("\n" + "="*60)
    logger.info("Loading Models...")
    logger.info("="*60)
    
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    
    # Load Ground Truth model
    network_gt = build_network(cfg, smpl)
    args.gt_checkpoint = _load_network_checkpoint(network_gt, args.gt_checkpoint, cfg.DEVICE, 'Ground Truth')
    
    # Load Prediction model
    network_pred = build_network(cfg, smpl)
    args.pred_checkpoint = _load_network_checkpoint(network_pred, args.pred_checkpoint, cfg.DEVICE, 'Prediction')
    
    # ========= Process all folders ========= #
    summary_metrics = process_folders(
        cfg,
        args.folders,
        args.output_base,
        args.gt_checkpoint,
        args.pred_checkpoint,
        network_gt,
        network_pred,
        args
    )
    
    # ========= Save summary ========= #
    save_summary(summary_metrics, args.output_base)
    
    logger.info("\n" + "="*60)
    logger.info("All processing complete!")
    logger.info("="*60)
