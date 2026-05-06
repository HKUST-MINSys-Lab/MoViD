import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
import sys
from pathlib import Path

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar

# Fix PyTorch3D CUDA library path issue - set LD_LIBRARY_PATH before any PyTorch3D imports
# Find torch lib directory and add it to LD_LIBRARY_PATH
torch_lib_path = None
for path in sys.path:
    potential_path = osp.join(path, 'torch', 'lib')
    if osp.exists(potential_path):
        torch_lib_path = potential_path
        break

if torch_lib_path:
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if torch_lib_path not in current_ld_path:
        os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}" if current_ld_path else torch_lib_path
        # Also try to add it to ctypes library search path for immediate effect
        try:
            import ctypes
            if hasattr(ctypes, 'CDLL'):
                # Preload torch CUDA libraries to ensure they're available
                lib_path = osp.join(torch_lib_path, 'libtorch_cuda.so')
                if osp.exists(lib_path):
                    try:
                        ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    except:
                        pass  # Ignore if already loaded
        except:
            pass  # ctypes might not be available

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

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=True,
        visualize=False,
        action_recognizer=None,
        run_smplify=False,
        skeleton_only=False,
        motiongpt_predictor=None,
        motiongpt_chunk_size=100,
        gpt_text_visualize=False):
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
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
            
            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None
            
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
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
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
            logger.info(f'Already processed data exists at {output_pth} ! Load the data .')
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run MoViD
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x, None,inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x, None, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
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
                pred = network(x,None, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        
        # if False:
        if run_smplify:
            smplify = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
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
        results[_id]['trans_cam'] = pred['trans_cam'].cpu().numpy()  # Save trans_cam for projection
        
        # Extract NTU 25 keypoints for action recognition (real-time)
        with torch.no_grad():
            vertices = pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)  # Shape may vary
            
            # Handle different vertex shapes
            if len(vertices.shape) == 3:
                print(f"vertices.shape: {vertices.shape}")
                # Shape: (T, 6890, 3) - single batch
                vertices_tensor = vertices.to(cfg.DEVICE)
                ntu_joints = network.smpl.get_ntu_joints(vertices_tensor)  # (T, 25, 3)
                ntu_joints = ntu_joints.cpu().numpy()
            elif len(vertices.shape) == 4:
                print(f"vertices.shape: {vertices.shape}")
                # Shape: (B, T, 6890, 3) - multiple batches
                B, T = vertices.shape[:2]
                vertices_flat = vertices.reshape(-1, vertices.shape[2], vertices.shape[3])  # (B*T, 6890, 3)
                vertices_tensor = vertices_flat.to(cfg.DEVICE)
                ntu_joints = network.smpl.get_ntu_joints(vertices_tensor)  # (B*T, 25, 3)
                ntu_joints = ntu_joints.reshape(B, T, 25, 3).cpu().numpy()  # (B, T, 25, 3)
                ntu_joints = ntu_joints[0]  # (T, 25, 3) for first batch
            else:
                # Fallback: try to reshape
                logger.warning(f"Unexpected vertices shape: {vertices.shape}, attempting to reshape")
                if vertices.shape[-1] == 3:
                    # Assume last dimension is 3D coordinates
                    vertices_flat = vertices.reshape(-1, vertices.shape[-2], vertices.shape[-1])
                    vertices_tensor = vertices_flat.to(cfg.DEVICE)
                    ntu_joints = network.smpl.get_ntu_joints(vertices_tensor)
                    ntu_joints = ntu_joints.cpu().numpy()
                else:
                    logger.error(f"Cannot process vertices with shape: {vertices.shape}")
                    ntu_joints = None
            
            if ntu_joints is not None:
                results[_id]['ntu_joints'] = ntu_joints
                
                # Real-time action prediction frame by frame
                if action_recognizer is not None:
                    frame_predictions = []
                    T_frames = ntu_joints.shape[0]
                    
                    # Reset buffer for each subject
                    action_recognizer.reset_buffer()
                    
                    # Process each frame for real-time prediction
                    for t in range(T_frames):
                        joints_frame = ntu_joints[t]  # (25, 3)
                        
                        # Check if joints are valid
                        if np.isnan(joints_frame).any() or np.isinf(joints_frame).any():
                            logger.debug(f"Frame {t}: Invalid NTU joints, skipping")
                            continue
                        
                        # Real-time prediction (adds frame to buffer and predicts)
                        class_idx, confidence, label = action_recognizer.predict_action(joints_frame)
                        
                        # Get actual frame index
                        actual_frame_idx = frame_id[t] if isinstance(frame_id, (list, np.ndarray)) and len(frame_id) > t else t
                        
                        # Store prediction for this frame
                        frame_pred = {
                            'frame_idx': actual_frame_idx,
                            'class_idx': int(class_idx),
                            'confidence': float(confidence),
                            'label': label
                        }
                        frame_predictions.append(frame_pred)
                        
                        # Real-time print action prediction for every frame
                        buffer_size = action_recognizer.get_buffer_size()
                        if class_idx >= 0 and 'buffering' not in label and label not in ["waiting...", "rate_limiting", "insufficient_frames"]:
                            # Valid prediction - print it
                            print(f"[Real-time prediction] Subject {_id}, Frame {actual_frame_idx:4d} | Action: {label:25s} | Confidence: {confidence:.4f} | Buffer: {buffer_size:3d}/{action_recognizer.window_size}")
                            logger.info(f"Subject {_id}, Frame {actual_frame_idx}: Action = {label}, Confidence = {confidence:.4f}, Buffer = {buffer_size}/{action_recognizer.window_size}")
                        elif buffer_size < action_recognizer.window_size:
                            # Still buffering
                            if t % 10 == 0:  # Print every 10 frames to avoid too much output
                                print(f"[Buffering] Subject {_id}, Frame {actual_frame_idx:4d} | Buffer: {buffer_size:3d}/{action_recognizer.window_size} frames...")
                        else:
                            # Invalid prediction but buffer is full
                            if t % 10 == 0:  # Print every 10 frames
                                print(f"[Predicting] Subject {_id}, Frame {actual_frame_idx:4d} | Status: {label} | Buffer: {buffer_size:3d}/{action_recognizer.window_size}")
                    
                    # Store frame-by-frame predictions
                    if 'action_predictions' not in results[_id]:
                        results[_id]['action_predictions'] = []
                    results[_id]['action_predictions'].extend(frame_predictions)
                    
                    # Get overall prediction (most confident or most common)
                    if len(frame_predictions) > 0:
                        valid_predictions = [p for p in frame_predictions if p['class_idx'] >= 0 and 'buffering' not in p['label'] and p['label'] not in ["waiting...", "rate_limiting", "insufficient_frames"]]
                        if len(valid_predictions) > 0:
                            best_pred = max(valid_predictions, key=lambda x: x['confidence'])
                            results[_id]['predicted_action'] = best_pred['label']
                            results[_id]['action_confidence'] = best_pred['confidence']
                            results[_id]['action_class_idx'] = best_pred['class_idx']
                            
                            # Print summary
                            print(f"\n{'='*80}")
                            print(f"[Summary] Subject {_id}: Overall predicted action = {best_pred['label']}")
                            print(f"       Confidence: {best_pred['confidence']:.4f}")
                            print(f"       Valid predicted frames: {len(valid_predictions)}/{len(frame_predictions)}")
                            print(f"{'='*80}\n")
                            logger.info(f"Subject {_id}: Overall Action = {best_pred['label']}, Confidence = {best_pred['confidence']:.4f}, Valid frames = {len(valid_predictions)}/{len(frame_predictions)}")
                        else:
                            print(f"\n[Warning] Subject {_id}: No valid action prediction result\n")
                            logger.warning(f"Subject {_id}: No valid action predictions")
    
    # Action recognition is now done frame-by-frame during processing above
    # Results are already stored in results[_id]['action_predictions'] and results[_id]['predicted_action']

    # MotionGPT motion-to-text prediction
    if motiongpt_predictor is not None:
        logger.info("Running MotionGPT motion-to-text prediction...")
        print(f"\n{'='*80}")
        print("MotionGPT Motion-to-Text Action Prediction")
        print(f"{'='*80}")

        for _id, subject_data in results.items():
            if 'pose' not in subject_data or 'trans' not in subject_data or 'betas' not in subject_data:
                logger.warning(f"Subject {_id}: Missing pose/trans/betas for MotionGPT prediction")
                continue

            try:
                # Predict using MotionGPT
                predictions = motiongpt_predictor.predict_from_movid(
                    subject_data,
                    chunk_size=motiongpt_chunk_size
                )

                # Store predictions
                results[_id]['motiongpt_predictions'] = predictions

                # Print results
                print(f"\nSubject {_id}:")
                for pred in predictions:
                    action_label = motiongpt_predictor.summarize_to_action(pred['description'])
                    print(f"  [{pred['start_time']:.1f}s - {pred['end_time']:.1f}s] <{action_label}> {pred['description']}")
                    pred['action_label'] = action_label

                # Overall summary - get most common action
                if predictions:
                    from collections import Counter
                    action_labels = [p.get('action_label', 'motion') for p in predictions]
                    most_common = Counter(action_labels).most_common(1)[0][0]
                    results[_id]['motiongpt_action'] = most_common
                    results[_id]['motiongpt_full_description'] = ' '.join([p['description'] for p in predictions])
                    print(f"  [Summary] Primary action: {most_common}")

                logger.info(f"Subject {_id}: MotionGPT prediction completed with {len(predictions)} chunks")

            except Exception as e:
                logger.error(f"Subject {_id}: MotionGPT prediction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Save MotionGPT predictions to output.txt
        output_txt_path = osp.join(output_pth, 'output.txt')
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for _id, subject_data in results.items():
                if 'motiongpt_predictions' in subject_data:
                    f.write(f"Subject {_id}:\n")
                    for pred in subject_data['motiongpt_predictions']:
                        action_label = pred.get('action_label', 'motion')
                        f.write(f"{pred['start_time']:.2f} {pred['end_time']:.2f} [{action_label}] {pred['description']}\n")
                    f.write("\n")
        logger.info(f"Saved MotionGPT predictions to {output_txt_path}")

        print(f"{'='*80}\n")

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "movid_output.pkl"))
        logger.info(f"Saved results to {output_pth}/movid_output.pkl")
        
        # Also save a summary of action predictions
        action_summary = {}
        for _id, subject_data in results.items():
            if 'predicted_action' in subject_data:
                action_summary[_id] = {
                    'predicted_action': subject_data['predicted_action'],
                    'confidence': subject_data.get('action_confidence', 0.0),
                    'class_idx': subject_data.get('action_class_idx', -1),
                    'num_frames': len(subject_data.get('action_predictions', []))
                }
        
        if action_summary:
            joblib.dump(action_summary, osp.join(output_pth, "action_recognition_summary.pkl"))
            logger.info(f"Saved action recognition summary to {output_pth}/action_recognition_summary.pkl")

        # Save MotionGPT predictions summary
        motiongpt_summary = {}
        for _id, subject_data in results.items():
            if 'motiongpt_predictions' in subject_data:
                motiongpt_summary[_id] = {
                    'action': subject_data.get('motiongpt_action', ''),
                    'full_description': subject_data.get('motiongpt_full_description', ''),
                    'predictions': subject_data['motiongpt_predictions'],
                }

        if motiongpt_summary:
            joblib.dump(motiongpt_summary, osp.join(output_pth, "motiongpt_predictions.pkl"))
            logger.info(f"Saved MotionGPT predictions to {output_pth}/motiongpt_predictions.pkl")

    # Visualize
    if visualize:
        with torch.no_grad():
            if skeleton_only:
                # Render the NTU skeleton only (faster, no mesh)
                from lib.vis.run_vis import run_skeleton_vis
                run_skeleton_vis(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
            else:
                # Render the NTU skeleton + mesh (full visualization)
                from lib.vis.run_vis import run_vis_on_demo
                run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
    
    # GPT Text Visualization (standalone visualization branch)
    # If both --visualize and --motiongpt are enabled, automatically generate two videos:
    # 1. output_gpt.mp4: NTU skeleton + action_recognition
    # 2. output_gpt_text.mp4: NTU skeleton + MotionGPT text
    if gpt_text_visualize or (visualize and motiongpt_predictor is not None):
        with torch.no_grad():
            from lib.vis.run_vis import run_gpt_text_vis
            run_gpt_text_vis(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
        
def run_stream(cfg,
               video,
               output_pth,
               network,
               calib=None,
               run_global=True,
               save_pkl=True,
               visualize=False,
               action_recognizer=None,
               run_smplify=False,
               skeleton_only=False,
               motiongpt_predictor=None,
               motiongpt_chunk_size=100,
               gpt_text_visualize=False,
               stream_window_size=10):
    """Stream mode: process video frame-by-frame using network.stream_inference()."""
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Failed to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    run_global = run_global and _run_global

    # Preprocess (same as run): load or compute tracking_results, slam_results
    with torch.no_grad():
        if not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and
                osp.exists(osp.join(output_pth, 'slam_results.pth'))):
            cap = cv2.VideoCapture(video)
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
            if run_global:
                slam = SLAMModel(video, output_pth, width, height, calib)
            else:
                slam = None
            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            while cap.isOpened():
                flag, img = cap.read()
                if not flag:
                    break
                detector.track(img, fps, length)
                if slam is not None:
                    slam.track()
                bar.next()
            cap.release()
            tracking_results = detector.process(fps)
            if slam is not None:
                slam_results = slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0
            tracking_results = extractor.run(video, tracking_results)
            logger.info('Complete Data preprocessing!')
            joblib.dump(tracking_results, osp.join(output_pth, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_pth}')
        else:
            tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_pth}! Load the data.')

    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    results = defaultdict(dict)
    n_subjs = len(dataset)
    device = cfg.DEVICE

    for subj in range(n_subjs):
        batch = dataset.load_data(subj)
        if batch is None:
            continue
        _id, x_full, inits, features_full, mask_full, init_root, cam_angvel_full, frame_id, kwargs = batch
        L = x_full.shape[1]
        if L == 0:
            continue

        # Optional flipped branch for FLIP_EVAL (same as demo mode)
        if cfg.FLIP_EVAL:
            flipped_batch = dataset.load_data(subj, True)
            if flipped_batch is None:
                flipped_batch = batch
            _id_f, x_full_f, inits_f, features_full_f, mask_full_f, init_root_f, cam_angvel_full_f, frame_id_f, kwargs_f = flipped_batch
        else:
            x_full_f = inits_f = features_full_f = mask_full_f = None
            init_root_f = cam_angvel_full_f = kwargs_f = None

        # Accumulators for this subject
        list_pose, list_pose_world, list_trans, list_trans_world = [], [], [], []
        list_betas, list_verts, list_joints2d, list_joints3d, list_trans_cam = [], [], [], [], []
        hidden_states = None
        prev_context = None
        prev_kp3d = None
        prev_output = None
        hidden_states_flip = None
        prev_context_flip = None
        prev_kp3d_flip = None
        prev_output_flip = None

        bar = Bar(f'Stream inference subject {_id}' + (' (flip)' if cfg.FLIP_EVAL else ''), fill='#', max=L)
        with torch.no_grad():
            for t in range(L):
                start = max(0, t - stream_window_size + 1)
                end = t + 1
                x = x_full[:, start:end]
                mask = mask_full[:, start:end]
                features = features_full[:, start:end]
                cam_angvel = cam_angvel_full[:, start:end]

                if cfg.FLIP_EVAL:
                    x_f = x_full_f[:, start:end]
                    mask_f = mask_full_f[:, start:end]
                    features_f = features_full_f[:, start:end]
                    cam_angvel_f = cam_angvel_full_f[:, start:end]

                    # Normal stream
                    pred, hidden_states, prev_context, prev_kp3d = network.stream_inference(
                        x, inits,
                        img_features=features,
                        mask=mask,
                        init_root=init_root,
                        cam_angvel=cam_angvel,
                        return_y_up=True,
                        window_size=stream_window_size,
                        hidden_states=hidden_states,
                        prev_context=prev_context,
                        prev_kp3d=prev_kp3d,
                        prev_output=prev_output,
                        **kwargs
                    )
                    prev_output = pred
                    saved_old_motion_context = network.old_motion_context.clone() if (hasattr(network, 'old_motion_context') and network.old_motion_context is not None) else None

                    # Flipped stream
                    flipped_pred, hidden_states_flip, prev_context_flip, prev_kp3d_flip = network.stream_inference(
                        x_f, inits_f,
                        img_features=features_f,
                        mask=mask_f,
                        init_root=init_root_f,
                        cam_angvel=cam_angvel_f,
                        return_y_up=True,
                        window_size=stream_window_size,
                        hidden_states=hidden_states_flip,
                        prev_context=prev_context_flip,
                        prev_kp3d=prev_kp3d_flip,
                        prev_output=prev_output_flip,
                        **kwargs_f
                    )
                    prev_output_flip = flipped_pred

                    # Restore normal stream's motion context for refiner
                    if saved_old_motion_context is not None:
                        network.old_motion_context = saved_old_motion_context

                    # Merge predictions (same as run() demo mode)
                    flipped_pose = flipped_pred['pose'][:, -1:].reshape(1, 24, 6)
                    pose = pred['pose'][:, -1:].reshape(1, 24, 6)
                    flipped_shape = flipped_pred['betas'][:, -1:].squeeze(1)
                    shape = pred['betas'][:, -1:].squeeze(1)
                    avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                    avg_pose = avg_pose.reshape(1, 1, 144)
                    avg_shape = avg_shape.unsqueeze(1)
                    avg_contact = (flipped_pred['contact'][:, -1:, [2, 3, 0, 1]] + pred['contact'][:, -1:]) / 2

                    network.pred_pose = avg_pose.to(pred['pose'].device)
                    network.pred_shape = avg_shape.to(pred['betas'].device)
                    network.pred_contact = avg_contact.to(pred['contact'].device)
                    network.pred_cam = (pred['cam'][:, -1:] + flipped_pred['cam'][:, -1:]) / 2
                    network.pred_root = pred['poses_root_r6d'][:, -1:]
                    network.pred_vel = pred['vel_root'][:, -1:]

                    smpl_kwargs = dict(kwargs)
                    smpl_kwargs['cam_intrinsics'] = kwargs.get('cam_intrinsics')
                    smpl_kwargs['res'] = kwargs.get('res')
                    bbox = kwargs.get('bbox')
                    if bbox is not None and bbox.dim() == 3 and bbox.shape[1] > 1:
                        smpl_kwargs['bbox'] = bbox[:, -1:]
                    else:
                        smpl_kwargs['bbox'] = bbox
                    output = network.forward_smpl(**smpl_kwargs)
                    # Stream has only 1 frame; refine_trajectory needs >=2 frames (reset_root_velocity uses [:, 1:])
                    pred = network.rollout(output, network.pred_root, network.pred_vel, return_y_up=True)
                else:
                    pred, hidden_states, prev_context, prev_kp3d = network.stream_inference(
                        x, inits,
                        img_features=features,
                        mask=mask,
                        init_root=init_root,
                        cam_angvel=cam_angvel,
                        return_y_up=True,
                        window_size=stream_window_size,
                        hidden_states=hidden_states,
                        prev_context=prev_context,
                        prev_kp3d=prev_kp3d,
                        prev_output=prev_output,
                        **kwargs
                    )
                    prev_output = pred

                # Take last frame from output (stream [1,T,...] or FLIP branch; rollout can be empty when T=1)
                poses_body_last = pred['poses_body'].reshape(-1, 23, 3, 3)[-1]
                pred_body_pose = matrix_to_axis_angle(poses_body_last).cpu().numpy().reshape(-1, 69)
                proot_cam = pred['poses_root_cam'].reshape(-1, 1, 3, 3)
                pred_root = matrix_to_axis_angle(proot_cam[-1]).cpu().numpy().reshape(-1, 3)
                proot_world = pred['poses_root_world'].reshape(-1, 1, 3, 3)
                pred_root_world = matrix_to_axis_angle(proot_world[-1]).cpu().numpy().reshape(-1, 3) if proot_world.shape[0] > 0 else pred_root.copy()
                pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
                pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
                trans_cam_last = pred['trans_cam'].reshape(-1, 3)[-1]
                offset_frame = network.output.offset.reshape(-1, 3)[-1] if hasattr(network.output, 'offset') and network.output.offset is not None else torch.zeros_like(trans_cam_last).to(trans_cam_last.device)
                trans = (trans_cam_last - offset_frame).cpu().numpy().flatten()
                verts_last = (pred['verts_cam'].reshape(-1, pred['verts_cam'].shape[-2], 3)[-1] + trans_cam_last.unsqueeze(0).to(pred['verts_cam'].device)).cpu().numpy()
                list_pose.append(pose.squeeze(0))
                list_pose_world.append(pose_world.squeeze(0))
                list_trans.append(trans)
                trans_world_flat = pred['trans_world'].reshape(-1, 3)
                list_trans_world.append(trans_world_flat[-1].cpu().numpy().flatten() if trans_world_flat.shape[0] > 0 else trans.copy())
                list_betas.append(pred['betas'].reshape(-1, 10)[-1].cpu().numpy().flatten())
                list_verts.append(verts_last)
                list_joints2d.append(pred['joints2d'].reshape(-1, pred['joints2d'].shape[-2], pred['joints2d'].shape[-1])[-1].cpu().numpy())
                list_joints3d.append(pred['joints3d'].reshape(-1, pred['joints3d'].shape[-2], 3)[-1].cpu().numpy())
                list_trans_cam.append(trans_cam_last.cpu().numpy().flatten())
                bar.next()

        bar.finish()
        results[_id]['pose'] = np.concatenate(list_pose, axis=0)
        results[_id]['pose_world'] = np.concatenate(list_pose_world, axis=0)
        results[_id]['trans'] = np.concatenate(list_trans, axis=0)
        results[_id]['trans_world'] = np.concatenate(list_trans_world, axis=0)
        results[_id]['betas'] = list_betas[0]
        results[_id]['verts'] = np.concatenate(list_verts, axis=0)
        results[_id]['frame_ids'] = frame_id
        results[_id]['joints2d'] = np.stack(list_joints2d, axis=0)
        results[_id]['joints3d'] = np.stack(list_joints3d, axis=0)
        results[_id]['trans_cam'] = np.concatenate(list_trans_cam, axis=0)

        # NTU joints and action recognition (same as run)
        vertices = np.concatenate(list_verts, axis=0)
        vertices_tensor = torch.from_numpy(vertices).float().to(device)
        ntu_joints = network.smpl.get_ntu_joints(vertices_tensor)
        ntu_joints = ntu_joints.cpu().numpy()
        results[_id]['ntu_joints'] = ntu_joints

        if action_recognizer is not None:
            frame_predictions = []
            action_recognizer.reset_buffer()
            for t in range(ntu_joints.shape[0]):
                joints_frame = ntu_joints[t]
                if np.isnan(joints_frame).any() or np.isinf(joints_frame).any():
                    continue
                class_idx, confidence, label = action_recognizer.predict_action(joints_frame)
                actual_frame_idx = frame_id[t] if isinstance(frame_id, (list, np.ndarray)) and len(frame_id) > t else t
                frame_predictions.append({
                    'frame_idx': actual_frame_idx,
                    'class_idx': int(class_idx),
                    'confidence': float(confidence),
                    'label': label
                })
                buffer_size = action_recognizer.get_buffer_size()
                if class_idx >= 0 and 'buffering' not in label and label not in ["waiting...", "rate_limiting", "insufficient_frames"]:
                    print(f"[Real-time prediction] Subject {_id}, Frame {actual_frame_idx:4d} | Action: {label:25s} | Confidence: {confidence:.4f} | Buffer: {buffer_size:3d}/{action_recognizer.window_size}")
                elif buffer_size < action_recognizer.window_size and t % 10 == 0:
                    print(f"[Buffering] Subject {_id}, Frame {actual_frame_idx:4d} | Buffer: {buffer_size:3d}/{action_recognizer.window_size} frames...")
            results[_id]['action_predictions'] = frame_predictions
            valid = [p for p in frame_predictions if p['class_idx'] >= 0 and 'buffering' not in p['label'] and p['label'] not in ["waiting...", "rate_limiting", "insufficient_frames"]]
            if valid:
                best = max(valid, key=lambda x: x['confidence'])
                results[_id]['predicted_action'] = best['label']
                results[_id]['action_confidence'] = best['confidence']
                results[_id]['action_class_idx'] = best['class_idx']
                print(f"\n[Summary] Subject {_id}: Overall predicted action = {best['label']}, Confidence: {best['confidence']:.4f}\n")

    # MotionGPT (same as run)
    if motiongpt_predictor is not None:
        logger.info("Running MotionGPT motion-to-text prediction (stream mode)...")
        for _id, subject_data in results.items():
            if 'pose' not in subject_data or 'trans' not in subject_data:
                continue
            try:
                predictions = motiongpt_predictor.predict_from_movid(subject_data, chunk_size=motiongpt_chunk_size)
                results[_id]['motiongpt_predictions'] = predictions
                for pred in predictions:
                    action_label = motiongpt_predictor.summarize_to_action(pred['description'])
                    pred['action_label'] = action_label
                if predictions:
                    from collections import Counter
                    action_labels = [p.get('action_label', 'motion') for p in predictions]
                    most_common = Counter(action_labels).most_common(1)[0][0]
                    results[_id]['motiongpt_action'] = most_common
                    results[_id]['motiongpt_full_description'] = ' '.join([p['description'] for p in predictions])
            except Exception as e:
                logger.error(f"Subject {_id}: MotionGPT failed: {e}")
        output_txt_path = osp.join(output_pth, 'output.txt')
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for _id, subject_data in results.items():
                if 'motiongpt_predictions' in subject_data:
                    for pred in subject_data['motiongpt_predictions']:
                        f.write(f"{pred['start_time']:.2f} {pred['end_time']:.2f} [{pred.get('action_label', '')}] {pred['description']}\n")
        logger.info(f"Saved MotionGPT predictions to {output_txt_path}")

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "movid_output.pkl"))
        logger.info(f"Saved results to {output_pth}/movid_output.pkl")
        action_summary = {}
        for _id, subject_data in results.items():
            if 'predicted_action' in subject_data:
                action_summary[_id] = {
                    'predicted_action': subject_data['predicted_action'],
                    'confidence': subject_data.get('action_confidence', 0.0),
                    'class_idx': subject_data.get('action_class_idx', -1),
                    'num_frames': len(subject_data.get('action_predictions', []))
                }
        if action_summary:
            joblib.dump(action_summary, osp.join(output_pth, "action_recognition_summary.pkl"))
        motiongpt_summary = {k: {'action': v.get('motiongpt_action', ''), 'full_description': v.get('motiongpt_full_description', ''), 'predictions': v.get('motiongpt_predictions', [])}
                             for k, v in results.items() if 'motiongpt_predictions' in v}
        if motiongpt_summary:
            joblib.dump(motiongpt_summary, osp.join(output_pth, "motiongpt_predictions.pkl"))

    if visualize:
        with torch.no_grad():
            if skeleton_only:
                from lib.vis.run_vis import run_skeleton_vis
                run_skeleton_vis(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
            else:
                from lib.vis.run_vis import run_vis_on_demo
                run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)

    if gpt_text_visualize or (visualize and motiongpt_predictor is not None):
        with torch.no_grad():
            from lib.vis.run_vis import run_gpt_text_vis
            run_gpt_text_vis(cfg, video, results, output_pth, network.smpl, vis_global=run_global)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str, 
                        default='examples/demo_video.mp4', 
                        help='input video path or youtube link')

    parser.add_argument('--output_pth', type=str, default='output/demo', 
                        help='output folder to write results')
    
    parser.add_argument('--calib', type=str, default=None, 
                        help='Camera calibration file path')

    parser.add_argument('--estimate_local_only', action='store_true',
                        help='Only estimate motion in camera coordinate if True')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the output mesh if True')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='Save output as pkl file')
    
    parser.add_argument('--run_smplify', action='store_true',
                        help='Run Temporal SMPLify for post processing')
    
    parser.add_argument('--action_config', type=str, default=None,
                        help='Path to action recognition config file (e.g., stgcn++ config)')
    parser.add_argument('--action_checkpoint', type=str, default=None,
                        help='Path to action recognition checkpoint file')
    parser.add_argument('--action_label_map', type=str, default=None,
                        help='Path to action label map file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to MoViD checkpoint file (overrides config)')

    parser.add_argument('--skeleton_only', action='store_true',
                        help='Only render NTU skeleton visualization (no mesh), faster than full visualization')

    # MotionGPT arguments for motion-to-text action prediction
    parser.add_argument('--motiongpt', action='store_true',
                        help='Enable MotionGPT motion-to-text action prediction')
    parser.add_argument('--motiongpt_path', type=str, default=os.environ.get('MOTIONGPT_PATH', 'MotionGPT3'),
                        help='Path to MotionGPT3 repository')
    parser.add_argument('--motiongpt_config', type=str, default=None,
                        help='Path to MotionGPT config file (default: configs/test_m2t.yaml)')
    parser.add_argument('--motiongpt_checkpoint', type=str, default=None,
                        help='Path to MotionGPT checkpoint file')
    parser.add_argument('--motiongpt_chunk_size', type=int, default=100,
                        help='Number of frames per chunk for MotionGPT (default: 100 = 5 seconds at 20fps)')
    parser.add_argument('--gpt_text_visualize', action='store_true',
                        help='Visualize MotionGPT text predictions on video (independent visualization branch)')

    # Stream mode: process video frame-by-frame using stream_inference
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'stream'],
                        help='demo: process whole video at once; stream: frame-by-frame with stream_inference')
    parser.add_argument('--stream_window_size', type=int, default=10,
                        help='Temporal window size for stream_inference (default: 10)')

    args = parser.parse_args()
    args.video = _resolve_cli_path(args.video)
    args.calib = _resolve_cli_path(args.calib)
    args.checkpoint = _resolve_cli_path(args.checkpoint)
    args.action_config = _resolve_cli_path(args.action_config)
    args.action_checkpoint = _resolve_cli_path(args.action_checkpoint)
    args.action_label_map = _resolve_cli_path(args.action_label_map)
    args.motiongpt_path = _resolve_cli_path(args.motiongpt_path)
    args.motiongpt_config = _resolve_cli_path(args.motiongpt_config)
    args.motiongpt_checkpoint = _resolve_cli_path(args.motiongpt_checkpoint)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(REPO_ROOT / 'configs' / 'yamls' / 'demo.yaml'))
    cfg = _prepare_runtime_cfg(cfg)
    
    # Override checkpoint if provided
    if args.checkpoint:
        cfg.TRAIN.CHECKPOINT = args.checkpoint
        logger.info(f"Using checkpoint: {args.checkpoint}")
    
    _log_device_info(cfg.DEVICE)
    
    # ========= Load MoViD ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    
    # Load checkpoint if specified
    if cfg.TRAIN.CHECKPOINT and os.path.exists(cfg.TRAIN.CHECKPOINT):
        logger.info(f"Loading checkpoint from {cfg.TRAIN.CHECKPOINT}")
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT, map_location=cfg.DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            # Filter out SMPL parameters (SMPL is not trainable and shouldn't be loaded)
            state_dict = {k: v for k, v in checkpoint['model'].items() if not k.startswith('smpl.')}
        else:
            # Try direct loading, but filter SMPL if present
            state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('smpl.')}
        
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
            logger.warning(f"Skipped {len(skipped_shape_keys)} checkpoint keys with incompatible shapes")
            logger.debug(f"First few shape-mismatched keys: {skipped_shape_keys[:5]}")
        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {len(missing_keys)} keys")
            logger.debug(f"First few missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            logger.debug(f"First few unexpected keys: {unexpected_keys[:5]}")
        logger.info(f"Checkpoint loaded successfully ({len(compatible_state_dict)} compatible parameters)")
    
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)
    
    # Initialize action recognizer if provided (will be passed to run function)
    action_recognizer = None
    if args.action_config and args.action_checkpoint:
        try:
            from lib.action_recognition import ActionRecognizer
            
            logger.info("Initializing action recognition model for real-time prediction...")
            action_recognizer = ActionRecognizer(
                config_path=args.action_config,
                checkpoint_path=args.action_checkpoint,
                label_map_path=args.action_label_map,
                device=cfg.DEVICE.lower(),
                window_size=150,
                num_keypoints=25
            )
            logger.info("Action recognition model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load action recognition model: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Initialize MotionGPT predictor if enabled
    motiongpt_predictor = None
    if args.motiongpt:
        try:
            from lib.motiongpt_predictor import MotionGPTPredictor

            logger.info("Initializing MotionGPT motion-to-text predictor...")
            motiongpt_predictor = MotionGPTPredictor(
                motiongpt_path=args.motiongpt_path,
                config_path=args.motiongpt_config,
                checkpoint_path=args.motiongpt_checkpoint,
                device=cfg.DEVICE.lower(),
            )
            logger.info("MotionGPT predictor loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load MotionGPT predictor: {e}")
            import traceback
            logger.error(traceback.format_exc())

    if args.mode == 'stream':
        logger.info('Running in STREAM mode (frame-by-frame with stream_inference)')
        run_stream(cfg,
                   args.video,
                   output_pth,
                   network,
                   args.calib,
                   run_global=not args.estimate_local_only,
                   save_pkl=args.save_pkl,
                   visualize=args.visualize,
                   action_recognizer=action_recognizer,
                   run_smplify=args.run_smplify,
                   skeleton_only=args.skeleton_only,
                   motiongpt_predictor=motiongpt_predictor,
                   motiongpt_chunk_size=args.motiongpt_chunk_size,
                   gpt_text_visualize=args.gpt_text_visualize,
                   stream_window_size=args.stream_window_size)
    else:
        run(cfg,
            args.video,
            output_pth,
            network,
            args.calib,
            run_global=not args.estimate_local_only,
            save_pkl=args.save_pkl,
            visualize=args.visualize,
            action_recognizer=action_recognizer,
            run_smplify=args.run_smplify,
            skeleton_only=args.skeleton_only,
            motiongpt_predictor=motiongpt_predictor,
            motiongpt_chunk_size=args.motiongpt_chunk_size,
            gpt_text_visualize=args.gpt_text_visualize)

    print()
    logger.info('Done !')
