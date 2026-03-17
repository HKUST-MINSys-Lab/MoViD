import os
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
import json

import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar
import psutil
import time

from configs.config import get_cfg_defaults
from lib.data.datasets import CustomDataset
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify

# Try to import jtop for Jetson resource monitoring
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
    logger.info('jtop is available for resource monitoring')
except ImportError:
    JTOP_AVAILABLE = False
    logger.info('jtop is not available, skipping resource monitoring')

try: 
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except: 
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def collect_jetson_stats(jetson_ctx):
    """Collect current Jetson resource statistics"""
    if not jetson_ctx or not jetson_ctx.ok():
        return None
    
    stats = jetson_ctx.stats
    
    def safe_get_value(key, default=0):
        """Safely get value from stats, handling both direct values and 'OFF' strings"""
        val = stats.get(key, default)
        if val == 'OFF':
            return 0
        elif isinstance(val, (int, float)):
            return val
        else:
            return default
    
    return {
        'timestamp': time.time(),
        'CPU1': safe_get_value('CPU1'),
        'CPU2': safe_get_value('CPU2'),
        'CPU3': safe_get_value('CPU3'),
        'CPU4': safe_get_value('CPU4'),
        'CPU5': safe_get_value('CPU5'),
        'CPU6': safe_get_value('CPU6'),
        'CPU7': safe_get_value('CPU7'),
        'CPU8': safe_get_value('CPU8'),
        'RAM': safe_get_value('RAM'),
        'GPU': safe_get_value('GPU'),
        'Power_TOT': safe_get_value('Power TOT'),
        'Temp_CPU': safe_get_value('Temp CPU'),
        'Temp_tj': safe_get_value('Temp tj'),
        'Fan_pwmfan0': safe_get_value('Fan pwmfan0'),
        'SWAP': safe_get_value('SWAP'),
        'EMC': safe_get_value('EMC')
    }

def run(cfg,
        video,
        output_pth,
        network,
        calib=None,
        run_global=True,
        save_pkl=False,
        visualize=False):
    import time
    frame_start_time = time.time()
    
    # Initialize jtop context manager
    jetson_ctx = None
    jetson_resource_log = []
    
    if JTOP_AVAILABLE:
        try:
            jetson_ctx = jtop()
            jetson_ctx.start()
            logger.info('jtop monitoring started')
        except Exception as e:
            logger.warning(f'Failed to start jtop: {e}')
            jetson_ctx = None
    
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Whether or not estimating motion in global coordinates
    run_global = run_global and _run_global
    
    # Check if we need to preprocess
    need_preprocess = not (osp.exists(osp.join(output_pth, 'tracking_results.pth')) and
                           osp.exists(osp.join(output_pth, 'slam_results.pth')))

    # If cached data exists, check if it has flipped data when FLIP_EVAL is enabled
    if not need_preprocess:
        tracking_results = joblib.load(osp.join(output_pth, 'tracking_results.pth'))
        slam_results = joblib.load(osp.join(output_pth, 'slam_results.pth'))
        logger.info(f'Already processed data exists at {output_pth} ! Load the data .')

        # Debug: flip data availability (for FLIP_EVAL)
        sample_id = list(tracking_results.keys())[0]
        has_flipped = any(k.startswith('flipped_') for k in tracking_results[sample_id].keys())
        logger.info(f'Cached data flip keys: has_flipped={has_flipped}, FLIP_EVAL={cfg.FLIP_EVAL}')

        if cfg.FLIP_EVAL:
            if 'flipped_keypoints' not in tracking_results[sample_id]:
                logger.warning('FLIP_EVAL is enabled but cached data has no flipped keys. Deleting cache to reprocess...')
                os.remove(osp.join(output_pth, 'tracking_results.pth'))
                os.remove(osp.join(output_pth, 'slam_results.pth'))
                need_preprocess = True

    # Preprocess
    if need_preprocess:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with torch.no_grad():
            detector = DetectionModel(cfg.DEVICE.lower())
            extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)

            if run_global: slam = SLAMModel(video, output_pth, width, height, calib)
            else: slam = None

            bar = Bar('Preprocess: 2D detection and SLAM', fill='#', max=length)
            frame_idx = 0
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break

                # Collect resource stats for preprocessing phase
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'preprocessing'
                        stats['frame_idx'] = frame_idx
                        jetson_resource_log.append(stats)

                # 2D detection and tracking
                detector.track(img, fps, length)

                # SLAM
                if slam is not None:
                    slam.track()

                bar.next()
                frame_idx += 1

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
    
    # Build dataset
    dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
    
    # run WHAM
    results = defaultdict(dict)
    
    n_subjs = len(dataset)
    for subj in range(n_subjs):

        with torch.no_grad():
            if cfg.FLIP_EVAL:
                # Collect resource stats before flipped forward pass
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'wham_inference_flipped'
                        stats['subject_idx'] = subj
                        jetson_resource_log.append(stats)
                
                # Forward pass with flipped input
                flipped_batch = dataset.load_data(subj, True)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
                flipped_pred = network(x,None, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Collect resource stats before normal forward pass
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'wham_inference_normal'
                        stats['subject_idx'] = subj
                        jetson_resource_log.append(stats)
                
                # Forward pass with normal input
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                pred = network(x,None, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Collect resource stats after inference
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'wham_post_inference'
                        stats['subject_idx'] = subj
                        jetson_resource_log.append(stats)
                
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
                #pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
            
            else:
                # Collect resource stats before inference
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'wham_inference'
                        stats['subject_idx'] = subj
                        jetson_resource_log.append(stats)
                
                # data
                batch = dataset.load_data(subj)
                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
                
                # inference
                pred = network(x, None, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
                
                # Collect resource stats after inference
                if jetson_ctx:
                    stats = collect_jetson_stats(jetson_ctx)
                    if stats:
                        stats['phase'] = 'wham_post_inference'
                        stats['subject_idx'] = subj
                        jetson_resource_log.append(stats)
        
        # if False:
        if args.run_smplify:
            # Collect resource stats before SMPLify
            if jetson_ctx:
                stats = collect_jetson_stats(jetson_ctx)
                if stats:
                    stats['phase'] = 'smplify'
                    stats['subject_idx'] = subj
                    jetson_resource_log.append(stats)
            
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
        pred_pose = np.concatenate((pred_root, pred_body_pose), axis=-1)
        pred_trans = (pred['trans_cam'] - network.output.offset).cpu().numpy()

        results[_id]['pose'] = pred_pose
        results[_id]['trans'] = pred_trans
        results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
        results[_id]['verts'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
        results[_id]['frame_ids'] = frame_id
        results[_id]['joints2d'] = pred['joints2d'].cpu().numpy()

        if 'poses_root_world' in pred:
            pred_root_world = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)
            pred_pose_world = np.concatenate((pred_root_world, pred_body_pose), axis=-1)
            results[_id]['pose_world'] = pred_pose_world
        if 'trans_world' in pred:
            results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()

    if save_pkl:
        joblib.dump(results, osp.join(output_pth, "wham_output.pkl"))
     
    # Visualize
    if visualize:
        # Collect resource stats before visualization
        if jetson_ctx:
            stats = collect_jetson_stats(jetson_ctx)
            if stats:
                stats['phase'] = 'visualization'
                jetson_resource_log.append(stats)

        from lib.vis.run_vis import run_vis_on_demo, run_skeleton_vis, RENDERER_AVAILABLE
        with torch.no_grad():
            if RENDERER_AVAILABLE:
                run_vis_on_demo(cfg, video, results, output_pth, network.smpl, vis_global=run_global)
            else:
                logger.info("pytorch3d not available, falling back to skeleton visualization")
                run_skeleton_vis(cfg, video, results, output_pth, network.smpl, vis_global=False)

    # Save resource monitoring data
    if jetson_resource_log:
        resource_log_path = osp.join(output_pth, 'jetson_resource_log.json')
        with open(resource_log_path, 'w') as f:
            json.dump(jetson_resource_log, f, indent=2)
        logger.info(f'Saved resource monitoring data to {resource_log_path}')
        
        # Create summary statistics
        if len(jetson_resource_log) > 0:
            # Calculate average CPU usage across all cores
            cpu_usages = []
            for record in jetson_resource_log:
                total_cpu = 0
                active_cores = 0
                for i in range(1, 9):
                    cpu_val = record.get(f'CPU{i}', 0)
                    if cpu_val > 0:  # Only count active cores
                        total_cpu += cpu_val
                        active_cores += 1
                if active_cores > 0:
                    cpu_usages.append(total_cpu / active_cores)
            
            summary = {
                'total_records': len(jetson_resource_log),
                'avg_cpu_usage': np.mean(cpu_usages) if cpu_usages else 0,
                'avg_gpu_usage': np.mean([r.get('GPU', 0) for r in jetson_resource_log]),
                'avg_ram_usage': np.mean([r.get('RAM', 0) for r in jetson_resource_log]) * 100,  # Convert to percentage
                'avg_power_consumption': np.mean([r.get('Power_TOT', 0) for r in jetson_resource_log]) / 1000,  # Convert to watts
                'max_temp_cpu': np.max([r.get('Temp_CPU', 0) for r in jetson_resource_log]),
                'max_temp_tj': np.max([r.get('Temp_tj', 0) for r in jetson_resource_log]),
                'avg_swap_usage': np.mean([r.get('SWAP', 0) for r in jetson_resource_log]) * 100  # Convert to percentage
            }
            
            summary_path = osp.join(output_pth, 'resource_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f'Saved resource summary to {summary_path}')
            
            # Log summary to console
            logger.info(f'Resource Usage Summary:')
            logger.info(f'  Average CPU Usage: {summary["avg_cpu_usage"]:.2f}%')
            logger.info(f'  Average GPU Usage: {summary["avg_gpu_usage"]:.2f}%')
            logger.info(f'  Average RAM Usage: {summary["avg_ram_usage"]:.2f}%')
            logger.info(f'  Average SWAP Usage: {summary["avg_swap_usage"]:.2f}%')
            logger.info(f'  Average Power Consumption: {summary["avg_power_consumption"]:.2f}W')
            logger.info(f'  Max CPU Temperature: {summary["max_temp_cpu"]:.2f}°C')
            logger.info(f'  Max Tj Temperature: {summary["max_temp_tj"]:.2f}°C')

    # Clean up jtop
    if jetson_ctx:
        try:
            jetson_ctx.close()
            logger.info('jtop monitoring stopped')
        except Exception as e:
            logger.warning(f'Error closing jtop: {e}')

    duration = time.time() - frame_start_time
    logger.info(f'Process {length} frames in {duration:.2f} seconds, FPS: {length / duration:.2f}')


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

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')    
    
    # ========= Load WHAM ========= #
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()
    
    # Output folder
    sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])
    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)

    t0 = time.time()
    run(cfg, 
        args.video, 
        output_pth, 
        network, 
        args.calib, 
        run_global=not args.estimate_local_only, 
        save_pkl=args.save_pkl,
        visualize=args.visualize)
    t1 = time.time()

    logger.info(f"Total run time: {t1-t0:.2f} seconds")
    print()
    logger.info('Done !')