from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import sys
import torch
import numpy as np
import cv2
import pickle
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
from lib.utils.imutils import compute_cam_intrinsics
from lib.models.preproc.detector import DetectionModel
from lib.data.utils.normalizer import Normalizer
from lib.models import build_body_model
from lib.utils.kp_utils import root_centering

@torch.no_grad()
def preprocess_totalcapture(totalcapture_files, args, batch_size=128):
    """Preprocess TotalCapture dataset
    Args:
        totalcapture_files: List of pickle files containing TotalCapture data
        batch_size: batch size for feature extraction
    """
    tt = lambda x: torch.from_numpy(x).float()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_pth = osp.join(_C.PATHS.PARSED_DATA, f'totalcapture_processed_train.pth')
    extractor = FeatureExtractor(device, flip_eval=True, max_batch_size=batch_size)
    detector = DetectionModel('cuda')
    smpl = build_body_model('cpu')
    keypoints_normalizer = Normalizer(None)
    
    # Initialize dataset dictionary to store processed data
    dataset = defaultdict(list)
    
    # Process each pickle file (train/validation)
    for file_path in totalcapture_files:
        logger.info(f'Loading TotalCapture data from {file_path}...')
        with open(file_path, 'rb') as f:
            totalcapture_data = pickle.load(f)
        
        # Group data by video_id and camera_id
        grouped_data = defaultdict(lambda: defaultdict(list))
        for item in totalcapture_data:
            grouped_data[item['video_id']][item['camera_id']].append(item)
        
        # Process each video and camera view
        for video_id, camera_views in grouped_data.items():
            logger.info(f'Processing video {video_id}...')
            
            for camera_id, frames in camera_views.items():
                # Sort frames by image_id to ensure correct order
                frames.sort(key=lambda x: x['image_id'])
                
                logger.info(f'Processing camera view {camera_id} with {len(frames)} frames')
                detector.initialize_tracking()
                
                # Get paths to all images in this sequence
                image_paths = [os.path.join(args.pickle_dir, item['image']) for item in frames]
                
                # Check if we can access the first image to get dimensions
                try:
                    sample_img = cv2.imread(image_paths[0])
                    if sample_img is None:
                        logger.warning(f'Cannot read image {image_paths[0]}. Skipping sequence.')
                        continue
                    height, width = sample_img.shape[:2]
                except Exception as e:
                    logger.warning(f'Error reading image: {e}. Skipping sequence.')
                    continue

                cam = frames[0]['camera']
                intrinsic_mat = np.array([[cam['fx'], 0., cam['cx']],
                                        [0., cam['fy'], cam['cy']],
                                        [0., 0., 1.]])
                zup2ydown = transforms.axis_angle_to_matrix(torch.tensor([[-np.pi/2, 0, 0]])).float()
                yaw = transforms.axis_angle_to_matrix(torch.tensor([[0, 2 * np.pi * np.random.uniform(), 0]])).float()
                zup2ydown = torch.matmul(yaw, zup2ydown)
                R_numpy = cam['R']
                R_tensor = torch.from_numpy(R_numpy).float()
                R = torch.matmul(R_tensor, zup2ydown)
                T = cam['T']/1000.0
                extrinsics = np.concatenate((R.reshape(3,3), np.reshape( T, (-1,1))), axis=1)
                
                # # Track people in the sequence
                # fps = 30  # Adjust if needed
                # length = len(frames)
                
                # bar = Bar(f'Loading frames from {video_id}_camera{camera_id}', fill='#', max=length)
                # frame_id = 0
                
                # # Process each frame for tracking
                # for img_path in image_paths:
                #     img = cv2.imread(img_path)
                #     if img is not None:
                #         detector.track(img, fps, length)
                #     bar.next()
                #     frame_id += 1
                # bar.finish()
                
                # # Process tracking results
                # tracking_results = detector.process(fps)
                # tracking_results = extractor.run(image_paths, tracking_results)
                
                # if not tracking_results or not tracking_results[0]:
                #     logger.warning(f'No tracking results for {video_id} camera {camera_id}')
                #     continue
                
                # # Extract data
                # frame_ids = np.array(tracking_results[0]['frame_id'])
                # res = torch.tensor([width, height]).float()
                # bbox = torch.from_numpy(tracking_results[0]['bbox']).float()
                # flipped_bbox = torch.from_numpy(tracking_results[0]['flipped_bbox']).float()
                # kp2d = tt(tracking_results[0]['keypoints'])
                # flipped_kp2d = tt(tracking_results[0]['flipped_keypoints'])
                
                # # Collect 3D poses, IMU data, and other data for selected frames
                # poses = []
                # betas = []
                # transl = []
                # joints_3d = []
                # bone_vectors_data = []  # Store bone vectors (IMU data)
                
                for frame_idx in [0]:
                    if frame_idx < len(frames):
                        # Extract 3D joint positions
                        #joints_3d.append(frames[frame_idx]['joints_3d'] / 1000.0)  # Convert mm to m
                        pose3d = np.array(frames[frame_idx]['joints_gt'] / 1000.0).T
                        num_pts = pose3d.shape[1]
                        pts = np.ones([4, num_pts])
                        pts[:3, :] = pose3d
                        xyz = np.dot(extrinsics, pts)
                        print(f'joints_3d: {xyz[:3, :].T}')
                        joints_3d.append(xyz[:3, :].T)
                        
                        
                        # Extract bone vectors (IMU data)
                        if 'bone_vec' in frames[frame_idx]:
                            bone_vectors_data.append(frames[frame_idx]['bone_vec'])
                        else:
                            # If bone_vec not available for this frame, add a placeholder
                            bone_vectors_data.append(None)
                        
                        # If SMPL parameters are not directly available, use placeholders
                        poses.append(np.zeros(72))  # Placeholder for SMPL pose parameters
                        betas.append(np.zeros(10))  # Placeholder for SMPL shape parameters
                        transl.append(np.zeros(3))  # Placeholder for SMPL translation
                
                # Convert lists to tensors
                if len(poses) > 0:
                    poses = tt(np.array(poses))
                    betas = tt(np.array(betas))
                    transl = tt(np.array(transl))
                    joints_3d = tt(np.array(joints_3d))
                else:
                    logger.warning(f'No valid frames found for {video_id} camera {camera_id}')
                    continue
                
                # Initialize SMPL parameters from tracking results
                init_output = smpl.get_output(
                    global_orient=tracking_results[0]['init_global_orient'],
                    body_pose=tracking_results[0]['init_body_pose'],
                    betas=tracking_results[0]['init_betas'],
                    pose2rot=False,
                    return_full_pose=True
                )
                
                init_kp3d = root_centering(init_output.joints[:, :17], 'coco')

                init_smpl = transforms.matrix_to_rotation_6d(init_output.full_pose)
                init_root = transforms.matrix_to_rotation_6d(init_output.global_orient)
                init_pose = transforms.matrix_to_axis_angle(torch.cat((
                    tracking_results[0]['init_global_orient'], 
                    tracking_results[0]['init_body_pose']
                ), dim=1))
                
                # Flipped initial predictions
                flipped_init_output = smpl.get_output(
                    global_orient=tracking_results[0]['flipped_init_global_orient'],
                    body_pose=tracking_results[0]['flipped_init_body_pose'],
                    betas=tracking_results[0]['flipped_init_betas'],
                    pose2rot=False,
                    return_full_pose=True
                )
                
                flipped_init_kp3d = root_centering(flipped_init_output.joints[:, :17], 'coco')

                flipped_init_smpl = transforms.matrix_to_rotation_6d(flipped_init_output.full_pose)
                flipped_init_root = transforms.matrix_to_rotation_6d(flipped_init_output.global_orient)
                flipped_init_pose = transforms.matrix_to_axis_angle(torch.cat((
                    tracking_results[0]['flipped_init_global_orient'], 
                    tracking_results[0]['flipped_init_body_pose']
                ), dim=1))
                
                # Store data in the dataset dictionary
                sequence_id = f"{video_id}_{camera_id}"
                dataset['vid'].append(sequence_id)
                dataset['res'].append(res)
                dataset['pose'].append(poses)
                dataset['frame_id'].append(frame_ids)
                dataset['cam_poses'].append(tt(extrinsics).unsqueeze(0).repeat(len(frame_ids), 1, 1))
                dataset['features'].append(tracking_results[0]['features'])
                dataset['flipped_features'].append(tracking_results[0]['flipped_features'])
                dataset['betas'].append(betas)
                dataset['transl'].append(transl)
                dataset['gender'].append('neutral')  # Adjust if gender info is available
                dataset['kp2d'].append(kp2d)
                dataset['bbox'].append(bbox)
                dataset['flipped_bbox'].append(flipped_bbox)
                dataset['flipped_kp2d'].append(flipped_kp2d)
                dataset['init_kp3d'].append(init_kp3d)
                dataset['init_pose'].append(init_pose.cpu())
                dataset['flipped_init_kp3d'].append(flipped_init_kp3d)
                dataset['flipped_init_pose'].append(flipped_init_pose.cpu())
                dataset['init_smpl'].append(init_smpl)
                dataset['init_root'].append(init_root)
                dataset['flipped_init_smpl'].append(flipped_init_smpl)
                dataset['flipped_init_root'].append(flipped_init_root)
                
                # Add TotalCapture specific fields
                dataset['joints_3d'].append(joints_3d)
                dataset['subject'].append(frames[0]['subject'])
                dataset['action'].append(frames[0]['action'])
                
                # Add IMU data (bone vectors)
                dataset['bone_vectors'].append(bone_vectors_data)

                torch.save(dataset, save_pth)
    
    # Save processed data
    #torch.save(dataset, save_pth)
    logger.info(f'==> Saved preprocessed data to {save_pth}')
    return dataset

def process_bone_vectors(bone_vectors_dict):
    """
    Process bone vectors dictionary into a format suitable for model input
    
    Args:
        bone_vectors_dict: Dictionary containing bone vectors for each frame
        
    Returns:
        Processed bone vectors as numpy array or None if not available
    """
    if not bone_vectors_dict:
        return None
    
    # Example processing - adjust according to your model requirements
    # This depends on how your model expects to receive the bone vector data
    processed_data = []
    
    for bone_name, vector in bone_vectors_dict.items():
        processed_data.append(vector)
    
    return np.array(processed_data)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--pickle_dir', type=str, required=True, help='Path to TotalCapture pickle files')
    args = parser.parse_args()
    
    # Find pickle files
    totalcapture_files = [
        os.path.join(args.pickle_dir, 'totalcapture_train.pkl'),
        #os.path.join(args.pickle_dir, 'totalcapture_validation.pkl')
    ]
    
    # Check if files exist
    for file_path in totalcapture_files:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found!")
    
    preprocess_totalcapture(totalcapture_files, args, args.batch_size)
