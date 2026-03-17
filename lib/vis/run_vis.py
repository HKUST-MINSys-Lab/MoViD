import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from loguru import logger

from lib.vis.renderer import Renderer, get_global_cameras

def _render_ntu_skeleton(img, ntu_joints_2d, color=(0, 255, 255)):
    """
    Render the NTU RGB+D 25-joint skeleton onto the image

    NTU RGB+D 25 joint order:
        0: Base of spine (pelvis/mid-hip)
        1: Mid spine
        2: Neck
        3: Head
        4: Left shoulder
        5: Left elbow
        6: Left wrist
        7: Left hand
        8: Right shoulder
        9: Right elbow
        10: Right wrist
        11: Right hand
        12: Left hip
        13: Left knee
        14: Left ankle
        15: Left foot
        16: Right hip
        17: Right knee
        18: Right ankle
        19: Right foot
        20: Spine (between neck and mid-spine)
        21: Tip of left hand
        22: Left thumb
        23: Tip of right hand
        24: Right thumb

    Args:
        img: BGR image (numpy array)
        ntu_joints_2d: NTU 25-joint 2D coordinates (25, 2)
        color: skeleton line color (B, G, R)

    Returns:
        rendered BGR image
    """
    h, w = img.shape[:2]

    # NTU skeleton connections (based on the correct NTU 25 joint order)
    ntu_skeleton = [
        # torso (spine): 0-1-20-2-3
        (0, 1), (1, 20), (20, 2), (2, 3),
        # left arm: 2-4-5-6-7, 7-21, 7-22 (connected from the neck to the shoulder)
        (2, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        # right arm: 2-8-9-10-11, 11-23, 11-24 (connected from the neck to the shoulder)
        (2, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        # left leg: 0-12-13-14-15
        (0, 12), (12, 13), (13, 14), (14, 15),
        # right leg: 0-16-17-18-19
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]

    # Draw skeleton connections
    for start_idx, end_idx in ntu_skeleton:
        if start_idx < len(ntu_joints_2d) and end_idx < len(ntu_joints_2d):
            pt1 = ntu_joints_2d[start_idx]
            pt2 = ntu_joints_2d[end_idx]

            # Ensure pt1 and pt2 are scalars
            pt1 = pt1.flatten() if isinstance(pt1, np.ndarray) else pt1
            pt2 = pt2.flatten() if isinstance(pt2, np.ndarray) else pt2

            # Check whether the points are in a valid range
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])

            if (0 <= x1 < w and 0 <= y1 < h and
                0 <= x2 < w and 0 <= y2 < h):
                cv2.line(img, (int(x1), int(y1)),
                        (int(x2), int(y2)), color, 2)

    # Draw keypoints
    for i, pt in enumerate(ntu_joints_2d):
        # Ensure pt is a scalar
        pt = pt.flatten() if isinstance(pt, np.ndarray) else pt
        x, y = float(pt[0]), float(pt[1])

        if 0 <= x < w and 0 <= y < h:
            # Use different colors for different body parts
            if i in [0, 1, 2, 3, 20]:  # torso
                pt_color = (0, 255, 0)  # green
            elif i in [4, 5, 6, 7, 21, 22]:  # left arm
                pt_color = (255, 0, 0)  # blue
            elif i in [8, 9, 10, 11, 23, 24]:  # right arm
                pt_color = (0, 0, 255)  # red
            elif i in [12, 13, 14, 15]:  # left leg
                pt_color = (255, 255, 0)  # cyan
            else:  # right leg
                pt_color = (255, 0, 255)  # magenta

            cv2.circle(img, (int(x), int(y)), 4, pt_color, -1)
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)

    return img

def _project_ntu_joints_simple(ntu_joints_3d, width, height):
    """
    Simple orthographic projection: project NTU 3D keypoints to 2D
    This is a fallback method that uses simple projection
    """
    # Simple orthographic projection: use only x and y coordinates, with z reserved for depth ordering
    # Normalize 3D coordinates to the image size
    joints_2d = ntu_joints_3d[:, :2].copy()  # (25, 2)
    
    # Normalize to the image center
    # Assume the 3D coordinates are in a reasonable range and apply simple scaling and translation
    center = joints_2d.mean(axis=0)
    joints_2d = joints_2d - center
    
    # Scale to fit the image - increase the scaling factor from 0.3 to 1.0 so the skeleton size matches the mesh
    scale = min(width, height) / (joints_2d.max() - joints_2d.min() + 1e-6) * 1.0
    joints_2d = joints_2d * scale
    
    # Translate to the image center
    joints_2d[:, 0] += width / 2
    joints_2d[:, 1] += height / 2
    
    return joints_2d

def _project_ntu_joints_to_2d(ntu_joints_3d, width, height, val, frame_i2, trans_cam=None, device='cuda:0'):
    """
    Project NTU 3D keypoints to 2D using camera parameters

    Args:
        ntu_joints_3d: NTU 25-joint 3D keypoints (25, 3) - already in camera coordinates (including trans_cam)
        width: image width
        height: image height
        val: result dictionary containing trans_cam, joints2d, joints3d, etc.
        frame_i2: index of the current frame in results
        trans_cam: camera translation parameters (optional, used only for the fallback path)
        device: compute device

    Returns:
        ntu_joints_2d: NTU 25-joint 2D keypoints (25, 2)
    """
    try:
        from lib.models.smpl import full_perspective_projection
        from lib.utils.imutils import compute_cam_intrinsics

        # Ensure device is a torch.device
        if isinstance(device, str):
            device = torch.device(device)

        # Convert NTU joints to a tensor
        # Important: ntu_joints_3d is extracted from verts_cam + trans_cam and is already in camera coordinates
        # Therefore no extra translation is needed during projection
        ntu_joints_3d_tensor = torch.from_numpy(ntu_joints_3d).float().to(device)
        ntu_joints_3d_tensor = ntu_joints_3d_tensor.unsqueeze(0)  # (1, 25, 3)

        # Compute camera intrinsics
        res = torch.tensor([width, height]).float()
        cam_intrinsics = compute_cam_intrinsics(res)  # (1, 3, 3)

        # Ensure cam_intrinsics has a batch dimension
        if cam_intrinsics.dim() == 2:
            cam_intrinsics = cam_intrinsics.unsqueeze(0)  # (1, 3, 3)

        cam_intrinsics = cam_intrinsics.to(device)  # (1, 3, 3)

        # Project to 2D without passing translation because ntu_joints_3d already includes trans_cam
        ntu_joints_2d = full_perspective_projection(
            ntu_joints_3d_tensor,
            cam_intrinsics,
            translation=None  # translation is not required because the coordinates are already in camera space
        )  # (1, 25, 2)

        ntu_joints_2d = ntu_joints_2d[0].cpu().numpy()  # (25, 2)
        return ntu_joints_2d

    except Exception as e:
        # Fallback to simple projection if projection fails
        import traceback
        print(f"Warning: Failed to project NTU joints: {e}")
        print(traceback.format_exc())
        return _project_ntu_joints_fallback(ntu_joints_3d, width, height, val, frame_i2)

def _project_ntu_joints_fallback(ntu_joints_3d, width, height, val, frame_i2):
    """
    Fallback method: estimate projection parameters from joints2d and joints3d
    """
    # If joints2d and joints3d are available, use them to estimate projection parameters
    if 'joints2d' in val and 'joints3d' in val and frame_i2 < len(val['joints2d']) and frame_i2 < len(val['joints3d']):
        joints2d = val['joints2d'][frame_i2]  # (17, 2) or (N, 2) COCO format
        joints3d = val['joints3d'][frame_i2]  # (17, 3) or (N, 3) COCO format
        
        # Ensure joints2d and joints3d are 2D arrays
        if joints2d.ndim == 1:
            joints2d = joints2d.reshape(-1, 2)
        if joints3d.ndim == 1:
            joints3d = joints3d.reshape(-1, 3)
        
        # Check if joints2d is normalized (range [-1, 1]) or pixel coordinates
        if joints2d.max() <= 1.0 and joints2d.min() >= -1.0:
            # Normalized coordinates, convert to pixel coordinates
            joints2d_pixel = joints2d.copy()
            joints2d_pixel[:, 0] = (joints2d_pixel[:, 0] + 1) * width / 2
            joints2d_pixel[:, 1] = (joints2d_pixel[:, 1] + 1) * height / 2
        else:
            # Already in pixel coordinates
            joints2d_pixel = joints2d
        
        # Ensure we have at least one joint
        if joints2d_pixel.shape[0] > 0 and joints3d.shape[0] > 0:
            # Use COCO joints to estimate projection parameters
            # Use pelvis (joint 0) and several keypoints as references
            pelvis_2d = joints2d_pixel[0].flatten()  # COCO pelvis 2D, ensure shape (2,)
            pelvis_3d = joints3d[0].flatten() if joints3d[0].ndim > 1 else joints3d[0]  # COCO pelvis 3D, ensure shape (3,)
            
            # Use multiple points to estimate focal length and projection parameters
            # Choose several stable keypoints: pelvis, left_hip, right_hip, and neck
            ref_indices = [0]  # pelvis
            if joints3d.shape[0] > 11:
                ref_indices.append(11)  # left_hip
            if joints3d.shape[0] > 12:
                ref_indices.append(12)  # right_hip
            if joints3d.shape[0] > 15:
                ref_indices.append(15)  # neck/head
            
            # Estimate focal length and scale
            valid_refs = []
            for idx in ref_indices:
                if idx < len(joints3d):
                    joint_3d = joints3d[idx].flatten() if joints3d[idx].ndim > 1 else joints3d[idx]
                    joint_2d = joints2d_pixel[idx].flatten() if joints2d_pixel[idx].ndim > 1 else joints2d_pixel[idx]
                    if len(joint_3d) >= 3 and joint_3d[2] > 0:
                        valid_refs.append((joint_2d, joint_3d))
            
            if len(valid_refs) > 0:
                # Estimate scale from the average depth
                avg_depth = np.mean([ref[1][2] for ref in valid_refs])
                # Use the same focal-length computation as the renderer
                focal_length = (width ** 2 + height ** 2) ** 0.5
                fx = fy = focal_length
                cx, cy = width / 2, height / 2
                
                # Project NTU joints using the correct perspective projection formula
                ntu_joints_2d = np.zeros((25, 2))
                for i in range(25):
                    if ntu_joints_3d[i, 2] > 0:
                        # Correct perspective projection: x' = fx * X/Z + cx, y' = fy * Y/Z + cy
                        ntu_joints_2d[i, 0] = fx * ntu_joints_3d[i, 0] / ntu_joints_3d[i, 2] + cx
                        ntu_joints_2d[i, 1] = fy * ntu_joints_3d[i, 1] / ntu_joints_3d[i, 2] + cy
                    else:
                        # Fallback: use x, y directly with scaling (Use a larger scaling factor)
                        scale_fallback = focal_length / (avg_depth + 1e-6) if avg_depth > 0 else width * 0.5
                        ntu_joints_2d[i, 0] = ntu_joints_3d[i, 0] * scale_fallback + width / 2
                        ntu_joints_2d[i, 1] = ntu_joints_3d[i, 1] * scale_fallback + height / 2
                
                # Align to the pelvis position (NTU joint 0 corresponds to the COCO pelvis)
                ntu_pelvis_2d = ntu_joints_2d[0].flatten()  # Ensure shape (2,)
                pelvis_2d_flat = pelvis_2d.flatten() if pelvis_2d.ndim > 1 else pelvis_2d
                
                if (len(ntu_pelvis_2d) == 2 and len(pelvis_2d_flat) == 2 and 
                    not (np.isnan(ntu_pelvis_2d).any() or np.isnan(pelvis_2d_flat).any())):
                    offset = pelvis_2d_flat - ntu_pelvis_2d  # Shape (2,)
                    ntu_joints_2d = ntu_joints_2d + offset  # Broadcasting (25, 2) + (2,)
                
                return ntu_joints_2d
    
    # Final fallback to simple projection
    return _project_ntu_joints_simple(ntu_joints_3d, width, height)

def draw_motiongpt_predictions(img, motiongpt_predictions, frame_i, width, height, chunk_size=100):
    """
    Draw MotionGPT prediction text on the image (standalone visualization function)
    
    Args:
        img: RGB image (numpy array)
        motiongpt_predictions: MotionGPT prediction list
        frame_i: current frame index
        width: image width
        height: image height
        chunk_size: number of frames per chunk (default 100)
    
    Returns:
        rendered RGB image
    """
    if not motiongpt_predictions:
        return img
    
    # Switch chunks by frame count: one chunk every chunk_size frames
    chunk_idx = frame_i // chunk_size
    current_pred = None
    
    # Find the corresponding prediction by chunk index
    if chunk_idx < len(motiongpt_predictions):
        current_pred = motiongpt_predictions[chunk_idx]
    
    if not current_pred:
        return img
    
    # Ensure BGR format by converting from the current RGB image
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = img[..., ::-1].copy()  # RGB to BGR
    else:
        img_bgr = img.copy()
    
    # Match the classifier: use a 1920x1080 diagonal reference and base_font_scale=1.2 for larger, clearer text
    reference_diagonal = np.sqrt(1920**2 + 1080**2)  # ~2203
    current_diagonal = np.sqrt(width**2 + height**2)
    scale_factor = current_diagonal / reference_diagonal
    base_font_scale = 1.2
    font_scale = base_font_scale * scale_factor
    font_scale = max(0.5, min(font_scale, 3.0))
    thickness = max(2, int(font_scale * 2.5))
    
    # Position at top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Adjust the position based on the video height
    y_start = int(height * 0.04)  # 4% from top
    y_start = max(30, min(y_start, 100))  # Clamp between 30 and 100 pixels
    x_start = int(width * 0.01)  # 1% from left
    x_start = max(10, min(x_start, 50))  # Clamp between 10 and 50 pixels
    
    # Format text: [action_label] description
    action_label = current_pred['action_label']
    description = current_pred['description']
    # Truncate description if too long (Adjust according to the video width)
    max_desc_len = int(width / 15)  # Dynamically adjust according to the video width
    max_desc_len = max(40, min(max_desc_len, 80))  # Clamp between 40 and 80 characters
    if len(description) > max_desc_len:
        description = description[:max_desc_len] + "..."
    
    text_line1 = f"[{action_label}]"
    text_line2 = description
    
    # Calculate text sizes (getTextSize: text sits above baseline; box must enclose both lines + padding)
    (text_width1, text_height1), baseline1 = cv2.getTextSize(text_line1, font, font_scale, thickness)
    (text_width2, text_height2), baseline2 = cv2.getTextSize(text_line2, font, font_scale * 0.9, thickness)
    
    max_width = max(text_width1, text_width2)
    text_x = x_start + 5
    pad = 8
    box_y1 = y_start - text_height1 - pad
    box_y2 = y_start + (text_height1 + baseline1 + 5 + text_height2 + baseline2) + pad
    box_x1 = x_start
    box_x2 = text_x + max_width + pad
    
    # Draw semi-transparent background
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, img_bgr)
    
    # Draw border (magenta/pink color: BGR format)
    border_thickness = max(2, int(thickness * 0.8))
    cv2.rectangle(img_bgr, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 255), border_thickness)
    
    # Draw action label (magenta in BGR) - in the top-left corner
    cv2.putText(img_bgr, text_line1, (text_x, y_start), font, font_scale, (255, 0, 255), thickness)
    
    # Draw description (cyan in BGR, not white) - below the action label
    cv2.putText(img_bgr, text_line2, (text_x, y_start + text_height1 + baseline1 + 5),
               font, font_scale * 0.9, (255, 255, 0), thickness)  # BGR cyan
    
    # Convert back to RGB
    img = img_bgr[..., ::-1].copy()
    return img

def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=True):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)
    
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # create renderer with cliff focal length estimation
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    if vis_global:
        # setup global coordinate subject
        # current implementation only visualize the subject appeared longest
        n_frames = {k: len(results[k]['frame_ids']) for k in results.keys()}
        sid = max(n_frames, key=n_frames.get)
        global_output = smpl.get_output(
            body_pose=tt(results[sid]['pose_world'][:, 3:]), 
            global_orient=tt(results[sid]['pose_world'][:, :3]),
            betas=tt(results[sid]['betas']),
            transl=tt(results[sid]['trans_world']))
        verts_glob = global_output.vertices.cpu()
        verts_glob[..., 1] = verts_glob[..., 1] - verts_glob[..., 1].min()
        cx, cz = (verts_glob.mean(1).max(0)[0] + verts_glob.mean(1).min(0)[0])[[0, 2]] / 2.0
        sx, sz = (verts_glob.mean(1).max(0)[0] - verts_glob.mean(1).min(0)[0])[[0, 2]]
        scale = max(sx.item(), sz.item()) * 1.5
        
        # set default ground
        renderer.set_ground(scale, cx.item(), cz.item())
        
        # build global camera
        global_R, global_T, global_lights = get_global_cameras(verts_glob, cfg.DEVICE)
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
        osp.join(output_pth, 'output_classifier.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    # Load MotionGPT output.txt if it exists
    output_txt_path = osp.join(output_pth, 'output.txt')
    logger.info(f"Looking for output.txt at: {output_txt_path}")
    motiongpt_predictions = []
    if osp.exists(output_txt_path):
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Subject'):
                        continue
                    # Parse line: start_time end_time [action_label] description
                    # Format: "0.00 5.00 [walk] a person walks / slowly" or "5.00 10.00 [walk forward] a person walks..."
                    # First locate the first `[` and its matching `]`
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # Extract time
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # Extract action_label (inside `[]`, it may contain spaces)
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # Extract the description (the content after `]`)
                                description = line[end_bracket+1:].strip()
                                motiongpt_predictions.append({
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'action_label': action_label,
                                    'description': description
                                })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipping line '{line}': {e}")
                            continue
            if motiongpt_predictions:
                logger.info(f"Loaded {len(motiongpt_predictions)} MotionGPT predictions from {output_txt_path}")
                # Debug: print all predictions
                for i, pred in enumerate(motiongpt_predictions):
                    logger.info(f"  Prediction {i}: {pred['start_time']:.2f}s - {pred['end_time']:.2f}s: [{pred['action_label']}] {pred['description'][:50]}")
            else:
                logger.warning(f"No valid predictions found in {output_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to load output.txt: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    else:
        logger.info(f"output.txt not found at {output_txt_path}, skipping MotionGPT text overlay")
    
    # Build action prediction lookup: frame_id -> action info
    action_lookup = {}
    for _id, val in results.items():
        if 'action_predictions' in val:
            for pred in val['action_predictions']:
                frame_idx = pred['frame_idx']
                if frame_idx not in action_lookup:
                    action_lookup[frame_idx] = []
                action_lookup[frame_idx].append({
                    'subject_id': _id,
                    'label': pred['label'],
                    'confidence': pred['confidence'],
                    'class_idx': pred['class_idx']
                })
    
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        
        # Render NTU 25 skeleton instead of mesh
        img_bgr = img[..., ::-1].copy()  # Convert to BGR for OpenCV
        
        for _id, val in results.items():
            # Find frame index in results
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            
            # Get NTU joints if available, otherwise extract from vertices
            if 'ntu_joints' in val and frame_i2 < len(val['ntu_joints']):
                ntu_joints_3d = val['ntu_joints'][frame_i2]  # (25, 3)
            else:
                # Extract NTU joints from vertices if not available
                vertices = val['verts'][frame_i2]  # (6890, 3)
                vertices_tensor = torch.from_numpy(vertices).float().to(cfg.DEVICE).unsqueeze(0)
                ntu_joints_3d_tensor = smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints_3d = ntu_joints_3d_tensor[0].cpu().numpy()  # (25, 3)
            
            # Project NTU 3D joints to 2D using full_perspective_projection (refer to real_time.py)
            trans_cam = None
            if 'trans_cam' in val and frame_i2 < len(val['trans_cam']):
                trans_cam = val['trans_cam'][frame_i2]
            ntu_joints_2d = _project_ntu_joints_to_2d(ntu_joints_3d, width, height, val, frame_i2, trans_cam=trans_cam, device=cfg.DEVICE)
            
            # Render NTU skeleton
            img_bgr = _render_ntu_skeleton(img_bgr, ntu_joints_2d)
        
        # Convert back to RGB
        img = img_bgr[..., ::-1].copy()
        
        # Draw action recognition labels on the image (NTU action recognition)
        has_ntu_action = False
        if frame_i in action_lookup:
            # Convert image back to BGR for OpenCV
            img_bgr = img[..., ::-1].copy()
            
            # Calculate adaptive font scale based on video size
            # Use diagonal length as reference, normalized to 1920x1080 (diagonal ~2203)
            reference_diagonal = np.sqrt(1920**2 + 1080**2)  # ~2203
            current_diagonal = np.sqrt(width**2 + height**2)
            scale_factor = current_diagonal / reference_diagonal
            
            # Base font scale for 1920x1080 resolution
            base_font_scale = 1.2
            font_scale = base_font_scale * scale_factor
            
            # Ensure minimum and maximum font sizes for readability
            font_scale = max(0.5, min(font_scale, 3.0))  # Clamp between 0.5 and 3.0
            
            # Adaptive thickness based on font scale
            thickness = max(1, int(font_scale * 2.5))
            
            # Adaptive y_offset based on image height
            y_offset = int(height * 0.03)  # 3% of image height
            y_offset = max(20, min(y_offset, 100))  # Clamp between 20 and 100
            
            for i, action_info in enumerate(action_lookup[frame_i]):
                subject_id = action_info['subject_id']
                label = action_info['label']
                confidence = action_info['confidence']
                class_idx = action_info['class_idx']
                
                # Skip invalid predictions
                if class_idx < 0 or 'buffering' in label or label in ["waiting...", "rate_limiting", "insufficient_frames"]:
                    continue
                
                # Mark that valid NTU action recognition is available
                has_ntu_action = True
                
                # Prepare text
                action_text = f"Action: {label}"
                conf_text = f"Conf: {confidence:.2f}"
                
                # Draw background rectangle for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Calculate text size (box must fully enclose both lines + baseline + padding)
                (text_width1, text_height1), baseline1 = cv2.getTextSize(action_text, font, font_scale, thickness)
                (text_width2, text_height2), baseline2 = cv2.getTextSize(conf_text, font, font_scale * 0.9, thickness)
                
                max_width = max(text_width1, text_width2)
                text_x = 15
                pad = 8
                box_y1 = y_offset - text_height1 - pad
                box_y2 = y_offset + (text_height1 + baseline1 + 5 + text_height2 + baseline2) + pad
                box_x1 = 10
                box_x2 = text_x + max_width + pad
                total_height = (text_height1 + baseline1 + 5 + text_height2 + baseline2) + 2 * pad
                
                # Draw semi-transparent background
                overlay = img_bgr.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
                
                # Draw border with adaptive thickness
                border_thickness = max(1, int(thickness * 0.67))
                cv2.rectangle(img_bgr, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), border_thickness)
                
                # Draw action label (green, larger)
                cv2.putText(img_bgr, action_text, (text_x, y_offset), font, font_scale, (0, 255, 0), thickness)
                
                # Draw confidence (yellow)
                cv2.putText(img_bgr, conf_text, (text_x, y_offset + text_height1 + baseline1 + 5),
                           font, font_scale * 0.9, (0, 255, 255), thickness)
                
                y_offset += total_height + 10  # Move down for next subject
            
            # Convert back to RGB
            img = img_bgr[..., ::-1].copy()
        
        # output_classifier.mp4 does not show MotionGPT text; it only shows NTU action recognition
        
        if vis_global:
            # render the global coordinate
            if frame_i in results[sid]['frame_ids']:
                frame_i3 = np.where(results[sid]['frame_ids'] == frame_i)[0]
                verts = verts_glob[[frame_i3]].to(cfg.DEVICE)
                faces = renderer.faces.clone().squeeze(0)
                colors = torch.ones((1, 4)).float().to(cfg.DEVICE); colors[..., :3] *= 0.9
                
                if _global_R is None:
                    _global_R = global_R[frame_i3].clone(); _global_T = global_T[frame_i3].clone()
                cameras = renderer.create_camera(global_R[frame_i3], global_T[frame_i3])
                img_glob = renderer.render_with_ground(verts, faces, colors, cameras, global_lights)
            
            try: img = np.concatenate((img, img_glob), axis=1)
            except: img = np.concatenate((img, np.ones_like(img) * 255), axis=1)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()
def render_skeleton(joints, img, line_thickness=2, point_radius=4):
    """
    Render skeleton visualization on the image using the provided joint positions.
    
    Args:
        joints: joint position tensor with shape (num_joints, 3)
        img: image to render the skeleton onto
        line_thickness: line thickness for connecting joints
        point_radius: radius of the points used to draw joints
    
    Returns:
        image with the rendered skeleton
    """
    # Ensure the joint data is on CPU for OpenCV operations
    joints_np = joints.detach().cpu().numpy()
    
    img_h, img_w = img.shape[:2]
    joints_2d = joints_np[:, :2].copy()  # Use only x and y coordinates
    # joints_2d[:, 0] = joints_2d[:, 0]  * img_w / 2 + img_w / 2
    # joints_2d[:, 1] = joints_2d[:, 1]  * img_h / 2 + img_h / 2
    joints_2d = joints_2d.astype(np.int32)
    
    
    # Use the provided bone connections
    connections = [    
                (0, 1), (0, 2), (1, 3), (2,4),  # head
                (6, 8), (8, 10),  # left arm
                (7, 9), (5, 7),   # right arm
                (5, 6), (5, 11), (6, 12),  # torso
                (11, 13), (13, 15),  # left leg
                (12, 14), (14, 16),  # right leg
                (11, 12),  # pelvis
    ]

    
    # Draw connections
    img_copy = img.copy()
    for i, (j1, j2) in enumerate(connections):
        # Check whether the joints are inside image bounds and within the valid index range
        if (j1 < len(joints_2d) and j2 < len(joints_2d) and
            0 <= joints_2d[j1, 0] < img_w and 0 <= joints_2d[j1, 1] < img_h and
            0 <= joints_2d[j2, 0] < img_w and 0 <= joints_2d[j2, 1] < img_h):
            
                
            # Draw lines
            cv2.line(img_copy, 
                    tuple(joints_2d[j1]), 
                    tuple(joints_2d[j2]), 
                    (51, 255, 51), 
                    thickness=line_thickness)
    
    # Draw joint points
    for i, joint in enumerate(joints_2d):
        # Check whether the joints are inside image bounds
        if 0 <= joint[0] < img_w and 0 <= joint[1] < img_h:
            cv2.circle(img_copy, 
                      tuple(joint), 
                      radius=point_radius, 
                      color=(255, 255, 255),  # use white for joint points
                      thickness=-1)  # filled circle
        
            # Optional: add joint index labels for debugging
            # cv2.putText(img_copy, str(i), (joint[0]+5, joint[1]+5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_3d_joints(joints3d, bones=None, save_path='3d_skeleton.png'):
    """
    Visualize 3D joint positions, annotate joint indices, and save the image.
    Args:
        joints3d (torch.Tensor or np.ndarray): 3D joint positions, shape (N, 3)
        bones (list of tuple): bone connections used for drawing lines, optional
        save_path (str): path to save the image
    """
    joints3d = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw joint points
    ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2], c='b', marker='o')

    # Annotate joint indices
    for i, (x, y, z) in enumerate(joints3d):
        if i >=17:
            ax.text(x, y, z, str(i), color='red', fontsize=8)

    # Draw bone connections
    if bones:
        for (j1, j2) in bones:
            x_vals = [joints3d[j1, 0], joints3d[j2, 0]]
            y_vals = [joints3d[j1, 1], joints3d[j2, 1]]
            z_vals = [joints3d[j1, 2], joints3d[j2, 2]]
            ax.plot(x_vals, y_vals, z_vals, c='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Joints Visualization')
    
    # Save the image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D joint plot saved to {save_path}")
    plt.show()



import os.path as osp
import numpy as np
import torch
import cv2
import imageio
from progress.bar import Bar
from lib.vis.renderer import Renderer

class GaitAnalyzer:
    def __init__(self, fps):
        self.fps = fps
        self.step_events = []
        self.prev_sign = 0
        self.current_step_max_length = 0
        self.step_lengths = []

    def calculate_gait_metrics(self, joints3d, frame_i):
        """
        Calculate gait metrics like step length, cadence (steps per minute), and knee angles.
        Step length is calculated as the average of maximum distances between ankles for each step.
        joints3d: (17, 3) numpy array representing 3D joint positions
        """
        # Define key joints
        left_hip, right_hip = joints3d[11], joints3d[12]
        left_knee, right_knee = joints3d[13], joints3d[14]
        left_ankle, right_ankle = joints3d[15], joints3d[16]

        # Calculate current distance between ankles
        current_distance = np.linalg.norm(left_ankle - right_ankle)
        
        # Update max distance for current step
        self.current_step_max_length = max(self.current_step_max_length, current_distance)

        # Knee angle using law of cosines
        def calculate_angle(a, b, c):
            ab = a - b
            cb = c - b
            cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Detect step events using zero-crossing of ankle height difference
        ankle_height_diff = left_ankle[1] - right_ankle[1]
        if abs(ankle_height_diff) < 0.1: ankle_height_diff = 0
        current_sign = np.sign(ankle_height_diff)
        
        # If sign changes, we've detected a step
        if current_sign != self.prev_sign and current_sign != 0:
            # Store the max step length from the completed step
            if len(self.step_events) > 0:  # Not the first step event
                self.step_lengths.append(self.current_step_max_length)
            
            self.step_events.append(frame_i)
            self.prev_sign = current_sign
            self.current_step_max_length = current_distance  # Reset for next step

        # Calculate average of maximum step lengths
        step_length = np.mean(self.step_lengths) if len(self.step_lengths) > 0 else current_distance

        # Calculate cadence (steps per minute)
        if len(self.step_events) > 1:
            time_elapsed = (self.step_events[-1] - self.step_events[0]) / self.fps
            cadence = ((len(self.step_events)-1) / time_elapsed ) if time_elapsed > 0 else 0
        else:
            cadence = 0

        return step_length, left_knee_angle, right_knee_angle, cadence

def render_metrics(img, step_length, left_knee_angle, right_knee_angle, cadence, font_scale=1):
    """ Render the calculated metrics on the image. """
    cv2.putText(img, f"Cadence: {cadence:.2f} steps/sec", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(255, 255, 255), thickness=2)
    cv2.putText(img, f"Step Length: {step_length:.2f}m", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(255, 255, 255), thickness=2)
    cv2.putText(img, f"Left Knee Angle: {left_knee_angle:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(255, 255, 255), thickness=2)
    cv2.putText(img, f"Right Knee Angle: {right_knee_angle:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color=(255, 255, 255), thickness=2)
    return img

def run_skeleton_vis(cfg, video, results, output_pth, smpl, vis_global=True):
    """
    Render skeleton visualization using the NTU 60 skeleton format (see real_time.py)
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    analyzer = GaitAnalyzer(fps)

    # Load MotionGPT output.txt if it exists
    output_txt_path = osp.join(output_pth, 'output.txt')
    motiongpt_predictions = []
    if osp.exists(output_txt_path):
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Subject'):
                        continue
                    # Parse line: start_time end_time [action_label] description
                    # Format: "0.00 5.00 [walk] a person walks / slowly" or "5.00 10.00 [walk forward] a person walks..."
                    # First locate the first `[` and its matching `]`
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # Extract time
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # Extract action_label (inside `[]`, it may contain spaces)
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # Extract the description (the content after `]`)
                                description = line[end_bracket+1:].strip()
                                motiongpt_predictions.append({
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'action_label': action_label,
                                    'description': description
                                })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipping line '{line}': {e}")
                            continue
            if motiongpt_predictions:
                logger.info(f"Loaded {len(motiongpt_predictions)} MotionGPT predictions from {output_txt_path}")
            else:
                logger.warning(f"No valid predictions found in {output_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to load output.txt: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    else:
        logger.info(f"output.txt not found at {output_txt_path}, skipping MotionGPT text overlay")

    writer = imageio.get_writer(osp.join(output_pth, 'output_classifier.mp4'), fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    bar = Bar('Rendering NTU skeleton results...', fill='#', max=length)

    # Build action prediction lookup: frame_id -> action info (for NTU action recognition)
    action_lookup = {}
    for _id, val in results.items():
        if 'action_predictions' in val:
            for pred in val['action_predictions']:
                frame_idx = pred['frame_idx']
                if frame_idx not in action_lookup:
                    action_lookup[frame_idx] = []
                action_lookup[frame_idx].append({
                    'subject_id': _id,
                    'label': pred['label'],
                    'confidence': pred['confidence'],
                    'class_idx': pred['class_idx']
                })

    frame_i = 0
    while cap.isOpened():
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        img_bgr = img[..., ::-1].copy()  # Convert to BGR for OpenCV

        for _id, val in results.items():
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]

            # Get NTU joints (refer to real_time.py helper used in)
            if 'ntu_joints' in val and frame_i2 < len(val['ntu_joints']):
                ntu_joints_3d = val['ntu_joints'][frame_i2]  # (25, 3)
            else:
                # Extract NTU joints from vertices
                vertices = val['verts'][frame_i2]  # (6890, 3)
                vertices_tensor = torch.from_numpy(vertices).float().to(cfg.DEVICE).unsqueeze(0)
                ntu_joints_3d_tensor = smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints_3d = ntu_joints_3d_tensor[0].cpu().numpy()  # (25, 3)

            # Project with trans_cam (see full_perspective_projection in real_time.py)
            trans_cam = None
            if 'trans_cam' in val and frame_i2 < len(val['trans_cam']):
                trans_cam = val['trans_cam'][frame_i2]

            ntu_joints_2d = _project_ntu_joints_to_2d(
                ntu_joints_3d, width, height, val, frame_i2,
                trans_cam=trans_cam, device=cfg.DEVICE
            )

            # Render the NTU skeleton (always draw it)
            img_bgr = _render_ntu_skeleton(img_bgr, ntu_joints_2d)

            if False:
                joints3d = val['joints3d'][frame_i2]
                # Calculate and render gait metrics
                step_length, left_knee_angle, right_knee_angle, cadence = analyzer.calculate_gait_metrics(joints3d, frame_i)
                img_bgr = render_metrics(img_bgr, step_length, left_knee_angle, right_knee_angle, cadence)

        # Convert back to RGB for imageio
        img = img_bgr[..., ::-1].copy()
        
        # output_classifier.mp4 does not show MotionGPT text; it only shows the NTU skeleton
        writer.append_data(img)
        bar.next()
        frame_i += 1

    writer.close()
    bar.finish()
    logger.info(f"NTU skeleton video saved to: {osp.join(output_pth, 'output_classifier.mp4')}")

def run_gpt_text_vis(cfg, video, results, output_pth, smpl, vis_global=True):
    """
    Render MotionGPT text predictions + NTU skeleton (standalone visualization branch)
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load MotionGPT output.txt if it exists
    output_txt_path = osp.join(output_pth, 'output.txt')
    logger.info(f"Looking for output.txt at: {output_txt_path}")
    motiongpt_predictions = []
    if osp.exists(output_txt_path):
        try:
            with open(output_txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('Subject'):
                        continue
                    # Parse line: start_time end_time [action_label] description
                    # Format: "0.00 5.00 [walk] a person walks / slowly" or "5.00 10.00 [walk forward] a person walks..."
                    # First locate the first `[` and its matching `]`
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # Extract time
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # Extract action_label (inside `[]`, it may contain spaces)
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # Extract the description (the content after `]`)
                                description = line[end_bracket+1:].strip()
                                motiongpt_predictions.append({
                                    'start_time': start_time,
                                    'end_time': end_time,
                                    'action_label': action_label,
                                    'description': description
                                })
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skipping line '{line}': {e}")
                            continue
            if motiongpt_predictions:
                logger.info(f"Loaded {len(motiongpt_predictions)} MotionGPT predictions from {output_txt_path}")
                # Debug: print all predictions
                for i, pred in enumerate(motiongpt_predictions):
                    logger.info(f"  Prediction {i}: {pred['start_time']:.2f}s - {pred['end_time']:.2f}s: [{pred['action_label']}] {pred['description'][:50]}")
            else:
                logger.warning(f"No valid predictions found in {output_txt_path}")
        except Exception as e:
            logger.warning(f"Failed to load output.txt: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    else:
        logger.warning(f"output.txt not found at {output_txt_path}, cannot visualize GPT text")

    writer = imageio.get_writer(osp.join(output_pth, 'output_gpt_text.mp4'), fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    bar = Bar('Rendering GPT text + NTU skeleton visualization...', fill='#', max=length)

    frame_i = 0
    while cap.isOpened():
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()  # Convert to RGB
        img_bgr = img[..., ::-1].copy()  # Convert to BGR for OpenCV
        
        # Render NTU skeleton (always draw)
        for _id, val in results.items():
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]

            # Get NTU joints
            if 'ntu_joints' in val and frame_i2 < len(val['ntu_joints']):
                ntu_joints_3d = val['ntu_joints'][frame_i2]  # (25, 3)
            else:
                # Extract NTU joints from vertices
                vertices = val['verts'][frame_i2]  # (6890, 3)
                vertices_tensor = torch.from_numpy(vertices).float().to(cfg.DEVICE).unsqueeze(0)
                ntu_joints_3d_tensor = smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints_3d = ntu_joints_3d_tensor[0].cpu().numpy()  # (25, 3)

            # Project using trans_cam
            trans_cam = None
            if 'trans_cam' in val and frame_i2 < len(val['trans_cam']):
                trans_cam = val['trans_cam'][frame_i2]

            ntu_joints_2d = _project_ntu_joints_to_2d(
                ntu_joints_3d, width, height, val, frame_i2,
                trans_cam=trans_cam, device=cfg.DEVICE
            )

            # Render the NTU skeleton
            img_bgr = _render_ntu_skeleton(img_bgr, ntu_joints_2d)
        
        # Convert back to RGB
        img = img_bgr[..., ::-1].copy()
        
        # Display MotionGPT predictions (draw text)
        img = draw_motiongpt_predictions(img, motiongpt_predictions, frame_i, width, height, chunk_size=100)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1

    writer.close()
    bar.finish()
    logger.info(f"GPT text + NTU skeleton visualization saved to: {osp.join(output_pth, 'output_gpt_text.mp4')}")
