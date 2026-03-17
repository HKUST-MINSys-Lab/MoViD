import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar
from loguru import logger
import time

# Lazy import for renderer (requires pytorch3d)
try:
    from lib.vis.renderer import Renderer, get_global_cameras
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    logger.warning("pytorch3d not available. 3D mesh rendering will be disabled.")

def run_skeleton_vis_streaming(cfg, video_path, frame_buffer, output_path, smpl, vis_global=True):
    """Real-time visualization of skeleton from streaming data with better error handling and synchronization"""
    if not RENDERER_AVAILABLE:
        raise ImportError("pytorch3d is required for 3D mesh rendering. Please install it or disable mesh rendering.")
    import imageio
    from lib.vis.renderer import Renderer
    
    # Initialize video parameters
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()  # Close the video capture as we'll read frames from the buffer
    
    # Setup renderer and analyzer
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)
    
    # Create writer with error handling
    try:
        output_file = osp.join(output_path, 'output_skeleton.mp4')
        writer = imageio.get_writer(output_file, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    except Exception as e:
        logger.error(f"Error creating video writer: {str(e)}")
        return
    
    bar = Bar('Rendering results...', fill='#', max=length)
    
    # Process results as they become available
    current_frame = 0
    subject_results = {}  # Store latest results for each subject
    frames_buffer = []    # Store frames
    processed_frames = set()  # Track which frames we've already processed
    
    logger.info('Starting skeleton visualization')
    
    # Keep collecting frames until we've processed all of them
    while True:
        # Get new results if available
        result = frame_buffer.get_result()
        while result is not None:
            subject_id, frame_result = result
            subject_results[subject_id] = frame_result
            result = frame_buffer.get_result()
        
        # Get frames if available
        frame = frame_buffer.get_frame()
        if frame is not None:
            frames_buffer.append(frame)
        
        # Process frames that have both frame data and results
        while current_frame < len(frames_buffer) and len(subject_results) > 0:
            # Skip frames we've already processed
            if current_frame in processed_frames:
                current_frame += 1
                continue
            
            # Get the current frame
            org_img = frames_buffer[current_frame]
            img = org_img[..., ::-1].copy()  # Convert BGR to RGB
            
            # Flag to track if we've rendered anything for this frame
            rendered = False
            
            # Render skeleton for each subject
            for subject_id, result in subject_results.items():
                # Find the corresponding frame index in the results
                frame_ids = result['frame_ids']
                if isinstance(frame_ids, np.ndarray):
                    frame_i2 = np.where(frame_ids == current_frame)[0]
                else:
                    # Handle the case where frame_ids is a list or single frame
                    if hasattr(frame_ids, '__iter__'):
                        frame_i2 = [i for i, fid in enumerate(frame_ids) if fid == current_frame]
                    else:
                        # Single frame case
                        frame_i2 = [0] if frame_ids == current_frame else []
                
                if len(frame_i2) == 0:
                    continue
                
                frame_i2 = frame_i2[0]
                
                # Render 2D skeleton
                joints2d = result['joints2d']
                if len(joints2d) > 0:
                    try:
                        # Handle different array shapes
                        if len(joints2d.shape) == 3:  # [frames, joints, coords]
                            joints = torch.from_numpy(joints2d[frame_i2]).to(cfg.DEVICE)
                        elif len(joints2d.shape) == 4:  # [batch, frames, joints, coords]
                            joints = torch.from_numpy(joints2d[0, frame_i2]).to(cfg.DEVICE)
                        else:
                            # Try to handle any other unexpected shapes
                            joints = torch.from_numpy(
                                np.array(joints2d[frame_i2] if frame_i2 < len(joints2d) else joints2d[-1])
                            ).to(cfg.DEVICE)
                        
                        img = render_skeleton(joints, img)
                        rendered = True
                    except Exception as e:
                        logger.warning(f"Error rendering skeleton for frame {current_frame}: {str(e)}")
            
            # Only add frame to output if we actually rendered something
            if rendered:
                writer.append_data(img)
                processed_frames.add(current_frame)
            
            bar.next()
            current_frame += 1
        
        # Check if we're done
        if frame_buffer.stopped:
            # Check if we've processed all available frames
            if current_frame >= len(frames_buffer) and frame_buffer.results.empty():
                # Wait a bit more to see if any final frames arrive
                time.sleep(0.5)
                if frame_buffer.results.empty() and current_frame >= len(frames_buffer):
                    break
        
        # If we're waiting for more frames/results, sleep briefly
        if current_frame >= len(frames_buffer) or len(subject_results) == 0:
            time.sleep(0.05)
    
    # Close the writer and clean up
    writer.close()
    bar.finish()
    logger.info(f'Output video saved to {output_file}')
    
    # Log statistics for debugging
    logger.info(f'Processed {len(processed_frames)} frames out of {len(frames_buffer)} available')
    if len(processed_frames) < len(frames_buffer):
        logger.warning(f'Some frames were not processed: {len(frames_buffer) - len(processed_frames)} frames skipped')
        
    return

def run_vis_on_demo(cfg, video, results, output_pth, smpl, vis_global=True):
    # Check if renderer is available
    if not RENDERER_AVAILABLE:
        logger.error("pytorch3d is required for 3D mesh rendering. Please install it or run with --visualize disabled.")
        logger.info("Skipping visualization due to missing pytorch3d dependency.")
        return

    # Import Renderer here since we know it's available
    from lib.vis.renderer import Renderer, get_global_cameras

    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
        osp.join(output_pth, 'output_smpl.mp4'), 
        fps=fps, mode='I', format='FFMPEG', macro_block_size=1
    )
    bar = Bar('Rendering results ...', fill='#', max=length)
    
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        
        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(val['verts'][frame_i2]).to(cfg.DEVICE), img)
        
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
    在图像上渲染骨骼可视化，使用提供的关节位置。
    
    Args:
        joints: 关节位置张量，形状为 (num_joints, 3)
        img: 要渲染骨骼的图像
        line_thickness: 连接关节的线条粗细
        point_radius: 表示关节的点的半径
    
    Returns:
        带有渲染骨骼的图像
    """
    # 确保关节数据在CPU上以便OpenCV操作
    joints_np = joints.detach().cpu().numpy()
    
    img_h, img_w = img.shape[:2]
    joints_2d = joints_np[:, :2].copy()  # Use only x and y coordinates
    # joints_2d[:, 0] = joints_2d[:, 0]  * img_w / 2 + img_w / 2
    # joints_2d[:, 1] = joints_2d[:, 1]  * img_h / 2 + img_h / 2
    joints_2d = joints_2d.astype(np.int32)
    
    
    # 使用提供的骨骼连接
    connections = [    
                (0, 1), (0, 2), (1, 3), (2,4),  # 头部
                (6, 8), (8, 10),  # 左臂
                (7, 9), (5, 7),   # 右臂
                (5, 6), (5, 11), (6, 12),  # 躯干
                (11, 13), (13, 15),  # 左腿
                (12, 14), (14, 16),  # 右腿
                (11, 12),  # 骨盆
    ]

    
    # 绘制连接
    img_copy = img.copy()
    for i, (j1, j2) in enumerate(connections):
        # 检查关节是否在图像边界内和索引范围内
        if (j1 < len(joints_2d) and j2 < len(joints_2d) and
            0 <= joints_2d[j1, 0] < img_w and 0 <= joints_2d[j1, 1] < img_h and
            0 <= joints_2d[j2, 0] < img_w and 0 <= joints_2d[j2, 1] < img_h):
            
                
            # 绘制线条
            cv2.line(img_copy, 
                    tuple(joints_2d[j1]), 
                    tuple(joints_2d[j2]), 
                    (51, 255, 51), 
                    thickness=line_thickness)
    
    # 绘制关节点
    for i, joint in enumerate(joints_2d):
        # 检查关节是否在图像边界内
        if 0 <= joint[0] < img_w and 0 <= joint[1] < img_h:
            cv2.circle(img_copy, 
                      tuple(joint), 
                      radius=point_radius, 
                      color=(255, 255, 255),  # 关节点使用白色
                      thickness=-1)  # 填充圆形
        
            # 可选：添加关节编号标签，便于调试
            # cv2.putText(img_copy, str(i), (joint[0]+5, joint[1]+5), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

def plot_3d_joints(joints3d, bones=None, save_path='3d_skeleton.png'):
    """
    可视化3D关节位置，并标注关节编号，同时保存为图片。
    Args:
        joints3d (torch.Tensor or np.ndarray): 3D关节位置，形状 (N, 3)
        bones (list of tuple): 骨骼连接，用于连线，可选
        save_path (str): 图片保存路径
    """
    joints3d = joints3d.cpu().numpy() if isinstance(joints3d, torch.Tensor) else joints3d
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制关节点
    ax.scatter(joints3d[:, 0], joints3d[:, 1], joints3d[:, 2], c='b', marker='o')

    # 标注关节编号
    for i, (x, y, z) in enumerate(joints3d):
        if i >=17:
            ax.text(x, y, z, str(i), color='red', fontsize=8)

    # 绘制骨骼连接
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
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"3D关节图已保存至 {save_path}")
    plt.show()



import os.path as osp
import numpy as np
import torch
import cv2
import imageio
from progress.bar import Bar
# Renderer is already imported at the top with lazy loading

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


def run_skeleton_vis_sequential(cfg, video_path, results, output_pth, smpl, vis_global=True):
    """
    Sequentially render skeleton visualization from pose estimation results.
    
    Args:
        cfg: Configuration object
        video_path: Path to input video
        results: Dictionary containing pose estimation results
        output_pth: Output path for rendered video
        smpl: SMPL model object
        vis_global: Whether to visualize in global coordinates
    """
    try:
        import imageio
        from lib.vis.renderer import Renderer
    except ImportError:
        logger.error("Required visualization modules not found. Please install imageio and other required packages.")
        return
    
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Failed to open video at {video_path}"
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize renderer
    focal_length = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)

    
    # Set up output video writer
    output_video_path = osp.join(output_pth, 'output_skeleton.mp4')
    writer = imageio.get_writer(output_video_path, fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    
    # Progress bar for rendering
    bar = Bar('Rendering skeleton visualization...', fill='#', max=length)
    
    # Process each frame sequentially
    frame_i = 0
    while cap.isOpened():
        flag, org_img = cap.read()
        if not flag:
            break
            
        # Convert BGR to RGB for rendering
        img = org_img[..., ::-1].copy()
        
        # Process each subject in the results
        for subject_id, val in results.items():
            # Find the current frame in the results
            if 'frame_ids' not in val or len(val['frame_ids']) == 0:
                continue
                
            # Find the index of the current frame in the results
            frame_matches = np.where(np.array(val['frame_ids']) == frame_i)[0]
            if len(frame_matches) == 0:
                continue
                
            frame_i2 = frame_matches[0]
            
            # Render 2D joints if available
            if 'joints2d' in val and len(val['joints2d']) > 0:
                if isinstance(val['joints2d'], list):
                    # Handle case where joints2d is a list
                    if frame_i2 < len(val['joints2d']):
                        joints2d = val['joints2d'][frame_i2]
                        if isinstance(joints2d, np.ndarray):
                            joints2d = torch.from_numpy(joints2d).to(cfg.DEVICE)
                        img = render_skeleton(joints2d, img)
                else:
                    # Handle case where joints2d is an array
                    if frame_i2 < val['joints2d'].shape[0]:
                        joints2d = torch.from_numpy(val['joints2d'][frame_i2]).to(cfg.DEVICE)
                        img = render_skeleton(joints2d, img)
            
            # Render 3D SMPL mesh if available
            if vis_global and all(k in val for k in ['pose', 'trans', 'betas']):
                # Extract pose, translation, and shape parameters
                if frame_i2 < len(val['pose']) and frame_i2 < len(val['trans']) and frame_i2 < len(val['betas']):
                    pose = val['pose'][frame_i2] if isinstance(val['pose'], list) else val['pose'][frame_i2]
                    trans = val['trans'][frame_i2] if isinstance(val['trans'], list) else val['trans'][frame_i2]
                    betas = val['betas'][frame_i2] if isinstance(val['betas'], list) else val['betas'][frame_i2]
                    
                    # Convert to tensors
                    pose_tensor = torch.tensor(pose, device=cfg.DEVICE).float()
                    trans_tensor = torch.tensor(trans, device=cfg.DEVICE).float()
                    betas_tensor = torch.tensor(betas, device=cfg.DEVICE).float()
                    
                    # Get SMPL output
                    output = smpl(
                        global_orient=pose_tensor[:3].unsqueeze(0),
                        body_pose=pose_tensor[3:].unsqueeze(0),
                        betas=betas_tensor.unsqueeze(0),
                        transl=trans_tensor.unsqueeze(0),
                    )
                    
                    # Render mesh
                    verts = output.vertices.squeeze().cpu().numpy()
                    img = renderer(verts, img)
            
            # Optionally compute and render gait metrics
            if 'joints3d' in val and len(val['joints3d']) > 0 and False:  # Currently disabled (same as original)
                joints3d = val['joints3d'][frame_i2]
                # Calculate and render gait metrics
                step_length, left_knee_angle, right_knee_angle, cadence = analyzer.calculate_gait_metrics(joints3d, frame_i)
                img = render_metrics(img, step_length, left_knee_angle, right_knee_angle, cadence)
        
        # Add the rendered frame to the output video
        writer.append_data(img)
        bar.next()
        frame_i += 1
    
    # Cleanup
    cap.release()
    writer.close()
    bar.finish()
    
    logger.info(f"Skeleton visualization saved to {output_video_path}")

def run_skeleton_vis(cfg, video, results, output_pth, smpl, vis_global=True):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Debug: log results structure
    for _id, val in results.items():
        logger.info(f"Subject {_id}: keys={list(val.keys())}")
        for k, v in val.items():
            if isinstance(v, np.ndarray):
                logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                logger.info(f"  {k}: type={type(v).__name__}")

    writer = imageio.get_writer(osp.join(output_pth, 'output_skeleton.mp4'), fps=fps, mode='I', format='FFMPEG', macro_block_size=1)
    bar = Bar('Rendering results...', fill='#', max=length)

    rendered_count = 0
    frame_i = 0
    while cap.isOpened():
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        for _id, val in results.items():
            frame_ids = np.array(val['frame_ids']) if not isinstance(val['frame_ids'], np.ndarray) else val['frame_ids']
            frame_i2 = np.where(frame_ids == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            j2d = val['joints2d']
            # Handle different shapes: [B, T, J, D] or [T, J, D]
            if j2d.ndim == 4:
                joints = torch.from_numpy(j2d[0][frame_i2]).to(cfg.DEVICE)
            else:
                joints = torch.from_numpy(j2d[frame_i2]).to(cfg.DEVICE)
            img = render_skeleton(joints, img)
            rendered_count += 1
            
            if False:
                joints3d = val['joints3d'][frame_i2]
                # Calculate and render gait metrics
                step_length, left_knee_angle, right_knee_angle, cadence = analyzer.calculate_gait_metrics(joints3d, frame_i)
                img = render_metrics(img, step_length, left_knee_angle, right_knee_angle, cadence)

        writer.append_data(img)
        bar.next()
        frame_i += 1

    writer.close()
    bar.finish()
    logger.info(f"Skeleton vis: rendered {rendered_count} skeleton overlays across {frame_i} frames")
