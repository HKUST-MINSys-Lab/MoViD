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
    渲染 NTU RGB+D 25 关键点骨架到图像上

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
        img: BGR图像 (numpy array)
        ntu_joints_2d: NTU 25关键点2D坐标 (25, 2)
        color: 骨架线条颜色 (B, G, R)

    Returns:
        渲染后的BGR图像
    """
    h, w = img.shape[:2]

    # NTU 骨架连接 (基于正确的 NTU 25 joint order)
    ntu_skeleton = [
        # 躯干 (spine): 0-1-20-2-3
        (0, 1), (1, 20), (20, 2), (2, 3),
        # 左臂: 2-4-5-6-7, 7-21, 7-22 (从 neck 连接到 shoulder)
        (2, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
        # 右臂: 2-8-9-10-11, 11-23, 11-24 (从 neck 连接到 shoulder)
        (2, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
        # 左腿: 0-12-13-14-15
        (0, 12), (12, 13), (13, 14), (14, 15),
        # 右腿: 0-16-17-18-19
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]

    # 绘制骨架连线
    for start_idx, end_idx in ntu_skeleton:
        if start_idx < len(ntu_joints_2d) and end_idx < len(ntu_joints_2d):
            pt1 = ntu_joints_2d[start_idx]
            pt2 = ntu_joints_2d[end_idx]

            # 确保 pt1 和 pt2 是标量
            pt1 = pt1.flatten() if isinstance(pt1, np.ndarray) else pt1
            pt2 = pt2.flatten() if isinstance(pt2, np.ndarray) else pt2

            # 检查点是否在有效范围内
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])

            if (0 <= x1 < w and 0 <= y1 < h and
                0 <= x2 < w and 0 <= y2 < h):
                cv2.line(img, (int(x1), int(y1)),
                        (int(x2), int(y2)), color, 2)

    # 绘制关键点
    for i, pt in enumerate(ntu_joints_2d):
        # 确保 pt 是标量
        pt = pt.flatten() if isinstance(pt, np.ndarray) else pt
        x, y = float(pt[0]), float(pt[1])

        if 0 <= x < w and 0 <= y < h:
            # 不同部位使用不同颜色
            if i in [0, 1, 2, 3, 20]:  # 躯干
                pt_color = (0, 255, 0)  # 绿色
            elif i in [4, 5, 6, 7, 21, 22]:  # 左臂
                pt_color = (255, 0, 0)  # 蓝色
            elif i in [8, 9, 10, 11, 23, 24]:  # 右臂
                pt_color = (0, 0, 255)  # 红色
            elif i in [12, 13, 14, 15]:  # 左腿
                pt_color = (255, 255, 0)  # 青色
            else:  # 右腿
                pt_color = (255, 0, 255)  # 紫色

            cv2.circle(img, (int(x), int(y)), 4, pt_color, -1)
            cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)

    return img

def _project_ntu_joints_simple(ntu_joints_3d, width, height):
    """
    简单的正交投影：将NTU 3D关键点投影到2D
    这是一个fallback方法，使用简单的投影
    """
    # 简单的正交投影：只使用x和y坐标，z用于深度排序
    # 将3D坐标归一化到图像尺寸
    joints_2d = ntu_joints_3d[:, :2].copy()  # (25, 2)
    
    # 归一化到图像中心
    # 假设3D坐标在合理范围内，进行简单的缩放和平移
    center = joints_2d.mean(axis=0)
    joints_2d = joints_2d - center
    
    # 缩放以适应图像 - 增大缩放因子从0.3到1.0，使skeleton大小与mesh一致
    scale = min(width, height) / (joints_2d.max() - joints_2d.min() + 1e-6) * 1.0
    joints_2d = joints_2d * scale
    
    # 平移到图像中心
    joints_2d[:, 0] += width / 2
    joints_2d[:, 1] += height / 2
    
    return joints_2d

def _project_ntu_joints_to_2d(ntu_joints_3d, width, height, val, frame_i2, trans_cam=None, device='cuda:0'):
    """
    使用相机参数将NTU 3D关键点投影到2D

    Args:
        ntu_joints_3d: NTU 25 3D关键点 (25, 3) - 已经在相机坐标系中（包含 trans_cam）
        width: 图像宽度
        height: 图像高度
        val: 包含 trans_cam, joints2d, joints3d 等的结果字典
        frame_i2: 当前帧在 results 中的索引
        trans_cam: 相机平移参数 (可选，仅用于 fallback)
        device: 计算设备

    Returns:
        ntu_joints_2d: NTU 25 2D关键点 (25, 2)
    """
    try:
        from lib.models.smpl import full_perspective_projection
        from lib.utils.imutils import compute_cam_intrinsics

        # 确保 device 是 torch.device 类型
        if isinstance(device, str):
            device = torch.device(device)

        # 将 NTU joints 转换为 tensor
        # 重要：ntu_joints_3d 是从 verts_cam + trans_cam 提取的，已经在相机坐标系中
        # 所以投影时不需要再添加 translation
        ntu_joints_3d_tensor = torch.from_numpy(ntu_joints_3d).float().to(device)
        ntu_joints_3d_tensor = ntu_joints_3d_tensor.unsqueeze(0)  # (1, 25, 3)

        # 计算相机内参
        res = torch.tensor([width, height]).float()
        cam_intrinsics = compute_cam_intrinsics(res)  # (1, 3, 3)

        # 确保 cam_intrinsics 有 batch 维度
        if cam_intrinsics.dim() == 2:
            cam_intrinsics = cam_intrinsics.unsqueeze(0)  # (1, 3, 3)

        cam_intrinsics = cam_intrinsics.to(device)  # (1, 3, 3)

        # 投影到 2D - 不传 translation，因为 ntu_joints_3d 已经包含了 trans_cam
        ntu_joints_2d = full_perspective_projection(
            ntu_joints_3d_tensor,
            cam_intrinsics,
            translation=None  # 不需要 translation，坐标已经在相机空间
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
    Fallback 方法：使用 joints2d 和 joints3d 来估算投影参数
    """
    # 如果有joints2d和joints3d，使用它们来估算投影参数
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
            # 使用COCO joints来估算投影参数
            # 找到pelvis (joint 0) 和几个关键点作为参考
            pelvis_2d = joints2d_pixel[0].flatten()  # COCO pelvis 2D, ensure shape (2,)
            pelvis_3d = joints3d[0].flatten() if joints3d[0].ndim > 1 else joints3d[0]  # COCO pelvis 3D, ensure shape (3,)
            
            # 使用多个点来估算focal length和投影参数
            # 选择几个稳定的关键点：pelvis, left_hip, right_hip, neck
            ref_indices = [0]  # pelvis
            if joints3d.shape[0] > 11:
                ref_indices.append(11)  # left_hip
            if joints3d.shape[0] > 12:
                ref_indices.append(12)  # right_hip
            if joints3d.shape[0] > 15:
                ref_indices.append(15)  # neck/head
            
            # 估算focal length和scale
            valid_refs = []
            for idx in ref_indices:
                if idx < len(joints3d):
                    joint_3d = joints3d[idx].flatten() if joints3d[idx].ndim > 1 else joints3d[idx]
                    joint_2d = joints2d_pixel[idx].flatten() if joints2d_pixel[idx].ndim > 1 else joints2d_pixel[idx]
                    if len(joint_3d) >= 3 and joint_3d[2] > 0:
                        valid_refs.append((joint_2d, joint_3d))
            
            if len(valid_refs) > 0:
                # 使用平均depth来估算scale
                avg_depth = np.mean([ref[1][2] for ref in valid_refs])
                # 使用与renderer相同的focal length计算方式
                focal_length = (width ** 2 + height ** 2) ** 0.5
                fx = fy = focal_length
                cx, cy = width / 2, height / 2
                
                # 投影NTU joints - 使用正确的透视投影公式
                ntu_joints_2d = np.zeros((25, 2))
                for i in range(25):
                    if ntu_joints_3d[i, 2] > 0:
                        # 正确的透视投影: x' = fx * X/Z + cx, y' = fy * Y/Z + cy
                        ntu_joints_2d[i, 0] = fx * ntu_joints_3d[i, 0] / ntu_joints_3d[i, 2] + cx
                        ntu_joints_2d[i, 1] = fy * ntu_joints_3d[i, 1] / ntu_joints_3d[i, 2] + cy
                    else:
                        # Fallback: use x, y directly with scaling (使用更大的缩放因子)
                        scale_fallback = focal_length / (avg_depth + 1e-6) if avg_depth > 0 else width * 0.5
                        ntu_joints_2d[i, 0] = ntu_joints_3d[i, 0] * scale_fallback + width / 2
                        ntu_joints_2d[i, 1] = ntu_joints_3d[i, 1] * scale_fallback + height / 2
                
                # 对齐到pelvis位置（NTU joint 0 对应 COCO pelvis）
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
    在图像上绘制MotionGPT预测文本（独立的可视化函数）
    
    Args:
        img: RGB图像 (numpy array)
        motiongpt_predictions: MotionGPT预测列表
        frame_i: 当前帧索引
        width: 图像宽度
        height: 图像高度
        chunk_size: 每个chunk的帧数（默认100）
    
    Returns:
        绘制后的RGB图像
    """
    if not motiongpt_predictions:
        return img
    
    # 根据帧数切换：每chunk_size帧一个chunk
    chunk_idx = frame_i // chunk_size
    current_pred = None
    
    # 找到对应的预测（根据chunk索引）
    if chunk_idx < len(motiongpt_predictions):
        current_pred = motiongpt_predictions[chunk_idx]
    
    if not current_pred:
        return img
    
    # 确保使用 BGR 格式 - 从当前 RGB 图像转换
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = img[..., ::-1].copy()  # RGB to BGR
    else:
        img_bgr = img.copy()
    
    # 与 classifier 一致：用对角线参考 1920x1080，base_font_scale=1.2，字更大更清楚
    reference_diagonal = np.sqrt(1920**2 + 1080**2)  # ~2203
    current_diagonal = np.sqrt(width**2 + height**2)
    scale_factor = current_diagonal / reference_diagonal
    base_font_scale = 1.2
    font_scale = base_font_scale * scale_factor
    font_scale = max(0.5, min(font_scale, 3.0))
    thickness = max(2, int(font_scale * 2.5))
    
    # Position at top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 根据视频高度调整位置
    y_start = int(height * 0.04)  # 4% from top
    y_start = max(30, min(y_start, 100))  # 限制在30-100像素之间
    x_start = int(width * 0.01)  # 1% from left
    x_start = max(10, min(x_start, 50))  # 限制在10-50像素之间
    
    # Format text: [action_label] description
    action_label = current_pred['action_label']
    description = current_pred['description']
    # Truncate description if too long (根据视频宽度调整)
    max_desc_len = int(width / 15)  # 根据视频宽度动态调整
    max_desc_len = max(40, min(max_desc_len, 80))  # 限制在40-80字符之间
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
    
    # Draw action label (magenta in BGR) - 在左上角
    cv2.putText(img_bgr, text_line1, (text_x, y_start), font, font_scale, (255, 0, 255), thickness)
    
    # Draw description (cyan in BGR, 不用白色) - 在action label下方
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
                    # 格式: "0.00 5.00 [walk] a person walks / slowly" 或 "5.00 10.00 [walk forward] a person walks..."
                    # 先找到第一个 [ 和对应的 ]
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # 提取时间
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # 提取action_label（在[]中，可能包含空格）
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # 提取description（]之后的内容）
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
            
            # Project NTU 3D joints to 2D using full_perspective_projection (参考 real_time.py)
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
                
                # 标记有有效的NTU action recognition
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
        
        # output_classifier.mp4 不显示 MotionGPT 文本，只显示 NTU action recognition
        
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
    使用 NTU 60 骨架格式渲染骨架可视化（参考 real_time.py）
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
                    # 格式: "0.00 5.00 [walk] a person walks / slowly" 或 "5.00 10.00 [walk forward] a person walks..."
                    # 先找到第一个 [ 和对应的 ]
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # 提取时间
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # 提取action_label（在[]中，可能包含空格）
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # 提取description（]之后的内容）
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

            # 获取 NTU joints (参考 real_time.py 中的方法)
            if 'ntu_joints' in val and frame_i2 < len(val['ntu_joints']):
                ntu_joints_3d = val['ntu_joints'][frame_i2]  # (25, 3)
            else:
                # 从 vertices 中提取 NTU joints
                vertices = val['verts'][frame_i2]  # (6890, 3)
                vertices_tensor = torch.from_numpy(vertices).float().to(cfg.DEVICE).unsqueeze(0)
                ntu_joints_3d_tensor = smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints_3d = ntu_joints_3d_tensor[0].cpu().numpy()  # (25, 3)

            # 使用 trans_cam 进行投影（参考 real_time.py 中的 full_perspective_projection）
            trans_cam = None
            if 'trans_cam' in val and frame_i2 < len(val['trans_cam']):
                trans_cam = val['trans_cam'][frame_i2]

            ntu_joints_2d = _project_ntu_joints_to_2d(
                ntu_joints_3d, width, height, val, frame_i2,
                trans_cam=trans_cam, device=cfg.DEVICE
            )

            # 渲染 NTU 骨架（始终绘制）
            img_bgr = _render_ntu_skeleton(img_bgr, ntu_joints_2d)

            if False:
                joints3d = val['joints3d'][frame_i2]
                # Calculate and render gait metrics
                step_length, left_knee_angle, right_knee_angle, cadence = analyzer.calculate_gait_metrics(joints3d, frame_i)
                img_bgr = render_metrics(img_bgr, step_length, left_knee_angle, right_knee_angle, cadence)

        # Convert back to RGB for imageio
        img = img_bgr[..., ::-1].copy()
        
        # output_classifier.mp4 不显示 MotionGPT 文本，只显示 NTU skeleton
        writer.append_data(img)
        bar.next()
        frame_i += 1

    writer.close()
    bar.finish()
    logger.info(f"NTU skeleton video saved to: {osp.join(output_pth, 'output_classifier.mp4')}")

def run_gpt_text_vis(cfg, video, results, output_pth, smpl, vis_global=True):
    """
    渲染 MotionGPT 文本预测 + NTU skeleton（独立的可视化分支）
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
                    # 格式: "0.00 5.00 [walk] a person walks / slowly" 或 "5.00 10.00 [walk forward] a person walks..."
                    # 先找到第一个 [ 和对应的 ]
                    start_bracket = line.find('[')
                    end_bracket = line.find(']', start_bracket)
                    if start_bracket > 0 and end_bracket > start_bracket:
                        try:
                            # 提取时间
                            time_part = line[:start_bracket].strip()
                            time_parts = time_part.split()
                            if len(time_parts) >= 2:
                                start_time = float(time_parts[0])
                                end_time = float(time_parts[1])
                                # 提取action_label（在[]中，可能包含空格）
                                action_label = line[start_bracket+1:end_bracket].strip()
                                # 提取description（]之后的内容）
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
        
        # Render NTU skeleton (始终绘制)
        for _id, val in results.items():
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]

            # 获取 NTU joints
            if 'ntu_joints' in val and frame_i2 < len(val['ntu_joints']):
                ntu_joints_3d = val['ntu_joints'][frame_i2]  # (25, 3)
            else:
                # 从 vertices 中提取 NTU joints
                vertices = val['verts'][frame_i2]  # (6890, 3)
                vertices_tensor = torch.from_numpy(vertices).float().to(cfg.DEVICE).unsqueeze(0)
                ntu_joints_3d_tensor = smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints_3d = ntu_joints_3d_tensor[0].cpu().numpy()  # (25, 3)

            # 使用 trans_cam 进行投影
            trans_cam = None
            if 'trans_cam' in val and frame_i2 < len(val['trans_cam']):
                trans_cam = val['trans_cam'][frame_i2]

            ntu_joints_2d = _project_ntu_joints_to_2d(
                ntu_joints_3d, width, height, val, frame_i2,
                trans_cam=trans_cam, device=cfg.DEVICE
            )

            # 渲染 NTU 骨架
            img_bgr = _render_ntu_skeleton(img_bgr, ntu_joints_2d)
        
        # Convert back to RGB
        img = img_bgr[..., ::-1].copy()
        
        # Display MotionGPT predictions (绘制文本)
        img = draw_motiongpt_predictions(img, motiongpt_predictions, frame_i, width, height, chunk_size=100)
        
        writer.append_data(img)
        bar.next()
        frame_i += 1

    writer.close()
    bar.finish()
    logger.info(f"GPT text + NTU skeleton visualization saved to: {osp.join(output_pth, 'output_gpt_text.mp4')}")
