# real_time.py
import os
import gc
import argparse
import os.path as osp
import time
import logging
import cv2
import torch
import joblib
import numpy as np
from loguru import logger
from progress.bar import Bar
import pyrealsense2 as rs

from configs.config import get_cfg_defaults
from lib.data.datasets.dataset_custom import convert_dpvo_to_cam_angvel
from lib.utils.imutils import avg_preds
from lib.utils.transforms import matrix_to_axis_angle
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
from lib.models.smplify import TemporalSMPLify
from lib.data.utils.normalizer import Normalizer
from lib.utils.imutils import compute_cam_intrinsics
from lib.utils.kp_utils import root_centering
from lib.utils import transforms


class _OneEuroFilterGeneric:
    """通用 OneEuroFilter，适用于任意实数信号（无 missing-keypoint mask）。"""

    def __init__(self, x0, min_cutoff=1.0, beta=0.5, d_cutoff=30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0.astype(np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)

    @staticmethod
    def _smoothing_factor(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x):
        t_e = 1.0 / self.d_cutoff  # 固定帧间隔 = 1/fps
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

# Lazy import for visualization functions
try:
    from lib.vis.run_vis import render_skeleton, run_skeleton_vis_sequential
    VIS_AVAILABLE = True
except ImportError as e:
    VIS_AVAILABLE = False
    logger.warning(f"Visualization modules not fully available: {e}. Basic skeleton rendering may still work.")
    # Define a minimal render_skeleton function if import fails
    def render_skeleton(joints, img, line_thickness=2, point_radius=4):
        return img
from tools.inference.optimized_streaming import OptimizedStreamingInference
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
try:
    from lib.models.preproc.slam import SLAMModel
    _run_global = True
except:
    _run_global = False

KEYPOINTS_THR = 0.3

def convert_cxys_to_xywh(bbox):
    """
    将边界框从 [center_x, center_y, scale] 格式
    转换为 [x_min, y_min, width, height] 格式。

    假设一个标准的裁剪框尺寸为 200px，并由 's' 进行缩放。
    这是在很多人体姿态估计流程中的常见约定。
    """
    cx, cy, s = bbox
    box_size = s * 200  # 基于常见约定的边长计算
    
    x_min = cx - box_size / 2
    y_min = cy - box_size / 2
    width = box_size
    height = box_size
    
    return [x_min, y_min, width, height]

def calculate_iou(box1, box2):
    """计算两个边界框的交并比(IoU)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
        
    return inter_area / union_area

def open_device_auto(width: int, height: int, fps: int) -> cv2.VideoCapture:
    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        if cap.isOpened():
            print(f"自动选择设备: /dev/video{idx}")
            return cap
        cap.release()

def process_init(prefix,data,norm_kp2d,smpl, device):
    tt = lambda x: x.unsqueeze(0).to(device)
    init_output = smpl.get_output(
        global_orient=data[prefix + 'init_global_orient'].to(device),
        body_pose=data[prefix + 'init_body_pose'].to(device),
        betas=data[prefix + 'init_betas'].to(device),
        pose2rot=False,
        return_full_pose=True
    )
    init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
    init_kp = tt(torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1).to(device)), dim=-1))
    init_smpl = tt(transforms.matrix_to_rotation_6d(init_output.full_pose))
    init_root = transforms.matrix_to_rotation_6d(init_output.global_orient).to(device)

    return (init_kp, init_smpl), init_root

def process_frame_data(prefix, data, slam_data, window_size, width, height, fps, device, cfg):
    """Process a single frame's tracking data"""

    tt = lambda x: x.unsqueeze(0).to(device)
    kp2d = torch.from_numpy(data[prefix + 'keypoints'][-window_size:]).float()
    mask = kp2d[..., -1] < KEYPOINTS_THR
    bbox = torch.from_numpy(data[prefix + 'bbox'][-window_size:]).float()
    res = torch.tensor([width, height]).float()
    intrinsics = compute_cam_intrinsics(res)
    keypoints_normalizer = Normalizer(cfg)
    norm_kp2d, _ = keypoints_normalizer(
        kp2d[..., :-1].clone(), res, intrinsics, 224, 224, bbox
    )

    if data[prefix + 'features'][-window_size:] ==[None]:
        features = None
    else:
        features =  tt(data[prefix + 'features'][-window_size:])

    cam_angvel = convert_dpvo_to_cam_angvel(slam_data, fps)
    return (
        tt(norm_kp2d),
        features,
        tt(mask),
        tt(cam_angvel),
        data['frame_id'],
        {'cam_intrinsics': tt(intrinsics),
         'bbox': tt(bbox),
         'res': tt(res)},
    )

class OptimizedSequentialVideoProcessor:
    """
    优化的视频处理器 - 替代原SequentialVideoProcessor
    
    主要改进:
    1. 使用OptimizedStreamingInference替代StreamingInference
    2. 自适应窗口大小
    3. 更好的内存管理
    4. 更详细的性能监控
    5. ✅ 保存原始视频和每帧output
    """
    def __init__(self, cfg, video_path, output_path, network, window_size,
                 calib=None, run_global=True, save_pkl=False, visualize=False, 
                 max_frames=1000, enable_adaptive_window=True, 
                 min_window=5, max_window=15,
                 action_config=None, action_checkpoint=None, action_label_map=None, action_engine=None,
                 flip_select='all', flip_interval=5):
        self.cfg = cfg
        self.video_path = video_path
        self.output_path = output_path
        self.network = network
        self.calib = calib
        self.run_global = run_global
        self.save_pkl = save_pkl
        self.visualize = visualize
        self.device = cfg.DEVICE.lower()
        self.window_size = window_size
        self.max_frames = max_frames
        self.flip_eval = cfg.FLIP_EVAL
        self.flip_select = flip_select  # 'all' | 'oblique' | 'interval'
        self.flip_interval = max(1, int(flip_interval))
        
        # 设备初始化
        self.is_realsense_camera = False
        self.is_fisheye_camera = False
        self.is_video_file = False
        
        if video_path == "realsense":
            self._init_realsense()
        elif video_path == "fisheye":
            self._init_fisheye()
        else:
            self._init_video_file(video_path)
        
        # 初始化检测和特征提取器
        self.detector = DetectionModel(cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
        
        # 初始化action recognition
        self.action_recognizer = None
        self.last_ntu_joints = None  # 用于可视化的 NTU 关键点
        # 优先使用 TensorRT 引擎
        if action_engine and os.path.exists(action_engine):
            try:
                from lib.action_recognition_trt import ActionRecognizerTRT
                base_recognizer = ActionRecognizerTRT(
                    engine_path=action_engine,
                    label_map_path=action_label_map,
                    window_size=100,
                    num_keypoints=25
                )
                
                # 使用稳定化包装器
                from lib.action_recognition_stable import StableActionRecognizer
                self.action_recognizer = StableActionRecognizer(
                    base_recognizer,
                    smoothing_window=5,
                    confidence_threshold=0.15,
                    min_switch_frames=8
                )
                
                logger.info(f"TensorRT action recognition initialized with stability wrapper: {action_engine}")
                logger.info(f"  Using NTU-25 3D skeleton format ({base_recognizer.num_keypoints} keypoints)")
                logger.info(f"  Window size: {base_recognizer.window_size} frames")
                logger.info(f"  Real-time buffer update: Every frame will be added to buffer")
                logger.info(f"  Prediction mode: Real-time (every frame after {15} frames buffered)")
            except Exception as e:
                logger.warning(f"Failed to initialize TensorRT action recognition: {e}")
                self.action_recognizer = None
        
        # 如果 TensorRT 不可用，回退到 PyTorch
        if self.action_recognizer is None and action_config and action_checkpoint:
            try:
                from lib.action_recognition import ActionRecognizer
                base_recognizer = ActionRecognizer(
                    config_path=action_config,
                    checkpoint_path=action_checkpoint,
                    label_map_path=action_label_map,
                    device=cfg.DEVICE.lower(),
                    window_size=48,  # Will auto-adjust based on model type
                    num_keypoints=17  # Default COCO format, will auto-detect from config
                )
                
                # 使用稳定化包装器
                from lib.action_recognition_stable import StableActionRecognizer
                self.action_recognizer = StableActionRecognizer(
                    base_recognizer,
                    smoothing_window=5,
                    confidence_threshold=0.15,
                    min_switch_frames=8
                )
                
                logger.info("PyTorch action recognition initialized with stability wrapper")
                logger.info(f"  Using {base_recognizer.num_keypoints} keypoints (auto-detected from model)")
                logger.info(f"  Window size: {base_recognizer.window_size} frames")
                logger.info(f"  Real-time buffer update: Every frame will be added to buffer")
                logger.info(f"  Prediction mode: Real-time (every frame after {15} frames buffered)")
                if base_recognizer.num_keypoints == 25:
                    logger.info("  ✓ NTU-25 3D skeleton format confirmed")
                else:
                    logger.warning(f"  ⚠ Expected 25 NTU keypoints, but model uses {base_recognizer.num_keypoints}")
            except Exception as e:
                logger.warning(f"Failed to initialize action recognition: {e}")
                self.action_recognizer = None
        
        # 使用优化的流式推理器
        self.stream_inference = OptimizedStreamingInference(
            network, 
            cfg.DEVICE.lower(),
            max_history_frames=window_size,
            enable_adaptive_window=enable_adaptive_window,
            min_window=min_window,
            max_window=max_window
        )
        logger.info(f"Optimized streaming inference initialized:")
        logger.info(f"  - Adaptive window: {enable_adaptive_window}")
        logger.info(f"  - Window range: {min_window}-{max_window} frames")
        logger.info(f"  - Flip evaluation: {self.flip_eval} (real_time flip_eval=FLIP_EVAL, same as demo)")
        if self.flip_eval:
            logger.info(f"  - Flip select: {self.flip_select} (all=每帧, oblique=仅斜侧角, interval=每N帧)")

        # 翻转推理器（flip evaluation时使用）
        if self.flip_eval:
            self.flipped_stream_inference = OptimizedStreamingInference(
                network,
                cfg.DEVICE.lower(),
                max_history_frames=window_size,
                enable_adaptive_window=enable_adaptive_window,
                min_window=min_window,
                max_window=max_window
            )
            logger.info("  - Flipped streaming inference initialized")

        self.last_subject_data = {}
        self.iou_threshold = 0.9
        logger.info(f"  - Feature Propagation enabled with IoU threshold: {self.iou_threshold}")
        
        # ✅ 初始化结果存储 - 用于保存每帧output
        self.results = {}
        self.frame_outputs = []  # 存储每一帧的output
        
        # ✅ 初始化视频写入器（包括原始视频和skeleton视频）
        self._init_video_writer()
        
        # 性能统计
        self.timing_stats = {
            'total': [],
            'tracking': [],
            'feature_extraction': [],
            'data_processing': [],
            'inference': [],
            'visualization': []
        }
        self.flip_mode_frames = 0   # 实际做了 flip 的帧数
        self.normal_mode_frames = 0  # 仅单遍推理的帧数

        # 输出平滑：OneEuroFilter 对 pose/shape/cam/root 做自适应时序平滑
        self._oef_filters = {}  # 延迟初始化，首帧时按实际 shape 创建

        # bbox 平滑：消除检测器帧间 bbox 跳动（bbox 经投影放大后是 2D 抖动主因）
        self._prev_smooth_bbox = None
        self._bbox_smooth_alpha = 0.5  # bbox EMA 权重（当前帧）

        # joints2d 平滑
        self._oef_joints2d = None

    def _init_realsense(self):
        """初始化RealSense相机"""
        import pyrealsense2 as rs
        self.is_realsense_camera = True
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        self.width = 1280
        self.height = 720
        self.fps = 30
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        
        try:
            logger.info("Starting RealSense camera...")
            self.pipeline.start(self.config)
            logger.info("RealSense camera started successfully")
            self.length = self.max_frames
            self.cap = None
        except Exception as e:
            logger.error(f"Failed to start RealSense camera: {e}")
            raise

    def _init_fisheye(self):
        """初始化Fisheye相机"""
        import cv2
        self.is_fisheye_camera = True
        self.width = 1280
        self.height = 720
        self.fps = 60
        
        gst_pipeline = (
            f"v4l2src device=/dev/video0 ! "
            f"image/jpeg, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            "jpegdec ! videoconvert ! appsink"
        )
        
        logger.info(f"Starting Fisheye camera: {gst_pipeline}")
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            logger.error("Failed to open Fisheye camera")
            raise RuntimeError("Cannot open /dev/video0")
        
        logger.info(f"Fisheye camera started ({self.width}x{self.height} @ {self.fps}fps)")
        self.length = self.max_frames
        self.pipeline = None

    def _init_video_file(self, video_path):
        """初始化视频文件"""
        import cv2
        self.is_video_file = True
        self.cap = cv2.VideoCapture(video_path)
        
        assert self.cap.isOpened(), f'Failed to load video file {video_path}'
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps == 0:
            logger.warning("Video FPS is 0, setting to default 30")
            self.fps = 30
        
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pipeline = None

    def _init_video_writer(self):
        """✅ 初始化视频写入器 - 与 demo 一致使用 imageio+FFMPEG，避免 mp4v 导致输出无法播放"""
        import cv2
        is_camera = self.is_realsense_camera or self.is_fisheye_camera
        
        # 与 demo.py / run_vis 一致：用 imageio FFMPEG 写 MP4，兼容性更好；否则回退到 cv2
        self._use_imageio_writer = IMAGEIO_AVAILABLE
        if self._use_imageio_writer:
            fps_float = float(self.fps)
            self.output_raw_video_path = os.path.join(self.output_path, "output_raw.mp4")
            self.output_raw_video = imageio.get_writer(
                self.output_raw_video_path, fps=fps_float, mode='I', format='FFMPEG', macro_block_size=1
            )
            logger.info(f"Raw video writer initialized (imageio/FFMPEG): {self.output_raw_video_path}")
        else:
            self.output_raw_video_path = os.path.join(self.output_path, "output_raw.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_raw_video = cv2.VideoWriter(
                self.output_raw_video_path, fourcc, self.fps, (self.width, self.height)
            )
            logger.warning("imageio not available, using cv2.VideoWriter (mp4v); output may not play on some players")
            logger.info(f"Raw video writer initialized: {self.output_raw_video_path}")
        
        if self.visualize:
            self.output_skeleton_video_path = os.path.join(self.output_path, "output_skeleton.mp4")
            if self._use_imageio_writer:
                self.output_skeleton_video = imageio.get_writer(
                    self.output_skeleton_video_path, fps=float(self.fps), mode='I', format='FFMPEG', macro_block_size=1
                )
                logger.info(f"Skeleton video writer initialized (imageio/FFMPEG): {self.output_skeleton_video_path}")
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.output_skeleton_video = cv2.VideoWriter(
                    self.output_skeleton_video_path, fourcc, self.fps, (self.width, self.height)
                )
                logger.info(f"Skeleton video writer initialized: {self.output_skeleton_video_path}")
            
            if is_camera:
                cv2.namedWindow('WHAM Real-time Tracking', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('WHAM Real-time Tracking', self.width, self.height)
                logger.info("Display window created")
        else:
            self.output_skeleton_video = None

    def _write_raw_frame(self, frame):
        """写一帧到原始视频（内部根据 imageio/cv2 做 BGR→RGB 转换）"""
        if self._use_imageio_writer:
            self.output_raw_video.append_data(frame[..., ::-1].copy())  # BGR -> RGB
        else:
            self.output_raw_video.write(frame)

    def _write_skeleton_frame(self, img):
        """写一帧到 skeleton 视频（img 为 BGR）"""
        if self.output_skeleton_video is None:
            return
        if self._use_imageio_writer:
            self.output_skeleton_video.append_data(img[..., ::-1].copy())  # BGR -> RGB
        else:
            self.output_skeleton_video.write(img)

    def _read_frame(self, frame_idx):
        """读取帧"""
        import numpy as np
        
        if self.is_realsense_camera:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return False, None
                frame = np.asanyarray(color_frame.get_data())
                return True, frame
            except Exception as e:
                logger.error(f"Error reading from RealSense: {e}")
                return False, None
        elif self.is_fisheye_camera or self.is_video_file:
            if self.is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            return self.cap.read()
        return False, None

    def _render_ntu_skeleton(self, img, ntu_joints_2d, color=(0, 255, 255)):
        """
        渲染 NTU RGB+D 25 关键点骨架
        
        NTU 25 骨架连接:
        躯干: 0-1-20-2-3 (spine)
        左臂: 20-4-5-6-7, 7-21, 7-22 (hand tips)
        右臂: 20-8-9-10-11, 11-23, 11-24 (hand tips)
        左腿: 0-12-13-14-15
        右腿: 0-16-17-18-19
        """
        import cv2
        
        # NTU 骨架连接
        ntu_skeleton = [
            # 躯干
            (0, 1), (1, 20), (20, 2), (2, 3),
            # 左臂
            (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),
            # 右臂
            (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),
            # 左腿
            (0, 12), (12, 13), (13, 14), (14, 15),
            # 右腿
            (0, 16), (16, 17), (17, 18), (18, 19),
        ]
        
        h, w = img.shape[:2]
        
        # 绘制骨架连线
        for start_idx, end_idx in ntu_skeleton:
            if start_idx < len(ntu_joints_2d) and end_idx < len(ntu_joints_2d):
                pt1 = ntu_joints_2d[start_idx]
                pt2 = ntu_joints_2d[end_idx]
                
                # 检查点是否在有效范围内
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(img, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), color, 2)
        
        # 绘制关键点
        for i, pt in enumerate(ntu_joints_2d):
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
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
                
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, pt_color, -1)
                cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 1)
        
        return img

    def _visualize_frame(self, frame, frame_idx, joints2d, action_label=None, action_confidence=None, ntu_joints_2d=None):
        """可视化当前帧"""
        if not self.visualize:
            return frame
        
        import cv2
        try:
            from lib.vis.run_vis import render_skeleton
        except ImportError:
            # Fallback: simple skeleton rendering without pytorch3d
            def render_skeleton(joints, img, line_thickness=2, point_radius=4):
                return img
        
        img = frame.copy()
        
        # 使用网络输出的 joints2d 绘制骨架（与 demo 一致，WHAM 前17关节为 COCO 格式）
        joints2d_tensor = torch.from_numpy(joints2d).float().to(self.cfg.DEVICE)
        j2d = joints2d_tensor.reshape(-1, 2)
        n_j = min(17, j2d.shape[0])  # COCO 17 joints
        img = render_skeleton(j2d[:n_j], img)
        
        # 显示action label - 更明显的显示效果
        if self.action_recognizer is not None:
            buffer_size = self.action_recognizer.get_buffer_size()
            
            if action_label and action_confidence is not None:
                # 显示action label和置信度
                label_text = f"Action: {action_label}"
                confidence_text = f"Confidence: {action_confidence:.2f}"
                
                # 绘制背景矩形以提高可读性
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (5, 5), (text_width + 15, text_height * 2 + baseline + 20), 
                            (0, 0, 0), -1)  # 黑色半透明背景
                cv2.rectangle(img, (5, 5), (text_width + 15, text_height * 2 + baseline + 20), 
                            (0, 255, 0), 2)  # 绿色边框
                
                # 显示action label（绿色，较大字体）
                cv2.putText(img, label_text, (10, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # 显示置信度（黄色）
                cv2.putText(img, confidence_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # 显示缓冲状态
                status_text = f"Action Recognition: Buffering ({buffer_size}/{self.action_recognizer.window_size} frames)"
                cv2.putText(img, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return img

    def _print_timing_stats(self, frame_idx):
        """打印计时统计"""
        if frame_idx % 30 == 0 and frame_idx > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Frame {frame_idx} Performance Metrics")
            logger.info(f"{'='*60}")
            
            for key in ['tracking', 'feature_extraction', 'data_processing', 'inference', 'visualization']:
                if self.timing_stats[key]:
                    avg_time = sum(self.timing_stats[key][-30:]) / min(30, len(self.timing_stats[key]))
                    logger.info(f"{key:20s}: {avg_time*1000:6.1f}ms")
            
            self.stream_inference.print_stats()

    def run(self):
        """运行视频处理"""
        import cv2
        from progress.bar import Bar
        import numpy as np
        
        start_total = time.time()
        frame_idx = 0
        skip = max(1, 30//self.fps)
        
        slam_results = np.zeros((self.length, 7))
        slam_results[:, 3] = 1.0
        
        bar = Bar('Processing frames', fill='#', max=self.length//skip)
        
        is_camera = self.is_realsense_camera or self.is_fisheye_camera

        # Debug: flip_eval status (same as demo.py FLIP_EVAL handling)
        if self.flip_eval:
            logger.info(f"real_time FLIP_EVAL enabled: two forward passes per frame, then average (same as demo --visualize flip)")

        # 主处理循环
        while frame_idx < self.length:
            frame_start_time = time.time()
            
            # 读取帧
            ret, frame = self._read_frame(frame_idx)
            if not ret or frame is None:
                if is_camera:
                    logger.warning(f"Failed to read frame {frame_idx}, retrying...")
                    continue
                else:
                    break
            
            # ✅ 保存原始帧（无skeleton）
            self._write_raw_frame(frame)
            
            # 跟踪检测
            track_start = time.time()
            self.detector.track(frame, self.fps, self.length, use_full_frame_fallback=True)
            
            if len(self.detector.tracking_results['id']) == 0:
                logger.warning(f"No detections at frame {frame_idx}")
                img = self._handle_no_detection(frame, is_camera)
                frame_idx += skip
                bar.next()
                continue
            
            tracking_results = self.detector.process(self.fps)
            track_end = time.time()
            self.timing_stats['tracking'].append(track_end - track_start)
            
            # 特征提取
            subject_id = self.detector.tracking_results['id'][-1]
            if frame_idx//skip not in tracking_results[subject_id]['frame_id']:
                logger.warning(f"No tracking results for frame {frame_idx}")
                img = self._handle_tracking_lost(frame, is_camera)
                frame_idx += skip
                bar.next()
                continue
            
            feat_start = time.time()
            
            bbox_idx = np.where(np.array(tracking_results[subject_id]['frame_id']) == frame_idx//skip)[0][0]
            current_bbox_cxys = tracking_results[subject_id]['bbox'][bbox_idx]

            use_cached_feature = False
            current_bbox_xywh = None
            
            if len(current_bbox_cxys) == 3:
                current_bbox_xywh = convert_cxys_to_xywh(current_bbox_cxys)
                
                if subject_id in self.last_subject_data:
                    cached = self.last_subject_data[subject_id]
                    last_bbox_xywh, last_feature = cached[0], cached[1]
                    iou = calculate_iou(current_bbox_xywh, last_bbox_xywh)
                    if iou > self.iou_threshold and last_feature is not None:
                        use_cached_feature = True
            else:
                logger.warning(f"Frame {frame_idx}: Received malformed bbox with {len(current_bbox_cxys)} elements. Expected 3. Skipping IoU check.")

            if use_cached_feature and current_bbox_xywh is not None:
                self.stream_inference.stats['cache_hits'] += 1
                cached = self.last_subject_data[subject_id]
                last_feature_tensor = cached[1]
                last_flipped_feature = cached[2] if self.flip_eval and len(cached) > 2 else last_feature_tensor
                # Detector output has no 'features'; init list if missing (same pattern as extractor)
                if 'features' not in tracking_results[subject_id]:
                    tracking_results[subject_id]['features'] = []
                tracking_results[subject_id]['features'].append(last_feature_tensor)
                if self.flip_eval:
                    from lib.utils.imutils import flip_kp, flip_bbox
                    if 'flipped_features' not in tracking_results[subject_id]:
                        tracking_results[subject_id]['flipped_features'] = []
                    tracking_results[subject_id]['flipped_features'].append(last_flipped_feature)
                    bbox = tracking_results[subject_id]['bbox'][bbox_idx]
                    keypoints = tracking_results[subject_id]['keypoints'][bbox_idx]
                    tracking_results[subject_id]['flipped_bbox'] = np.array([flip_bbox(bbox, self.width, self.height)])
                    tracking_results[subject_id]['flipped_keypoints'] = np.array([flip_kp(keypoints, self.width)])
                self.last_subject_data[subject_id] = (current_bbox_xywh, last_feature_tensor, last_flipped_feature) if self.flip_eval else (current_bbox_xywh, last_feature_tensor)
            else:
                self.stream_inference.stats['cache_misses'] += 1
                tracking_results = self.extractor.run_one_frame(frame, frame_idx//skip, tracking_results)
                new_feature = tracking_results[subject_id]['features'][-1]
                new_flipped = tracking_results[subject_id]['flipped_features'][-1] if self.flip_eval and 'flipped_features' in tracking_results[subject_id] else new_feature
                if current_bbox_xywh is not None:
                    if self.flip_eval:
                        self.last_subject_data[subject_id] = (current_bbox_xywh, new_feature, new_flipped)
                    else:
                        self.last_subject_data[subject_id] = (current_bbox_xywh, new_feature)
            
            feat_end = time.time()
            self.timing_stats['feature_extraction'].append(feat_end - feat_start)
            
            # 数据处理
            data_start = time.time()
            current_slam_data = slam_results[:frame_idx] if frame_idx < len(slam_results) else np.array([0, 0, 0, 1, 0, 0, 0])
            subject_data = tracking_results[subject_id]
            
            with torch.no_grad():
                batch = self._process_frame_data(subject_data, current_slam_data)
                x, features, mask, cam_angvel, frame_id, kwargs = batch
                
                # 保存 kwargs 用于后续投影
                self.current_kwargs = kwargs
                
                if 'init_global_orient' not in subject_data:
                    inits = (None, None)
                    init_root = None
                else:
                    inits, init_root = self._process_init(subject_data, x)
            
            data_end = time.time()
            self.timing_stats['data_processing'].append(data_end - data_start)
            
            # 推理
            infer_start = time.time()
            with torch.no_grad():
                output = self.stream_inference.process_frame(
                    x, inits,
                    window_size=self.window_size,
                    img_features=features,
                    mask=mask,
                    init_root=init_root,
                    cam_angvel=cam_angvel,
                    cam_intrinsics=kwargs['cam_intrinsics'],
                    bbox=kwargs['bbox'],
                    res=kwargs['res'],
                    return_y_up=True,
                    subject_id=subject_id
                )

                # Flip evaluation: 仅对选中帧做 flip（all=每帧, oblique=斜侧角, interval=每N帧）
                if self.flip_eval:
                    do_flip = self._should_do_flip_for_frame(frame_idx, subject_data)
                    flip_keys = ('flipped_keypoints', 'flipped_bbox', 'flipped_features')
                    has_flipped = all(k in subject_data for k in flip_keys)
                    if frame_idx == 0 or frame_idx % 100 == 0:
                        logger.debug(f"Frame {frame_idx} flip_eval: do_flip={do_flip}, has_flipped={has_flipped}")
                    if not do_flip:
                        # 本帧不做 flip，直接用 normal output
                        self.normal_mode_frames += 1
                    elif not has_flipped:
                        self.normal_mode_frames += 1
                        logger.warning(
                            f"Frame {frame_idx}: FLIP_EVAL enabled but subject_data missing flipped keys (e.g. cache path or first frame). Using normal output only."
                        )
                    else:
                        self.flip_mode_frames += 1
                        flipped_batch = self._process_flipped_frame_data(subject_data, current_slam_data)
                        fx, f_features, f_mask, f_cam_angvel, f_frame_id, f_kwargs = flipped_batch

                        # flip 无上一帧信息时：不复用 normal，直接从 extractor.predict_init 取 flipped init
                        has_flip_prev = self.flipped_stream_inference.prev_output is not None
                        if not has_flip_prev:
                            # 每次 flip stream 没有上一帧时都重新 predict_init，不复用旧的或 normal 的信息
                            tracking_results = self._predict_flipped_init(frame, frame_idx // skip, subject_id, tracking_results)
                            subject_data = tracking_results[subject_id]  # refresh after predict_init

                        if 'flipped_init_global_orient' in subject_data:
                            f_inits, f_init_root = self._process_flipped_init(subject_data, fx)
                        else:
                            f_inits, f_init_root = self._make_fallback_init(fx)

                        # 保存 normal pred_cam（stream_flip 用 cam[0:1] 投影平均 pose）
                        normal_pred_cam = self.network.pred_cam.clone()

                        # flip 有上一帧时直接用 flipped stream 自己的 hidden state 等，无需从 normal 复制
                        flipped_output = self.flipped_stream_inference.process_frame(
                            fx, f_inits,
                            window_size=self.window_size,
                            img_features=f_features,
                            mask=f_mask,
                            init_root=f_init_root,
                            cam_angvel=f_cam_angvel,
                            cam_intrinsics=f_kwargs['cam_intrinsics'],
                            bbox=f_kwargs['bbox'],
                            res=f_kwargs['res'],
                            return_y_up=True,
                            subject_id=subject_id
                        )

                        # Average normal and flipped predictions (same as demo.py)
                        pose = output['pose'].squeeze(0)
                        shape = output['betas'].squeeze(0)
                        flipped_pose = flipped_output['pose'].squeeze(0)
                        flipped_shape = flipped_output['betas'].squeeze(0)

                        pose = pose.reshape(-1, 24, 6)
                        flipped_pose = flipped_pose.reshape(-1, 24, 6)

                        avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
                        avg_pose = avg_pose.reshape(-1, 144)
                        avg_contact = (flipped_output['contact'][..., [2, 3, 0, 1]] + output['contact']) / 2

                        # 用 normal 的 pred_cam 投影（与 stream_flip _handle_flip_eval 一致）
                        self.network.pred_cam = normal_pred_cam
                        self.network.pred_pose = avg_pose.view_as(self.network.pred_pose)
                        self.network.pred_shape = avg_shape.view_as(self.network.pred_shape)
                        self.network.pred_contact = avg_contact.view_as(self.network.pred_contact)
                        # 单帧预测时只传最后一帧的 bbox/cam_intrinsics，否则投影位置错误
                        smpl_kwargs = dict(kwargs)
                        if kwargs.get('bbox') is not None and kwargs['bbox'].shape[1] > 1:
                            smpl_kwargs['bbox'] = kwargs['bbox'][:, -1:, :]
                        if kwargs.get('cam_intrinsics') is not None and kwargs['cam_intrinsics'].dim() == 4 and kwargs['cam_intrinsics'].shape[1] > 1:
                            smpl_kwargs['cam_intrinsics'] = kwargs['cam_intrinsics'][:, -1:, :, :]
                        output = self.network.forward_smpl(**smpl_kwargs)
                        # 融合 view-independent hidden states (motion_encoder, motion_decoder)
                        # 让 normal stream 后续帧受益于 flip 的运动信息，同时不干扰 view-dependent 状态
                        self.stream_inference.fuse_view_independent_states(
                            self.flipped_stream_inference, alpha=0.7
                        )
                else:
                    # 未开启 flip_eval，全部为 normal 模式
                    self.normal_mode_frames += 1
                # ---- 时序平滑：OneEuroFilter on pose, shape, cam, root ----
                # 自适应滤波：快动作少平滑（减少滞后），慢动作多平滑（消除抖动）
                smooth_targets = {
                    'pose':  self.network.pred_pose,
                    'shape': self.network.pred_shape,
                    'cam':   self.network.pred_cam,
                }
                if self.network.pred_root is not None:
                    smooth_targets['root'] = self.network.pred_root

                for key, tensor in smooth_targets.items():
                    arr = tensor.detach().cpu().numpy().flatten().astype(np.float64)
                    if key not in self._oef_filters:
                        # 首帧：初始化滤波器，不滤波
                        self._oef_filters[key] = _OneEuroFilterGeneric(
                            x0=arr,
                            min_cutoff=1.0,
                            beta=0.5,
                            d_cutoff=float(self.fps),
                        )
                    else:
                        arr = self._oef_filters[key](arr)
                    smoothed = torch.from_numpy(arr.reshape(tensor.shape)).float().to(tensor.device)
                    setattr(self.network, 'pred_' + key, smoothed)

                # 用平滑后的参数重新生成 SMPL output
                smpl_kwargs_smooth = dict(kwargs)
                if kwargs.get('bbox') is not None and kwargs['bbox'].shape[1] > 1:
                    smpl_kwargs_smooth['bbox'] = kwargs['bbox'][:, -1:, :]
                if kwargs.get('cam_intrinsics') is not None and kwargs['cam_intrinsics'].dim() == 4 and kwargs['cam_intrinsics'].shape[1] > 1:
                    smpl_kwargs_smooth['cam_intrinsics'] = kwargs['cam_intrinsics'][:, -1:, :, :]
                output = self.network.forward_smpl(**smpl_kwargs_smooth)

            infer_end = time.time()
            self.timing_stats['inference'].append(infer_end - infer_start)

            # Extract NTU 25 keypoints from SMPL vertices (always, for visualization)
            # This should be done regardless of whether action_recognizer is enabled
            try:
                # Get NTU 25 keypoints from SMPL vertices
                # Use the last frame's vertices from the network output
                vertices = self.network.output.vertices
                
                # Handle different vertex shapes
                if len(vertices.shape) == 3:
                    # Shape: (batch, 6890, 3) - take last frame
                    vertices = vertices[-1, :, :]  # (6890, 3)
                elif len(vertices.shape) == 4:
                    # Shape: (batch, 1, 6890, 3) - take last frame
                    vertices = vertices[:, [-1], :, :]  # (batch, 1, 6890, 3)
                    vertices = vertices[0, 0, :, :]  # (6890, 3)
                elif len(vertices.shape) == 2:
                    # Shape: (6890, 3) - already correct
                    pass
                else:
                    # Fallback: try to get last frame
                    vertices = vertices[-1] if vertices.shape[0] > 1 else vertices[0]
                    if len(vertices.shape) > 2:
                        vertices = vertices.reshape(-1, 3)
                
                # Ensure vertices is 2D (6890, 3)
                if len(vertices.shape) == 1:
                    vertices = vertices.reshape(-1, 3)
                
                # Extract NTU 25 keypoints from SMPL vertices
                vertices_tensor = vertices if isinstance(vertices, torch.Tensor) else torch.from_numpy(vertices)
                # Ensure vertices are on the correct device
                if not vertices_tensor.is_cuda and self.device.startswith('cuda'):
                    vertices_tensor = vertices_tensor.to(self.device)
                elif vertices_tensor.is_cuda and self.device == 'cpu':
                    vertices_tensor = vertices_tensor.cpu()
                
                if len(vertices_tensor.shape) == 2:
                    vertices_tensor = vertices_tensor.unsqueeze(0)  # (1, 6890, 3)
                
                # Extract NTU 25 keypoints from SMPL vertices using J_regressor_ntu
                ntu_joints = self.network.smpl.get_ntu_joints(vertices_tensor)  # (1, 25, 3)
                ntu_joints = ntu_joints[0].cpu().numpy()  # (25, 3)
                
                # 验证 NTU joints 形状
                assert ntu_joints.shape == (25, 3), f"Expected NTU joints shape (25, 3), got {ntu_joints.shape}"
                
                # 检查 NTU joints 是否有效（非零、非NaN）
                if not (np.isnan(ntu_joints).any() or np.isinf(ntu_joints).any()):
                    # 保存 NTU joints 用于可视化（无论是否有 action_recognizer）
                    self.last_ntu_joints = ntu_joints.copy()
                else:
                    logger.debug(f"Frame {frame_idx}: Invalid NTU joints (NaN/Inf detected), keeping previous joints for visualization")
            except Exception as e:
                logger.debug(f"Error extracting NTU joints: {e}")
                # 保持之前的 last_ntu_joints，不设置为 None
            
            # Action recognition
            action_label = None
            action_confidence = None
            if self.action_recognizer is not None and hasattr(self, 'last_ntu_joints') and self.last_ntu_joints is not None:
                try:
                    ntu_joints = self.last_ntu_joints
                    
                    # 检查 NTU joints 是否有效（非零、非NaN）
                    if np.isnan(ntu_joints).any() or np.isinf(ntu_joints).any():
                        logger.warning(f"Frame {frame_idx}: Invalid NTU joints (NaN/Inf detected), skipping HAR prediction")
                        action_label = None
                        action_confidence = None
                    else:
                        # 每50帧打印一次 NTU skeleton 和 buffer 状态
                        if frame_idx % 50 == 0:
                            buffer_size = self.action_recognizer.get_buffer_size()
                            joints_range = (ntu_joints.min(), ntu_joints.max())
                            pelvis = ntu_joints[0]
                            head = ntu_joints[3]
                            logger.info(f"Frame {frame_idx}: NTU-25 3D skeleton for HAR")
                            logger.info(f"  Buffer: {buffer_size}/{self.action_recognizer.window_size} frames")
                            logger.info(f"  Joints range: [{joints_range[0]:.3f}, {joints_range[1]:.3f}]")
                            logger.info(f"  Pelvis: {pelvis}, Head: {head}")
                        
                        # 实时更新 buffer：每帧都将当前的 NTU 3D skeleton 添加到 buffer
                        # predict_action 内部会调用 add_skeleton_frame，确保 buffer 实时更新
                        buffer_size_before = self.action_recognizer.get_buffer_size()
                        class_idx, confidence, label = self.action_recognizer.predict_action(ntu_joints)
                        buffer_size_after = self.action_recognizer.get_buffer_size()
                        
                        # 验证 buffer 已更新
                        if buffer_size_after != buffer_size_before + 1 and buffer_size_after < self.action_recognizer.window_size:
                            logger.warning(f"Frame {frame_idx}: Buffer may not have updated correctly: {buffer_size_before} -> {buffer_size_after}")
                        
                        # 统计信息：每100帧记录一次 buffer 更新情况
                        if frame_idx % 100 == 0:
                            logger.info(f"Frame {frame_idx}: Buffer real-time update confirmed - size={buffer_size_after}/{self.action_recognizer.window_size}")
                    
                        # 无论 class_idx 是否 >= 0，只要有有效的 label 就显示
                        # class_idx == -1 表示 rate limiting 或 buffering，但 label 仍然是上一次的预测结果
                        if "buffering" in label:
                            # 正在缓冲中，不显示 label
                            action_label = None
                            action_confidence = None
                        elif label and label not in ["waiting...", "rate_limiting", "insufficient_frames"]:
                            # 有有效的 action label，保持显示
                            action_label = label
                            action_confidence = confidence
                            
                            # 首次成功预测时记录日志
                            if not hasattr(self, '_har_first_prediction_logged'):
                                logger.info(f"Frame {frame_idx}: First HAR prediction using NTU-25 3D skeleton")
                                logger.info(f"  Action: {action_label}, Confidence: {action_confidence:.4f}")
                                logger.info(f"  Using {self.action_recognizer.num_keypoints} keypoints, window_size={self.action_recognizer.window_size}")
                                self._har_first_prediction_logged = True
                except Exception as e:
                    logger.warning(f"Action recognition error: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # ✅ 保存当前帧的output
            frame_output = {
                'frame_idx': frame_idx,
                'subject_id': subject_id,
                'output': {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                          for k, v in output.items()},
                'bbox': current_bbox_cxys,
                'timestamp': time.time(),
                'action_label': action_label,
                'action_confidence': action_confidence
            }
            
            # 保存 NTU 3D skeleton 用于验证
            if hasattr(self, 'last_ntu_joints') and self.last_ntu_joints is not None:
                frame_output['ntu_joints_3d'] = self.last_ntu_joints  # (25, 3)
                frame_output['ntu_joints_shape'] = self.last_ntu_joints.shape
            
            self.frame_outputs.append(frame_output)
            
            # 可视化
            viz_start = time.time()
            joints2d_raw = output['joints2d'][:, [-1]].cpu().numpy()
            # joints2d 平滑：最终渲染用，消除残余 2D 投影抖动
            j2d_flat = joints2d_raw.flatten().astype(np.float64)
            if self._oef_joints2d is None:
                self._oef_joints2d = _OneEuroFilterGeneric(
                    x0=j2d_flat, min_cutoff=1.5, beta=0.3, d_cutoff=float(self.fps))
                joints2d = joints2d_raw
            else:
                joints2d = self._oef_joints2d(j2d_flat).reshape(joints2d_raw.shape).astype(np.float32)
            
            # 获取 NTU 3D 关键点并投影到 2D
            # 方法：直接复用网络已经计算好的 full_joints2d 的投影参数
            ntu_joints_2d = None
            if hasattr(self, 'last_ntu_joints') and self.last_ntu_joints is not None:
                try:
                    # 网络的 joints2d 来源: output['joints2d'] = self.output.full_joints2d
                    # full_joints2d shape: (batch, T, J, 2)
                    # 我们在 line 903 用 joints2d = output['joints2d'][:, [-1]] 取最后帧
                    #
                    # joints2d 的计算过程 (smpl.py line 97):
                    #   joints3d shape: (batch, T, J, 3)
                    #   full_cam shape: (batch, T, 3)  -- 在 reshape 前
                    #   cam_intrinsics shape: (batch, T, 3, 3)
                    #   full_joints2d = full_perspective_projection(joints3d, cam_intrinsics, translation=full_cam)
                    #
                    # 然后 output.full_cam = full_cam.reshape(-1, 3) 存储
                    #
                    # 对于 NTU joints，最可靠的方式是直接用 vertices 在相机坐标系下的位置
                    # 因为 vertices 和 joints3d 在同一坐标系下，用同样的投影即可

                    from lib.models.smpl import full_perspective_projection

                    # 获取 full_joints2d 的形状来确定 T
                    full_joints2d = output['joints2d']  # (batch, T, J, 2)
                    T = full_joints2d.shape[1]

                    # full_cam: (batch*T, 3) → reshape 回 (batch, T, 3) → 取最后一帧
                    full_cam = self.network.output.full_cam  # (batch*T, 3)
                    full_cam = full_cam.reshape(-1, T, 3)  # (batch, T, 3)
                    last_cam = full_cam[:, -1:, :]  # (batch, 1, 3)

                    # cam_intrinsics: (batch, T, 3, 3) → 取最后一帧
                    cam_intrinsics = kwargs['cam_intrinsics']  # (batch, T, 3, 3)
                    last_intrinsics = cam_intrinsics[:, -1:, :, :]  # (batch, 1, 3, 3)

                    # NTU joints 3D → (batch, 1, 25, 3) 与 last_cam (batch, 1, 3) 对齐
                    ntu_joints_3d = torch.from_numpy(self.last_ntu_joints).float().to(self.device)
                    ntu_joints_3d = ntu_joints_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, 25, 3)

                    ntu_joints_2d = full_perspective_projection(
                        ntu_joints_3d,
                        last_intrinsics,
                        translation=last_cam
                    )[0, 0].cpu().numpy()  # (25, 2)

                    if frame_idx % 30 == 0:
                        j2d_last = full_joints2d[0, -1].cpu().numpy()  # (J, 2) 最后一帧所有关节
                        logger.info(f"Frame {frame_idx}: NTU projection debug")
                        logger.info(f"  vertices shape: {self.network.output.vertices.shape}")
                        logger.info(f"  full_cam shape: {self.network.output.full_cam.shape}, T={T}")
                        logger.info(f"  last_cam: {last_cam[0,0].cpu().numpy()}")
                        logger.info(f"  last_intrinsics:\n{last_intrinsics[0,0].cpu().numpy()}")
                        logger.info(f"  ntu_joints_3d[0] (pelvis 3D): {self.last_ntu_joints[0]}")
                        logger.info(f"  ntu_joints_2d[0] (pelvis 2D): {ntu_joints_2d[0]}")
                        logger.info(f"  ntu_joints_2d range: x=[{ntu_joints_2d[:,0].min():.0f},{ntu_joints_2d[:,0].max():.0f}] y=[{ntu_joints_2d[:,1].min():.0f},{ntu_joints_2d[:,1].max():.0f}]")
                        logger.info(f"  joints2d (nose) 2D: {j2d_last[0]}")
                        logger.info(f"  joints2d range: x=[{j2d_last[:,0].min():.0f},{j2d_last[:,0].max():.0f}] y=[{j2d_last[:,1].min():.0f},{j2d_last[:,1].max():.0f}]")
                        # 验证：用相同方法投影 network joints3d
                        net_joints3d = self.network.output.joints.reshape(-1, T, 31, 3)[:, -1:, :, :]  # (batch, 1, 31, 3)
                        verify_2d = full_perspective_projection(
                            net_joints3d[0:1], last_intrinsics, translation=last_cam
                        )[0, 0].cpu().numpy()
                        logger.info(f"  verify_2d (nose): {verify_2d[0]}")
                        logger.info(f"  verify_2d range: x=[{verify_2d[:,0].min():.0f},{verify_2d[:,0].max():.0f}] y=[{verify_2d[:,1].min():.0f},{verify_2d[:,1].max():.0f}]")
                except Exception as e:
                    logger.debug(f"Failed to project NTU joints to 2D: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            img = self._visualize_frame(frame, frame_idx, joints2d, action_label, action_confidence, ntu_joints_2d=ntu_joints_2d)
            
            if is_camera and self.visualize:
                cv2.imshow('WHAM Real-time Tracking', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
            
            viz_end = time.time()
            self.timing_stats['visualization'].append(viz_end - viz_start)
            
            # ✅ 保存skeleton视频
            self._write_skeleton_frame(img)
            
            # 总时间
            frame_time = time.time() - frame_start_time
            self.timing_stats['total'].append(frame_time)
            
            # 打印统计
            self._print_timing_stats(frame_idx)
            
            frame_idx += skip
            bar.next()
            
            # 定期清理缓存
            if frame_idx % 100 == 0:
                self.stream_inference.clear_cache()
            
            if is_camera and frame_idx >= self.max_frames * skip:
                logger.info(f"Reached maximum frames ({self.max_frames})")
                break
        
        bar.finish()
        
        # ✅ 保存所有帧的output到pkl文件
        self._save_frame_outputs()
        
        # 清理资源
        self._cleanup()
        
        # 最终统计
        logger.info(f"\nTotal processing time: {time.time() - start_total:.2f}s")
        self.stream_inference.print_stats()
        if self.flip_eval:
            total = self.flip_mode_frames + self.normal_mode_frames
            logger.info(f"Flip mode: {self.flip_mode_frames} frames, Normal mode: {self.normal_mode_frames} frames (total: {total})")
        
        return self.results

    def _save_frame_outputs(self):
        """✅ 保存所有帧的output到pkl文件"""
        if len(self.frame_outputs) > 0:
            output_pkl_path = os.path.join(self.output_path, "frame_outputs.pkl")
            joblib.dump(self.frame_outputs, output_pkl_path)
            logger.info(f"Saved {len(self.frame_outputs)} frame outputs to: {output_pkl_path}")
            
            # 同时保存一个更易读的总结文件
            summary_path = os.path.join(self.output_path, "output_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Total frames processed: {len(self.frame_outputs)}\n")
                f.write(f"Output path: {self.output_path}\n")
                f.write(f"Raw video: output_raw.mp4\n")
                f.write(f"Skeleton video: output_skeleton.mp4\n")
                f.write(f"Frame outputs: frame_outputs.pkl\n")
                f.write(f"\nOutput keys per frame:\n")
                if self.frame_outputs:
                    for key in self.frame_outputs[0]['output'].keys():
                        f.write(f"  - {key}\n")
            logger.info(f"Saved summary to: {summary_path}")

    def _handle_no_detection(self, frame, is_camera):
        """处理无检测情况"""
        import cv2
        img = frame.copy()
        if self.visualize:
            cv2.putText(img, "No Person Detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            if is_camera:
                cv2.imshow('WHAM Real-time Tracking', img)
                cv2.waitKey(1)
        self._write_skeleton_frame(img)
        return img

    def _handle_tracking_lost(self, frame, is_camera):
        """处理跟踪丢失情况"""
        import cv2
        img = frame.copy()
        if self.visualize:
            cv2.putText(img, "Tracking Lost", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
            if is_camera:
                cv2.imshow('WHAM Real-time Tracking', img)
                cv2.waitKey(1)
        self._write_skeleton_frame(img)
        return img

    def _process_frame_data(self, subject_data, slam_data):
        """处理帧数据"""
        from lib.data.datasets.dataset_custom import convert_dpvo_to_cam_angvel
        from lib.data.utils.normalizer import Normalizer
        from lib.utils.imutils import compute_cam_intrinsics
        
        tt = lambda x: x.unsqueeze(0).to(self.device)
        
        kp2d = torch.from_numpy(subject_data['keypoints'][-self.window_size:]).float()
        mask = kp2d[..., -1] < 0.3
        bbox = torch.from_numpy(subject_data['bbox'][-self.window_size:]).float()

        # bbox 平滑：对最后一帧的 bbox 做 EMA，消除检测器帧间跳动
        # bbox 经 convert_pare_to_full_img_cam 会放大到 full_cam，是 2D 抖动的主要来源
        cur_bbox = bbox[-1:].clone()  # [1, 3] (cx, cy, scale)
        if self._prev_smooth_bbox is not None:
            smooth_bbox = self._bbox_smooth_alpha * cur_bbox + (1 - self._bbox_smooth_alpha) * self._prev_smooth_bbox
        else:
            smooth_bbox = cur_bbox
        self._prev_smooth_bbox = smooth_bbox
        bbox[-1:] = smooth_bbox

        res = torch.tensor([self.width, self.height]).float()
        intrinsics = compute_cam_intrinsics(res)

        normalizer = Normalizer(self.cfg)
        norm_kp2d, _ = normalizer(kp2d[..., :-1].clone(), res, intrinsics, 224, 224, bbox)
        
        features = tt(subject_data['features'][-self.window_size:]) if subject_data['features'][-self.window_size:] != [None] else None
        cam_angvel = convert_dpvo_to_cam_angvel(slam_data, self.fps)
        
        return (tt(norm_kp2d), features, tt(mask), tt(cam_angvel), subject_data['frame_id'],
                {'cam_intrinsics': tt(intrinsics), 'bbox': tt(bbox), 'res': tt(res)})

    def _should_do_flip_for_frame(self, frame_idx, subject_data):
        """判断当前帧是否需要做 flip（仅部分帧 flip 以节省计算）

        oblique 模式使用 view_encoder 风格的 3D 几何特征判断：
        - 髋部/肩部宽度向量的 z 分量（深度差）反映身体旋转程度
        - 左右深度差越大，说明身体越偏向斜侧角，flip 越有价值
        """
        if not self.flip_eval:
            return False
        if self.flip_select == 'all':
            return True
        if self.flip_select == 'interval':
            return (frame_idx % self.flip_interval) == 0
        if self.flip_select == 'oblique':
            # 用 pred_kp3d 的 3D 几何特征判断是否为斜侧角
            # 与 MinimalViewEncoder 使用相同的关节索引和特征思路
            try:
                pred_kp3d = getattr(self.network, 'pred_kp3d', None)
                if pred_kp3d is None:
                    return False
                # pred_kp3d: [B, T, J, 3], 取最后一帧
                kp3d = pred_kp3d[0, -1]  # [J, 3]
                if kp3d.shape[0] < 13:
                    return False

                left_hip = kp3d[11]       # [3]
                right_hip = kp3d[12]      # [3]
                left_shoulder = kp3d[5]   # [3]
                right_shoulder = kp3d[6]  # [3]

                # 髋部和肩部的左右深度差（z分量差异）
                # 正面/背面时深度差接近0，斜侧角时深度差明显
                hip_depth_diff = abs(float(left_hip[2] - right_hip[2]))
                shoulder_depth_diff = abs(float(left_shoulder[2] - right_shoulder[2]))

                # 用髋部宽度作为归一化参考（避免距离远近影响）
                hip_width = float(torch.norm(left_hip - right_hip)) + 1e-6

                # 深度不对称比例：depth_diff / width 近似 sin(旋转角)
                oblique_ratio = (hip_depth_diff + shoulder_depth_diff) / (2.0 * hip_width)

                # oblique_ratio > 0.3 大约对应旋转角 > ~17°，视为斜侧角
                return oblique_ratio > 0.3
            except Exception:
                return False
        return False

    def _process_flipped_frame_data(self, subject_data, slam_data):
        """处理翻转帧数据（用于flip evaluation）"""
        from lib.data.datasets.dataset_custom import convert_dpvo_to_cam_angvel
        from lib.data.utils.normalizer import Normalizer
        from lib.utils.imutils import compute_cam_intrinsics

        tt = lambda x: x.unsqueeze(0).to(self.device)

        kp2d = torch.from_numpy(subject_data['flipped_keypoints'][-self.window_size:]).float()
        mask = kp2d[..., -1] < 0.3
        bbox = torch.from_numpy(subject_data['flipped_bbox'][-self.window_size:]).float()
        res = torch.tensor([self.width, self.height]).float()
        intrinsics = compute_cam_intrinsics(res)

        normalizer = Normalizer(self.cfg)
        norm_kp2d, _ = normalizer(kp2d[..., :-1].clone(), res, intrinsics, 224, 224, bbox)

        flipped_feats = subject_data['flipped_features'][-self.window_size:]
        if isinstance(flipped_feats, list) and (len(flipped_feats) == 0 or flipped_feats[0] is None):
            features = None
        elif isinstance(flipped_feats, torch.Tensor):
            features = tt(flipped_feats)
        elif isinstance(flipped_feats, list) and all(isinstance(t, torch.Tensor) for t in flipped_feats):
            features = tt(torch.stack(flipped_feats))
        else:
            features = None
        cam_angvel = convert_dpvo_to_cam_angvel(slam_data, self.fps)

        return (tt(norm_kp2d), features, tt(mask), tt(cam_angvel), subject_data['frame_id'],
                {'cam_intrinsics': tt(intrinsics), 'bbox': tt(bbox), 'res': tt(res)})

    def _process_init(self, subject_data, norm_kp2d):
        """处理初始化"""
        from lib.utils.kp_utils import root_centering
        from lib.utils import transforms

        tt = lambda x: x.unsqueeze(0).to(self.device)

        init_output = self.network.smpl.get_output(
            global_orient=subject_data['init_global_orient'].to(self.device),
            body_pose=subject_data['init_body_pose'].to(self.device),
            betas=subject_data['init_betas'].to(self.device),
            pose2rot=False,
            return_full_pose=True
        )

        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        init_kp = tt(torch.cat((init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1).to(self.device)), dim=-1))
        init_smpl = tt(transforms.matrix_to_rotation_6d(init_output.full_pose))
        init_root = transforms.matrix_to_rotation_6d(init_output.global_orient).to(self.device)

        return (init_kp, init_smpl), init_root

    def _predict_flipped_init(self, frame, frame_id, subject_id, tracking_results):
        """flip 无上一帧时，用 extractor.predict_init 获取 flipped init（优先 extractor_engine）"""
        from lib.models.preproc.backbone.utils import process_image
        subject_data = tracking_results[subject_id]
        if frame_id not in subject_data['frame_id']:
            return tracking_results
        bbox_idx = np.where(np.array(subject_data['frame_id']) == frame_id)[0][0]
        bbox = subject_data['bbox'][bbox_idx]
        cx, cy, scale = float(bbox[0]), float(bbox[1]), float(bbox[2])
        norm_img, _ = process_image(frame[..., ::-1], [cx, cy], scale, 256, 256)
        norm_img = torch.from_numpy(norm_img).unsqueeze(0).to(self.device)
        flipped_norm_img = torch.flip(norm_img, (3,))
        # 使用当前 extractor 的 predict_init（避免额外加载 extractor_engine 导致 OOM）
        return self.extractor.predict_init(flipped_norm_img, tracking_results, subject_id, flip_eval=True)

    def _make_fallback_init(self, norm_kp2d):
        """当无 SMPL init 时，用 zeros(3d) + 2d 关键点构建 fallback init（避免 init_kp.dim() 报错）"""
        # init_kp: [init_kp3d(51) + norm_kp2d(34)] 与 _process_init 格式一致
        init_kp3d = torch.zeros(1, 51, device=self.device)
        kp2d = norm_kp2d[:, -1] if norm_kp2d.dim() == 3 else norm_kp2d[0] if norm_kp2d.dim() == 2 else norm_kp2d
        kp2d_flat = kp2d.reshape(1, -1).to(self.device) if kp2d.numel() > 0 else torch.zeros(1, 34, device=self.device)
        init_kp = torch.cat((init_kp3d, kp2d_flat), dim=-1).to(self.device)
        init_smpl = torch.zeros(1, 144, device=self.device)
        init_root = torch.zeros(1, 6, device=self.device)
        return (init_kp, init_smpl), init_root

    def _process_flipped_init(self, subject_data, flipped_norm_kp2d):
        """处理翻转初始化（用于flip evaluation）"""
        from lib.utils.kp_utils import root_centering
        from lib.utils import transforms

        tt = lambda x: x.unsqueeze(0).to(self.device)

        init_output = self.network.smpl.get_output(
            global_orient=subject_data['flipped_init_global_orient'].to(self.device),
            body_pose=subject_data['flipped_init_body_pose'].to(self.device),
            betas=subject_data['flipped_init_betas'].to(self.device),
            pose2rot=False,
            return_full_pose=True
        )

        init_kp3d = root_centering(init_output.joints[:, :17], 'coco')
        init_kp = tt(torch.cat((init_kp3d.reshape(1, -1), flipped_norm_kp2d[0].clone().reshape(1, -1).to(self.device)), dim=-1))
        init_smpl = tt(transforms.matrix_to_rotation_6d(init_output.full_pose))
        init_root = transforms.matrix_to_rotation_6d(init_output.global_orient).to(self.device)

        return (init_kp, init_smpl), init_root

    def _cleanup(self):
        """✅ 清理资源（顺序重要，避免 segfault）"""
        import cv2
        
        if self.is_realsense_camera:
            self.pipeline.stop()
            logger.info("RealSense camera stopped")
        
        if self.is_fisheye_camera or self.is_video_file:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        
        is_camera = self.is_realsense_camera or self.is_fisheye_camera
        if is_camera and self.visualize:
            cv2.destroyAllWindows()
        
        # ✅ 释放原始视频写入器（imageio 用 close，cv2 用 release）
        if self.output_raw_video is not None:
            try:
                if getattr(self, '_use_imageio_writer', False):
                    self.output_raw_video.close()
                else:
                    self.output_raw_video.release()
                logger.info(f"Raw video saved: {self.output_raw_video_path}")
            except Exception as e:
                logger.warning(f"Error closing raw video: {e}")
            self.output_raw_video = None
        
        # ✅ 释放skeleton视频写入器
        if self.output_skeleton_video is not None:
            try:
                if getattr(self, '_use_imageio_writer', False):
                    self.output_skeleton_video.close()
                else:
                    self.output_skeleton_video.release()
                logger.info(f"Skeleton video saved: {self.output_skeleton_video_path}")
            except Exception as e:
                logger.warning(f"Error closing skeleton video: {e}")
            self.output_skeleton_video = None
        
        self.stream_inference.clear_cache()
        if self.flip_eval and hasattr(self, 'flipped_stream_inference'):
            self.flipped_stream_inference.clear_cache()
        
        # 显式释放 detector (TensorRT/PyCUDA)，避免 __del__ 在解释器退出时触发 segfault
        if hasattr(self, 'detector') and self.detector is not None:
            try:
                if hasattr(self.detector, 'cleanup'):
                    self.detector.cleanup()
            except Exception as e:
                logger.warning(f"Detector cleanup: {e}")
            self.detector = None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

def run(cfg, video, output_pth, network, calib=None, window_size=50,
                  run_global=True, save_pkl=False, visualize=False, max_frames=1000,
                  enable_adaptive_window=True, min_window=5, max_window=15,
                  action_config=None, action_checkpoint=None, action_label_map=None, action_engine=None,
                  flip_select='all', flip_interval=5):
    """
    运行优化的WHAM处理器
    """
    start_total = time.time()
    
    processor = OptimizedSequentialVideoProcessor(
        cfg, video, output_pth, network,
        calib=calib,
        run_global=run_global,
        save_pkl=save_pkl,
        visualize=visualize,
        window_size=window_size,
        max_frames=max_frames,
        enable_adaptive_window=enable_adaptive_window,
        min_window=min_window,
        max_window=max_window,
        action_config=action_config,
        action_checkpoint=action_checkpoint,
        action_label_map=action_label_map,
        action_engine=action_engine,
        flip_select=flip_select,
        flip_interval=flip_interval
    )
    
    results = processor.run()
    logger.info(f"Total time: {time.time() - start_total:.2f}s")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video', type=str,
                        default='examples/demo_video.mp4',
                        help='input video path, youtube link, or "realsense" for RealSense camera, or "fisheye" for a V4L2 fisheye camera')

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

    parser.add_argument('--max_frames', type=int, default=1000,
                        help='Maximum number of frames to process from camera')

    parser.add_argument('--enable_adaptive_window', action='store_true', default=True,
                        help='Enable adaptive window size prediction')

    parser.add_argument('--min_window', type=int, default=5,
                        help='Minimum window size for adaptive mode')

    parser.add_argument('--max_window', type=int, default=15,
                        help='Maximum window size for adaptive mode')

    parser.add_argument('--action_config', type=str, default=None,
                        help='Path to pyskl action recognition config file')
    
    parser.add_argument('--action_checkpoint', type=str, default=None,
                        help='Path to pyskl action recognition checkpoint file')
    
    parser.add_argument('--action_label_map', type=str, default=None,
                        help='Path to action label map file (e.g., nturgbd_120.txt)')
    
    parser.add_argument('--action_engine', type=str, default=None,
                        help='Path to TensorRT engine file for action recognition (optional, faster than PyTorch)')

    parser.add_argument('--flip_eval', action='store_true',
                        help='Enable flip evaluation for more accurate pose estimation (runs two forward passes per frame)')
    parser.add_argument('--flip_select', type=str, default='all', choices=['all', 'oblique', 'interval'],
                        help='Which frames to flip: all=every frame, oblique=only oblique/side-angle frames, interval=every N frames')
    parser.add_argument('--flip_interval', type=int, default=5,
                        help='When flip_select=interval, flip every N frames (default: 5)')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    if args.flip_eval:
        cfg.FLIP_EVAL = True

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # Load WHAM model
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    # Prepare output directory
    if args.video == "realsense":
        sequence = "realsense_capture"
    elif args.video == "fisheye":
        sequence = "fisheye_capture"
    else:
        sequence = '.'.join(args.video.split('/')[-1].split('.')[:-1])

    output_pth = osp.join(args.output_pth, sequence)
    os.makedirs(output_pth, exist_ok=True)

    # Run processing
    run(cfg,
        args.video,
        output_pth,
        network,
        args.calib,
        window_size=10,
        run_global=not args.estimate_local_only,
        save_pkl=args.save_pkl,
        visualize=args.visualize,
        max_frames=args.max_frames,
        enable_adaptive_window=args.enable_adaptive_window,
        min_window=args.min_window,
        max_window=args.max_window,
        action_config=args.action_config,
        action_checkpoint=args.action_checkpoint,
        action_label_map=args.action_label_map,
        action_engine=args.action_engine,
        flip_select=args.flip_select,
        flip_interval=args.flip_interval)

    print()
    logger.info('Done !')
    import os
    os._exit(0)
