import numpy as np
import torch
from lib.utils import transforms
from lib.eval.freeman_loader import FreeMan
from aniposelib.cameras import Camera, CameraGroup
from lib.data.utils.normalizer import Normalizer
import cv2

class FreemanDataset:
    def __init__(self, base_dir, fps=25, split=''):
        self.freeman = FreeMan(base_dir, fps, split)
        self.n_joints = 17  # COCO format
        self.fps = fps
        self.keypoints_normalizer = Normalizer(None)
        
    def load_data(self, session_name):
        target = self.__getitem__(session_name)
        for key, val in target.items():
            if isinstance(val, torch.Tensor):
                target[key] = val.unsqueeze(0)
        return target   

    def __getitem__(self, index):
        target = {}
        target = self.get_data(index)
        target = self.prepare_keypoints_data(target)
        target = self.prepare_smpl_data(target)
        return target

    def get_data(self, index):
        target = {}
        
        target = self.prepare_labels(index, target)
        target = self.prepare_inputs(index, target)
        target = self.prepare_initialization(index, target)
        
        return target
        

    def prepare_inputs(self, index, target):
        session_name = self.freeman.session_list[index]
        
        # Load 2D keypoints (without bbox)
        keypoints2d, center, scale = self.freeman.load_keypoints2d(
            self.freeman.keypoints2d_dir, 
            session_name, 
            bbox_dir=None
        )
        view_id=0
        keypoints2d = torch.from_numpy(keypoints2d).float()[view_id]
               
        # Normalize keypoints
        kp2d, bbox = self.keypoints_normalizer(
            keypoints2d[..., :2],
            target['res'], 
            target['cam_intrinsics'], 
            224, 224, 
            bbox=None
        )
        
        target['kp2d'] = kp2d
        target['bbox'] = bbox[1:]
        
        # Mask low confidence keypoints
        mask = keypoints2d[..., 2] < 0.3
        target['input_kp2d'] = keypoints2d[1:]
        target['input_kp2d'][mask[1:]] *= 0
        target['mask'] = mask[1:]
        
        return target

    def prepare_smpl_data(self, target):
        if 'pose' in target.keys():
            # Use only the main joints
            pose = target['pose'][:]
            # 6-D Rotation representation
            pose6d = transforms.matrix_to_rotation_6d(pose)
            target['pose'] = pose6d[1:]
        
        if 'betas' in target.keys():
            target['betas'] = target['betas'][1:]
        
        # Translation and shape parameters
        if 'transl' in target.keys():
            target['cam'] = target['transl'][1:]
        
        # Initial pose and translation
        target['init_pose'] = transforms.matrix_to_rotation_6d(target['init_pose'])

        return target

    def prepare_keypoints_data(self, target):
        """Prepare keypoints data"""
        
        # Prepare 2D keypoints
        target['init_kp2d'] = target['kp2d'][:1]
        target['kp2d'] = target['kp2d'][1:]
        if 'kp3d' in target:
            target['kp3d'] = target['kp3d'][1:]

        return target

    def prepare_labels(self, index, target):
        session_name = self.freeman.session_list[index]
        
        # Load SMPL parameters
        smpl_poses, smpl_scaling, smpl_trans = self.freeman.load_motion(
            self.freeman.motion_dir, 
            session_name
        )
        
        # Convert poses to rotation matrices
        target['pose'] = transforms.axis_angle_to_matrix(
            torch.from_numpy(smpl_poses).reshape(-1, 24, 3)
        )
        target['betas'] = torch.from_numpy(smpl_scaling)
        
        # Load camera parameters
        camera_group, camera_params = self.freeman.load_camera_group(
            self.freeman.camera_dir,
            session_name
        )
        
        # Set camera information
        target['res'] = [camera_params[0]['size'][1], camera_params[0]['size'][0]]  # [H, W]
        target['vid'] = session_name
        target['frame_id'] = np.arange(len(smpl_poses))[1:]
        
        # Camera intrinsics and poses
        target['cam_intrinsics'] = torch.from_numpy(np.array(camera_params[0]['matrix']))
        R = torch.from_numpy(cv2.Rodrigues(camera_group.cameras[0].rvec)[0])
        
        # Calculate camera angular velocity
        cam_angvel = torch.zeros((len(target['pose']) - 1, 6))
        target['cam_angvel'] = cam_angvel
        return target


    def prepare_initialization(self, index, target):
        session_name = self.freeman.session_list[index]
        
        # Load 3D keypoints for initialization
        keypoints3d = self.freeman.load_keypoints3d(
            self.freeman.keypoints3d_dir,
            session_name
        )
        
        # Initial frame estimation
        target['init_kp3d'] = self.root_centering(
            torch.from_numpy(keypoints3d[:1, :self.n_joints])
        ).reshape(1, -1)
        
        # Initial pose
        target['init_pose'] = torch.zeros(target['pose'].shape)[0:1]
        
        pose_root = target['pose'][:, 0].clone()
        target['init_root'] = transforms.matrix_to_rotation_6d(pose_root)
        
        return target



    def root_centering(self, keypoints):
        """Center keypoints around root joint"""
        root_joint = keypoints[..., 0:1, :]
        return keypoints - root_joint

