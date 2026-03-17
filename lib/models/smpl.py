from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys

import torch
import numpy as np
from lib.utils import transforms

from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput as ModelOutput
from smplx.lbs import vertices2joints

from configs import constants as _C

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        super(SMPL, self).__init__(*args, **kwargs)
        sys.stdout = sys.__stdout__
        
        J_regressor_movid = np.load(_C.BMODEL.JOINTS_REGRESSOR_MOVID)
        J_regressor_eval = np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        self.register_buffer('J_regressor_movid', torch.tensor(
            J_regressor_movid, dtype=torch.float32))
        self.register_buffer('J_regressor_eval', torch.tensor(
            J_regressor_eval, dtype=torch.float32))
        self.register_buffer('J_regressor_feet', torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_FEET)
        ).float())
        # Try to load extra regressor for finger joints
        if os.path.exists(_C.BMODEL.JOINTS_REGRESSOR_EXTRA):
            self.register_buffer('J_regressor_extra', torch.from_numpy(
                np.load(_C.BMODEL.JOINTS_REGRESSOR_EXTRA)
            ).float())
        else:
            self.J_regressor_extra = None
        
        # Try to load NTU regressor if available
        ntu_regressor_path = _C.BMODEL.FLDR.replace('smpl/', 'J_regressor_ntu.npy')
        if os.path.exists(ntu_regressor_path):
            self.register_buffer('J_regressor_ntu', torch.from_numpy(
                np.load(ntu_regressor_path)
            ).float())
        else:
            # Create approximate NTU 25 keypoints from SMPL joints
            # NTU format: 25 joints mapped from SMPL 24 joints + additional points
            self.J_regressor_ntu = None
        
    def get_local_pose_from_reduced_global_pose(self, reduced_pose):
        full_pose = torch.eye(
            3, device=reduced_pose.device
        )[(None, ) * 2].repeat(reduced_pose.shape[0], 24, 1, 1)
        full_pose[:, _C.BMODEL.MAIN_JOINTS] = reduced_pose
        return full_pose

    def forward(self, 
                pred_rot6d, 
                betas, 
                cam=None, 
                cam_intrinsics=None, 
                bbox=None, 
                res=None,
                return_full_pose=False,
                **kwargs):
        
        rotmat = transforms.rotation_6d_to_matrix(pred_rot6d.reshape(*pred_rot6d.shape[:2], -1, 6)
        ).reshape(-1, 24, 3, 3)

        output = self.get_output(body_pose=rotmat[:, 1:],
                                 global_orient=rotmat[:, :1],
                                 betas=betas.view(-1, 10),
                                 pose2rot=False,
                                 return_full_pose=return_full_pose)

        if cam is not None:
            joints3d = output.joints.reshape(*cam.shape[:2], -1, 3)
            
            # Weak perspective projection (for InstaVariety)
            weak_cam = convert_weak_perspective_to_perspective(cam)
            
            weak_joints2d = weak_perspective_projection(
                joints3d,
                rotation=torch.eye(3, device=cam.device).unsqueeze(0).unsqueeze(0).expand(*cam.shape[:2], -1, -1),
                translation=weak_cam,
                focal_length=5000.,
                camera_center=torch.zeros(*cam.shape[:2], 2, device=cam.device)
            )
            output.weak_joints2d = weak_joints2d
            
            # Full perspective projection
            full_cam = convert_pare_to_full_img_cam(
                cam, 
                bbox[:, :, 2] * 200., 
                bbox[:, :, :2], 
                res[:, 0].unsqueeze(-1), 
                res[:, 1].unsqueeze(-1), 
                focal_length=cam_intrinsics[:, :, 0, 0]
            )
            
            full_joints2d = full_perspective_projection(
                joints3d,
                translation=full_cam,
                cam_intrinsics=cam_intrinsics,
            )
            output.full_joints2d = full_joints2d
            output.full_cam = full_cam.reshape(-1, 3)
            
        return output
    
    def forward_nd(self, 
                pred_rot6d, 
                root,
                betas, 
                return_full_pose=False):
        
        rotmat = transforms.rotation_6d_to_matrix(pred_rot6d.reshape(*pred_rot6d.shape[:2], -1, 6)
        ).reshape(-1, 24, 3, 3)

        output = self.get_output(body_pose=rotmat[:, 1:],
                                 global_orient=root.reshape(-1, 1, 3, 3),
                                 betas=betas.view(-1, 10),
                                 pose2rot=False,
                                 return_full_pose=return_full_pose)

        return output

    def get_output(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = vertices2joints(self.J_regressor_movid, smpl_output.vertices)
        feet = vertices2joints(self.J_regressor_feet, smpl_output.vertices)
        
        offset = joints[..., [11, 12], :].mean(-2)
        if 'transl' in kwargs:
            offset = offset - kwargs['transl']
        vertices = smpl_output.vertices - offset.unsqueeze(-2)
        joints = joints - offset.unsqueeze(-2)
        feet = feet - offset.unsqueeze(-2)

        output = ModelOutput(vertices=vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        output.feet = feet
        output.offset = offset
        return output
    
    def get_offset(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        joints = vertices2joints(self.J_regressor_movid, smpl_output.vertices)
        
        offset = joints[..., [11, 12], :].mean(-2)
        return offset
    
    def get_ntu_joints(self, vertices):
        """
        Extract NTU RGB+D 25 keypoints from SMPL vertices

        Args:
            vertices: SMPL vertices, shape (..., 6890, 3)

        Returns:
            ntu_joints: NTU 25 keypoints, shape (..., 25, 3)

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

        SMPL 24 native joints:
            0: pelvis, 1: left_hip, 2: right_hip, 3: spine1 (lower),
            4: left_knee, 5: right_knee, 6: spine2 (mid), 7: left_ankle,
            8: right_ankle, 9: spine3 (upper), 10: left_foot, 11: right_foot,
            12: neck, 13: left_collar, 14: right_collar, 15: head,
            16: left_shoulder, 17: right_shoulder, 18: left_elbow, 19: right_elbow,
            20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand

        MoViD J_regressor_movid returns 31 joints in COCO format:
            0-16: COCO 17 joints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
            17-30: Additional joints

        COCO 17 joint order:
            0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
            5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
            9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
            13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        """
        if self.J_regressor_ntu is not None:
            # Use pre-computed NTU regressor if available
            return vertices2joints(self.J_regressor_ntu, vertices)
        else:
            # Get SMPL native joints (24 joints) - contains proper spine joints
            smpl_joints = vertices2joints(self.J_regressor, vertices)  # (..., 24, 3)

            # Create NTU 25 joints array
            ntu_joints = torch.zeros((*smpl_joints.shape[:-2], 25, 3),
                                    device=smpl_joints.device,
                                    dtype=smpl_joints.dtype)

            # ===== Spine joints from SMPL native (accurate positions) =====
            # SMPL: 0=pelvis, 3=spine1, 6=spine2, 9=spine3, 12=neck, 15=head

            # NTU 0: Base of spine (pelvis)
            ntu_joints[..., 0, :] = smpl_joints[..., 0, :]  # SMPL pelvis

            # NTU 1: Mid spine - use SMPL spine2 (mid spine)
            ntu_joints[..., 1, :] = smpl_joints[..., 6, :]  # SMPL spine2

            # NTU 20: Spine (upper spine, between neck and mid-spine)
            ntu_joints[..., 20, :] = smpl_joints[..., 9, :]  # SMPL spine3

            # NTU 2: Neck
            ntu_joints[..., 2, :] = smpl_joints[..., 12, :]  # SMPL neck

            # NTU 3: Head
            ntu_joints[..., 3, :] = smpl_joints[..., 15, :]  # SMPL head

            # ===== Arm joints from SMPL native =====
            # SMPL: 16=left_shoulder, 17=right_shoulder, 18=left_elbow, 19=right_elbow,
            #       20=left_wrist, 21=right_wrist, 22=left_hand, 23=right_hand

            # NTU 4-7: Left arm (shoulder, elbow, wrist, hand)
            ntu_joints[..., 4, :] = smpl_joints[..., 16, :]   # SMPL left shoulder
            ntu_joints[..., 5, :] = smpl_joints[..., 18, :]   # SMPL left elbow
            ntu_joints[..., 6, :] = smpl_joints[..., 20, :]   # SMPL left wrist
            ntu_joints[..., 7, :] = smpl_joints[..., 22, :]   # SMPL left hand

            # NTU 8-11: Right arm (shoulder, elbow, wrist, hand)
            ntu_joints[..., 8, :] = smpl_joints[..., 17, :]   # SMPL right shoulder
            ntu_joints[..., 9, :] = smpl_joints[..., 19, :]   # SMPL right elbow
            ntu_joints[..., 10, :] = smpl_joints[..., 21, :]  # SMPL right wrist
            ntu_joints[..., 11, :] = smpl_joints[..., 23, :]  # SMPL right hand

            # ===== Leg joints from SMPL native =====
            # SMPL: 1=left_hip, 2=right_hip, 4=left_knee, 5=right_knee,
            #       7=left_ankle, 8=right_ankle, 10=left_foot, 11=right_foot

            # NTU 12-15: Left leg (hip, knee, ankle, foot)
            ntu_joints[..., 12, :] = smpl_joints[..., 1, :]   # SMPL left hip
            ntu_joints[..., 13, :] = smpl_joints[..., 4, :]   # SMPL left knee
            ntu_joints[..., 14, :] = smpl_joints[..., 7, :]   # SMPL left ankle

            # NTU 16-19: Right leg (hip, knee, ankle, foot)
            ntu_joints[..., 16, :] = smpl_joints[..., 2, :]   # SMPL right hip
            ntu_joints[..., 17, :] = smpl_joints[..., 5, :]   # SMPL right knee
            ntu_joints[..., 18, :] = smpl_joints[..., 8, :]   # SMPL right ankle
            
            # ===== Foot joints =====
            # SMPL has foot joints: 10=left_foot, 11=right_foot
            # Use SMPL native foot joints directly
            ntu_joints[..., 15, :] = smpl_joints[..., 10, :]  # SMPL left foot
            ntu_joints[..., 19, :] = smpl_joints[..., 11, :]  # SMPL right foot

            # ===== Hand tip joints (NTU 21-24) =====
            # Extend from hand position for finger tips
            # SMPL: 20=left_wrist, 21=right_wrist, 22=left_hand, 23=right_hand

            # NTU 21-22: Left hand tips (tip of left hand, left thumb)
            left_hand = smpl_joints[..., 22, :]   # SMPL left hand
            left_wrist = smpl_joints[..., 20, :]  # SMPL left wrist
            finger_dir_left = left_hand - left_wrist
            finger_dir_left_norm = torch.norm(finger_dir_left, dim=-1, keepdim=True) + 1e-8
            finger_dir_left = finger_dir_left / finger_dir_left_norm
            # Tip of left hand - extend further forward
            ntu_joints[..., 21, :] = left_hand + 0.08 * finger_dir_left
            # Left thumb - extend slightly to the side
            up_vec = torch.zeros_like(finger_dir_left)
            up_vec[..., 2] = 1.0  # Z-up
            thumb_dir_left = torch.cross(finger_dir_left, up_vec, dim=-1)
            thumb_dir_left_norm = torch.norm(thumb_dir_left, dim=-1, keepdim=True) + 1e-8
            thumb_dir_left = thumb_dir_left / thumb_dir_left_norm
            ntu_joints[..., 22, :] = left_hand + 0.06 * finger_dir_left + 0.05 * thumb_dir_left

            # NTU 23-24: Right hand tips (tip of right hand, right thumb)
            right_hand = smpl_joints[..., 23, :]   # SMPL right hand
            right_wrist = smpl_joints[..., 21, :]  # SMPL right wrist
            finger_dir_right = right_hand - right_wrist
            finger_dir_right_norm = torch.norm(finger_dir_right, dim=-1, keepdim=True) + 1e-8
            finger_dir_right = finger_dir_right / finger_dir_right_norm
            # Tip of right hand - extend further forward
            ntu_joints[..., 23, :] = right_hand + 0.08 * finger_dir_right
            # Right thumb - extend slightly to the side
            thumb_dir_right = torch.cross(finger_dir_right, up_vec, dim=-1)
            thumb_dir_right_norm = torch.norm(thumb_dir_right, dim=-1, keepdim=True) + 1e-8
            thumb_dir_right = thumb_dir_right / thumb_dir_right_norm
            ntu_joints[..., 24, :] = right_hand + 0.06 * finger_dir_right + 0.05 * thumb_dir_right

            return ntu_joints
    

def convert_weak_perspective_to_perspective(
        weak_perspective_camera,
        focal_length=5000.,
        img_res=224,
):
    
    perspective_camera = torch.stack(
        [
            weak_perspective_camera[..., 1],
            weak_perspective_camera[..., 2],
            2 * focal_length / (img_res * weak_perspective_camera[..., 0] + 1e-9)
        ],
        dim=-1
    )
    return perspective_camera    


def weak_perspective_projection(
        points, 
        rotation, 
        translation,
        focal_length, 
        camera_center, 
        img_res=224,
        normalize_joints2d=True,
):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (b, f, N, 3): 3D points
        rotation (b, f, 3, 3): Camera rotation
        translation (b, f, 3): Camera translation
        focal_length (b, f,) or scalar: Focal length
        camera_center (b, f, 2): Camera center
    """

    K = torch.zeros([*points.shape[:2], 3, 3], device=points.device)
    K[:,:,0,0] = focal_length
    K[:,:,1,1] = focal_length
    K[:,:,2,2] = 1.
    K[:,:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bfij,bfkj->bfki', rotation, points)
    points = points + translation.unsqueeze(-2)

    # Apply perspective distortion
    projected_points = points / points[...,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bfij,bfkj->bfki', K, projected_points)
    
    if normalize_joints2d:
        projected_points = projected_points / (img_res / 2.) 

    return projected_points[..., :-1]

    
def full_perspective_projection(
        points, 
        cam_intrinsics, 
        rotation=None,
        translation=None,
):

    K = cam_intrinsics

    if rotation is not None:
        points = (rotation @ points.transpose(-1, -2)).transpose(-1, -2)
    if translation is not None:
        points = points + translation.unsqueeze(-2)
    projected_points = points / points[..., -1].unsqueeze(-1)
    projected_points = (K @ projected_points.transpose(-1, -2)).transpose(-1, -2)
    return projected_points[..., :-1]


def convert_pare_to_full_img_cam(
        pare_cam, 
        bbox_height, 
        bbox_center,
        img_w, 
        img_h, 
        focal_length, 
        crop_res=224
):

    s, tx, ty = pare_cam[..., 0], pare_cam[..., 1], pare_cam[..., 2]
    res = crop_res
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[..., 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[..., 1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    return cam_t


def cam_crop2full(crop_cam, center, scale, full_img_shape, focal_length):
    """
    convert the camera parameters from the crop camera to the full camera
    :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
    :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
    :param scale: shape=(N) square bbox resolution  (b / 200)
    :param full_img_shape: shape=(N, 2) original image height and width
    :param focal_length: shape=(N,)
    :return:
    """
    img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
    cx, cy, b = center[:, 0], center[:, 1], scale * 200
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * crop_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + crop_cam[:, 1]
    ty = (2 * (cy - h_2) / bs) + crop_cam[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam