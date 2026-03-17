from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from configs import constants as _C
from lib.models.layers import (MotionEncoder, MotionDecoder, TrajectoryDecoder, TrajectoryRefiner, Integrator, GatedFusion, EnhancedViewEncoder,LightweightMLP, CrossAttentionFusion,CLIPGatedFusion, ImprovedDynamicProjection,DynamicProjection,IMUProjection,ViewDecoder,MinimalViewEncoder,
                               rollout_global_motion, reset_root_velocity, compute_camera_motion)
from lib.utils.transforms import axis_angle_to_matrix
from lib.utils.kp_utils import root_centering
import math

import torch.nn.functional as F
import random

def plot_similarity_heatmap(motion_feat, view_feat, title='Similarity Heatmap', save_path=None, n_samples=30, random_seed=42):
    # Subsample for visualization
    np.random.seed(random_seed)
    idx_motion = np.random.choice(motion_feat.shape[0], min(n_samples, motion_feat.shape[0]), replace=False)
    idx_view = np.random.choice(view_feat.shape[0], min(n_samples, view_feat.shape[0]), replace=False)
    motion_feat = motion_feat[idx_motion]
    view_feat = view_feat[idx_view]

    motion_feat = normalize(motion_feat, axis=1)
    view_feat = normalize(view_feat, axis=1)
    similarity = np.matmul(motion_feat, view_feat.T)

    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    plt.figure(figsize=(8, 7))
    sns.heatmap(similarity, cmap='coolwarm', center=0, cbar=True, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

def safe_normalize(x, dim=1, eps=1e-6):
    """
    安全的归一化函数，防止除零
    
    参数:
    - x: 输入张量
    - dim: 归一化的维度
    - eps: 防止除零的小值
    
    返回:
    - 归一化后的张量
    """
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)

def transform_kp3d_to_front_view(kp3d, global_orient):
    """
    Transform 3D keypoints from current view to front view using global orientation.
    
    Args:
        kp3d: 3D keypoints in current view, shape [B, T, N, 3] or [B, T, N*3]
        global_orient: SMPL global orientation, shape [B, T, 3] (axis-angle) or [B, T, 6] (6D rotation)
    
    Returns:
        kp3d_front: 3D keypoints in front view, same shape as input
    """
    original_shape = kp3d.shape
    B, T = global_orient.shape[:2]
    
    # Reshape kp3d to [B, T, N, 3] if needed
    if len(kp3d.shape) == 3:  # [B, T, N*3]
        N = kp3d.shape[-1] // 3
        kp3d = kp3d.reshape(B, T, N, 3)
    else:  # [B, T, N, 3]
        N = kp3d.shape[2]
    
    # Check if global_orient is 6D or 3D (axis-angle)
    orient_dim = global_orient.shape[-1]
    
    if orient_dim == 6:
        # Convert 6D rotation to rotation matrix
        from pytorch3d.transforms import rotation_6d_to_matrix
        global_orient_flat = global_orient.reshape(B * T, 6)
        R_global = rotation_6d_to_matrix(global_orient_flat)  # [B*T, 3, 3]
        
    elif orient_dim == 3:
        # Convert axis-angle to rotation matrix

        from pytorch3d.transforms import axis_angle_to_matrix
        global_orient_flat = global_orient.reshape(B * T, 3)
        R_global = axis_angle_to_matrix(global_orient_flat)  # [B*T, 3, 3]

    else:
        raise ValueError(f"Unsupported global_orient dimension: {orient_dim}. Expected 3 or 6.")
    
    # Inverse rotation: kp3d_front = R^T @ kp3d
    R_global_T = R_global.transpose(-2, -1)  # [B*T, 3, 3]
    
    # Reshape for batch matrix multiplication
    kp3d_flat = kp3d.reshape(B * T, N, 3)  # [B*T, N, 3]
    
    # Apply inverse rotation to each keypoint
    kp3d_front_flat = torch.bmm(R_global_T, kp3d_flat.transpose(-2, -1))  # [B*T, 3, N]
    kp3d_front_flat = kp3d_front_flat.transpose(-2, -1)  # [B*T, N, 3]
    
    # Reshape back to original shape
    kp3d_front = kp3d_front_flat.reshape(B, T, N, 3)
    
    if len(original_shape) == 3:  # Flatten back to [B, T, N*3]
        kp3d_front = kp3d_front.reshape(B, T, N * 3)
    
    return kp3d_front

class Network(nn.Module):
    def __init__(self, 
                 smpl,
                 pose_dr=0.1,
                 d_embed=128,
                 n_layers=3,
                 d_feat=2048,
                 rnn_type='LSTM',
                 **kwargs
                 ):
        super().__init__()
        
        n_joints = _C.KEYPOINTS.NUM_JOINTS
        self.smpl = smpl
        in_dim = n_joints * 2 + 3
        view_dim = d_embed
        d_context = view_dim + n_joints * 3
        constrast_dim = 128
        
        self.mask_embedding = nn.Parameter(torch.zeros(1, 1, n_joints, 2))   
        
        # 替换视角编码器
        self.view_encoder = MinimalViewEncoder(joint_dim=3, d_embed=d_embed)
        self.dynamic_projection = DynamicProjection(d_model= view_dim)
        #self.view_encoder = ViewEncoder(in_dim=3, d_embed=512)
        
        # 增强运动编码器
        self.motion_encoder = MotionEncoder(in_dim=in_dim, 
                                            d_embed=d_embed,
                                            pose_dr=pose_dr,
                                            rnn_type=rnn_type,
                                            n_layers=n_layers,
                                            n_joints=n_joints)
        
        self.trajectory_decoder = TrajectoryDecoder(d_embed=d_context,
                                                    rnn_type=rnn_type,
                                                    n_layers=n_layers)
        self.gated_fusion = GatedFusion(view_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(138, constrast_dim),
            nn.ReLU(),
            nn.Linear(constrast_dim, constrast_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(view_dim, constrast_dim),
            nn.ReLU(),
            nn.Linear(constrast_dim, constrast_dim)
        )
        
        # CLIP特征投影层和融合层（新增）
        if True:
            self.clip_proj = nn.Linear(d_feat, view_dim)  # 将CLIP特征投影到d_embed维度

            self.clip_gated_fusion = GatedFusion(view_dim)       # 门控融合CLIP特征
            # Module 3. Feature Integrator
            self.integrator = Integrator(in_channel=d_feat + d_context, 
                                        out_channel=d_context)

            # Module 4. Motion Decoder
        # Module 5. Motion Decoder - 从motion特征预测body pose（不包括root）
        self.motion_decoder = MotionDecoder(
            d_embed=d_embed + n_joints * 3,
            rnn_type=rnn_type,
            n_layers=n_layers
        )
        
        # View Decoder - 只预测global_orient和cam
        self.view_decoder = ViewDecoder(
            d_embed=d_embed + n_joints * 3,
            rnn_type=rnn_type,
            n_layers=n_layers
        )

        # Module 5. Trajectory Refiner
        self.trajectory_refiner = TrajectoryRefiner(d_embed=d_context,
                                                    d_hidden=d_embed,
                                                    rnn_type=rnn_type,
                                                    n_layers=2)
        

    def orthogonal_loss(self, motion_context, view_feat):
        inner_product = torch.sum(motion_context * view_feat, dim=-1)
        ortho_loss = torch.mean(inner_product ** 2)
        return ortho_loss
    
    def compute_global_feet(self, root_world, trans):
        # # Compute world-coordinate motion
        cam_R, cam_T = compute_camera_motion(self.output, self.pred_pose[:, :, :6], root_world, trans, self.pred_cam)
        feet_cam = self.output.feet.reshape(self.b, self.f, -1, 3) + self.output.full_cam.reshape(self.b, self.f, 1, 3)
        feet_world = (cam_R.mT @ (feet_cam - cam_T.unsqueeze(-2)).mT).mT
        
        return feet_world, cam_R
    
    
    def forward_smpl(self, **kwargs):
        self.output = self.smpl(self.pred_pose, 
                                self.pred_shape,
                                cam=self.pred_cam,
                                return_full_pose=not self.training,
                                **kwargs,
                                )
        
        # Feet location in global coordinate
        root_world, trans = rollout_global_motion(self.pred_root, self.pred_vel)
        feet_world, cam_R = self.compute_global_feet(root_world, trans)
        
        # Return output
        output = {'feet': feet_world,
                  'contact': self.pred_contact,
                  'pose': self.pred_pose, 
                  'betas': self.pred_shape, 
                  'cam': self.pred_cam,
                  'poses_root_cam': self.output.global_orient,
                  'poses_root_r6d': self.pred_root,
                  'vel_root': self.pred_vel,
                  'pose_root': self.pred_root,
                  'verts_cam': self.output.vertices,
                  'joints2d': self.output.full_joints2d,
                  'joints3d': self.output.joints,}
        
        if self.training:
            output.update({
                'kp3d': self.output.joints,
                'kp3d_nn': self.pred_kp3d,
                'full_kp2d': self.output.full_joints2d,
                'weak_kp2d': self.output.weak_joints2d,
                'R': cam_R,
                'poses_body': self.output.body_pose,
            })
        else:
            output.update({
                'kp3d': self.output.joints,
                'kp3d_nn': self.pred_kp3d,
                'full_kp2d': self.output.full_joints2d,
                'weak_kp2d': self.output.weak_joints2d,
                'R': cam_R,
                'poses_root_r6d': self.pred_root,
                'trans_cam': self.output.full_cam,
                'poses_body': self.output.body_pose})
        
        return output     

    def preprocess(self, x, mask):
        self.b, self.f = x.shape[:2]
        
        # Treat masked keypoints
        mask_embedding = mask.unsqueeze(-1) * self.mask_embedding
        _mask = mask.unsqueeze(-1).repeat(1, 1, 1, 2).reshape(self.b, self.f, -1)
        _mask = torch.cat((_mask, torch.zeros_like(_mask[..., :3])), dim=-1)
        _mask_embedding = mask_embedding.reshape(self.b, self.f, -1)
        _mask_embedding = torch.cat((_mask_embedding, torch.zeros_like(_mask_embedding[..., :3])), dim=-1)
        x[_mask] = 0.0
        x = x + _mask_embedding
        return x
    
    
    def rollout(self, output, pred_root, pred_vel, return_y_up):
        root_world, trans_world = rollout_global_motion(pred_root, pred_vel)
        
        if return_y_up:
            yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(root_world.device)
            root_world = yup2ydown.mT @ root_world
            trans_world = (yup2ydown.mT @ trans_world.unsqueeze(-1)).squeeze(-1)
            
        output.update({
            'poses_root_world': root_world,
            'trans_world': trans_world,
        })
        
        return output

        
    def refine_trajectory(self, output, cam_angvel, return_y_up, **kwargs):
        
        # --------- Refine trajectory --------- #
        update_vel = reset_root_velocity(self.smpl, self.output, self.pred_contact, self.pred_root, self.pred_vel, thr=0.5)
        output = self.trajectory_refiner(self.old_motion_context, update_vel, output, cam_angvel, return_y_up=return_y_up)
        # --------- #
        
        # Do rollout
        output = self.rollout(output, output['poses_root_r6d_refined'], output['vel_root_refined'], return_y_up)

        # ---------  Compute refined feet --------- #
        if self.training:
            feet_world, cam_R = self.compute_global_feet(output['poses_root_world'], output['trans_world'])
            output.update({'feet_refined': feet_world})

        return output
        

      
    def contrastive_loss(self, pose_feat, motion_feat, temperature=0.1):
        """
        标准的对比学习 (InfoNCE) 损失
        - pose_feat: [N, D]
        - motion_feat: [N, D]
        """
        # squeeze & 确保维度正确
        pose_feat = pose_feat.squeeze()
        motion_feat = motion_feat.squeeze()
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if motion_feat.dim() == 1:
            motion_feat = motion_feat.unsqueeze(0)

        # 保证 batch 对齐
        N = min(pose_feat.size(0), motion_feat.size(0))
        pose_feat = pose_feat[:N]
        motion_feat = motion_feat[:N]

        # L2 normalize
        pose_feat = F.normalize(pose_feat, p=2, dim=1)
        motion_feat = F.normalize(motion_feat, p=2, dim=1)

        # 拼接两个模态: [2N, D]
        features = torch.cat([pose_feat, motion_feat], dim=0)  # [2N, D]

        # 相似度矩阵 [2N, 2N]
        sim_matrix = torch.matmul(features, features.t()) / temperature

        # 避免 self-contrast (mask 对角线)
        mask = torch.eye(2*N, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 构造 labels
        labels = torch.arange(N, device=sim_matrix.device)
        labels = torch.cat([labels + N, labels], dim=0)  # [2N]，正样本索引

        # cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def evaluate_view_feature(self, gt_joints):
        """
        评估 view_feature 的质量

        Args:
            gt_joints (torch.Tensor): ground truth 的 3D 关键点，
                                      shape 应与 pred_kp3d 一致, e.g., [B, F, num_joints, 3]

        Returns:
            dict: 包含各项评估指标的字典
        """
        # 使用 torch.no_grad()，因为这只是评估，不需要计算梯度
        with torch.no_grad():
            # 1. 获取预测的 view_feature
            # 假设 self.pred_view_feat 已经在 forward pass 中被计算并保存
            pred_feat = self.view_feat
            
            # 2. 从 ground truth 关键点生成 "真值" view_feature
            # 确保 gt_joints 的 shape 和类型与 view_encoder 的输入要求一致
            gt_feat = self.view_encoder(gt_joints.unsqueeze(0))  # 假设 gt_joints shape 是 [B, F, num_joints, 3]

            # 3. Reshape 特征以便计算指标
            # 原始 shape: [B, F, D], D 是特征维度
            # Reshape 成: [B*F, D]
            b, f, d = pred_feat.shape
            pred_feat_reshaped = pred_feat.reshape(-1, d)
            gt_feat_reshaped = gt_feat.reshape(-1, d)

            # 4. 计算评估指标
            
            # a) 余弦相似度
            # F.cosine_similarity 默认计算最后一维的相似度
            # 我们计算所有帧的平均相似度
            cos_sim = F.cosine_similarity(pred_feat_reshaped, gt_feat_reshaped, dim=1).mean()

            # b) 均方误差 (MSE)
            mse = F.mse_loss(pred_feat_reshaped, gt_feat_reshaped)

            # c) 可释方差分数
            # 计算 gt_feat 的方差
            var_gt = torch.var(gt_feat_reshaped, dim=0, unbiased=False)
            # 计算残差的方差
            var_err = torch.var(gt_feat_reshaped - pred_feat_reshaped, dim=0, unbiased=False)
            # 避免除以零
            explained_variance_ratio = 1 - (var_err / (var_gt + 1e-9))
            explained_variance_score = torch.mean(explained_variance_ratio)

        # 返回一个包含所有指标的字典
        return {
            'view_feat_cosine_similarity': cos_sim.item(),
            'view_feat_mse': mse.item(),
            'view_feat_explained_variance': explained_variance_score.item()
        }
    
    def forward(self, x, gt, inits, img_features=None, atten=True, mask=None, init_root=None, cam_angvel=None,
                cam_intrinsics=None, bbox=None, res=None, return_y_up=False, refine_traj=True, **kwargs):

        x = self.preprocess(x, mask)
        init_kp, init_smpl = inits

        # Stage 1. Encode motion - 用于body pose预测
        pred_kp3d, motion_context = self.motion_encoder(x, init_kp)
        motion_context_with_kp3d = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        self.old_motion_context = motion_context_with_kp3d.detach()
        if img_features is not None and self.integrator is not None and random.random() > 0.5:
            motion_context_with_kp3d = self.integrator(motion_context_with_kp3d, img_features)
        # Stage 2. Decode global trajectory
        pred_root, pred_vel = self.trajectory_decoder(motion_context_with_kp3d, init_root, cam_angvel)
        # Stage 5: 从view_context解码global_orient和cam

        if img_features is not None and self.integrator is not None and random.random() > 0.5:
            clip_feat = self.clip_proj(img_features)                     # 投影到d_embed维度
            motion_context = self.clip_gated_fusion(motion_context, clip_feat)  # 门控融合
        # 使用极简编码器：只提取髋部和肩部的基本几何特征
        view_feat = self.view_encoder(pred_kp3d)
        
        # view_feat: [B, T, d_embed] - 纯视角特征（基于髋部和肩部向量）
        motion_context = self.gated_fusion(motion_context, view_feat)
        
        motion_context_with_kp3d = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        pred_global_orient, pred_cam = self.view_decoder(
            motion_context_with_kp3d,  # 使用view特征
            init_smpl
        )
        #pred_kp3d_front = transform_kp3d_to_front_view(pred_kp3d, pred_global_orient)
    
        motion_context = self.dynamic_projection(motion_context, view_feat)
        # 计算正交性损失（鼓励两个特征空间解耦）
        ortho_loss = self.orthogonal_loss(motion_context, view_feat)


        # 保存用于refiner
        motion_context_with_kp3d = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        self.motion_context_with_kp3d = motion_context_with_kp3d
        self.motion_context = motion_context
        self.view_feat = view_feat

        # Stage 5. Decode body pose from MOTION context only
        # motion_decoder现在只预测body pose（不包括root）
        pred_body_pose, pred_shape, pred_contact = self.motion_decoder(
            motion_context_with_kp3d,  # 使用motion特征
            init_smpl
        )
        
        # 组合完整的pose: [B, T, 144] = [6 + 138]
        pred_pose = torch.cat([pred_global_orient, pred_body_pose], dim=-1)
        # --------- #

        # --------- Register predictions --------- #
        self.pred_kp3d = pred_kp3d
        self.pred_root = pred_root             
        self.global_orient = pred_global_orient 
        self.pred_vel = pred_vel
        self.pred_pose = pred_pose
        self.pred_shape = pred_shape
        self.pred_cam = pred_cam
        self.pred_contact = pred_contact
        # --------- #

        # --------- Build SMPL --------- #
        output = self.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)
        # --------- #

        # --------- Refine trajectory --------- #
        if refine_traj:
            output = self.refine_trajectory(output, cam_angvel, return_y_up)
        else:
            output = self.rollout(output, self.pred_root, self.pred_vel, return_y_up)
        # --------- #

        if self.training:
            b, f = gt['pose'].shape[:2]
            gt_pose_flat = gt['pose'].reshape(b, f, -1)
            
            epsilon = 1e-9
            valid_mask = torch.all(torch.abs(gt_pose_flat) > epsilon, dim=-1)
            
            # 原有的pose-motion对比学习
            gt_pose_filtered = gt_pose_flat[valid_mask]
            motion_context_reshaped = motion_context.reshape(b, f, -1)
            motion_context_filtered = motion_context_reshaped[valid_mask]
            
            if gt_pose_filtered.shape[0] > 0:
                # 3. 计算特征
                pose_feat = self.pose_proj(gt_pose_filtered[:, 6:])
                motion_feat = self.motion_proj(motion_context_filtered)

                # 4. 计算对比损失
                # 因为 pose_feat 和 motion_feat 都已经是 [N, feature_dim] 的形状，无需再 reshape
                contrastive_loss = self.contrastive_loss(
                    pose_feat,
                    motion_feat
                )
            else:
                contrastive_loss = torch.tensor(0.0, device=output['pose'].device)

            output['ortho_loss'] = ortho_loss
            output['contrastive_loss'] = contrastive_loss
        
        return output

    def stream_inference(self, x, inits, img_features=None, mask=None, init_root=None, cam_angvel=None,
                        cam_intrinsics=None, bbox=None, res=None, return_y_up=False, window_size=10, refine_traj=True,
                        hidden_states=None, prev_context=None, prev_kp3d=None, prev_output=None, **kwargs):
        """
        Streaming inference method that processes one frame at a time, maintaining RNN hidden states.
        Follows the decoupled architecture where:
        - motion_decoder: predicts body pose only (not including global_orient)
        - view_decoder: predicts global_orient and cam

        Args:
            x (tensor): Input keypoints for the current frame [B, 1, n_joints, 2]
            inits (tuple): Initial keypoints and SMPL parameters (init_kp, init_smpl)
            img_features (tensor, optional): Image features for the current frame
            mask (tensor, optional): Mask for missing keypoints
            init_root (tensor, optional): Initial root orientation
            cam_angvel (tensor, optional): Camera angular velocity
            cam_intrinsics (tensor, optional): Camera intrinsics
            bbox (tensor, optional): Bounding box
            res (tuple, optional): Resolution
            return_y_up (bool, optional): Whether to return y-up coordinate system
            refine_traj (bool, optional): Whether to refine trajectory
            hidden_states (dict, optional): Previous hidden states for RNNs
            prev_context (tensor, optional): Previous motion context
            prev_kp3d (tensor, optional): Previous 3D keypoints
            prev_output (dict, optional): Previous output for continuity

        Returns:
            dict: Output containing SMPL parameters, joint positions, etc.
            dict: Updated hidden states for next iteration
            tensor: Current motion context for buffer
            tensor: Current 3D keypoints for buffer
        """
        self.b, self.f = x.shape[:2]
        # Initialize hidden states if None
        if hidden_states is None:
            hidden_states = {
                'motion_encoder': None,
                'trajectory_decoder': None,
                'motion_decoder': None,
                'view_decoder': None,
                'trajectory_refiner': None
            }

        # Prepare window-based input for non-RNN networks
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            # Create combined context with history for window-based processing
            recent_context = prev_context[:, -window_size+1:] if prev_context.shape[1] >= window_size else prev_context
            recent_kp3d = prev_kp3d[:, -window_size+1:] if prev_kp3d.shape[1] >= window_size else prev_kp3d

            # Window-based input (for non-RNN networks)
            x_window = x[:, -window_size:] if x.shape[1] >= window_size else x
            mask_window = mask[:, -window_size:] if mask.shape[1] >= window_size else mask
        else:
            recent_context = None
            recent_kp3d = None
            x_window = x
            mask_window = mask

        self.b, self.f = x.shape[:2]  # Should be [B, 1, n_joints, 2]

        # Preprocess input - use only current frame for RNN
        x_current = self.preprocess(x[:, -1:], mask[:, -1:] if mask is not None else None)  # Only current frame
        x_window_processed = self.preprocess(x_window, mask_window)  # Window for other networks

        init_kp, init_smpl = inits

        # Get previous frame's 3D keypoints for RNN input (only 3D part; init_kp is 3D+2D concat)
        n_joints = _C.KEYPOINTS.NUM_JOINTS
        if prev_output is not None and 'kp3d_nn' in prev_output:
            prev_kp3d_single = prev_output['kp3d_nn'][:, -1:].clone()  # [B, 1, n_joints*3]
        else:
            # init_kp from dataset is [B, n_joints*3 + in_dim]; use only first n_joints*3
            prev_kp3d_single = init_kp[..., : n_joints * 3].reshape(self.b, 1, -1)

        # Stage 1. Motion Encoder - RNN using only previous frame
        pred_kp3d_current, motion_context_current, hidden_states['motion_encoder'] = self.motion_encoder.forward_step(
            x_window_processed, prev_kp3d_single, hidden_states['motion_encoder']
        )

        # Prepare motion_context_with_kp3d for trajectory decoder (before view fusion)
        motion_context_with_kp3d_for_traj = torch.cat(
            (motion_context_current, pred_kp3d_current.reshape(self.b, 1, -1)), dim=-1
        )

        # Store old motion context for trajectory refinement
        self.old_motion_context = motion_context_with_kp3d_for_traj.detach()

        # Prepare full sequences for trajectory decoder
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            full_motion_context_with_kp3d_for_traj = torch.cat([
                recent_context, motion_context_with_kp3d_for_traj
            ], dim=1)
        else:
            full_motion_context_with_kp3d_for_traj = motion_context_with_kp3d_for_traj

        # Get previous root for RNN
        if prev_output is not None and 'poses_root_r6d' in prev_output:
            prev_root = prev_output['poses_root_r6d'][:, -1:].clone()  # Only last frame
        else:
            prev_root = init_root if init_root is not None else torch.zeros(self.b, 1, 6, device=x.device)

        # Stage 2. Trajectory Decoder - predict global trajectory
        pred_root, pred_vel, hidden_states['trajectory_decoder'] = self.trajectory_decoder.forward_step(
            full_motion_context_with_kp3d_for_traj, prev_root, cam_angvel, hidden_states['trajectory_decoder']
        )

        # Stage 3. Process CLIP features and apply view encoding
        # First apply CLIP feature fusion
        if img_features is not None:
            clip_feat = self.clip_proj(img_features[:, [-1]])
            motion_context_current = self.clip_gated_fusion(motion_context_current, clip_feat)

        # Apply view encoding using predicted 3D keypoints
        view_feat = self.view_encoder(pred_kp3d_current)
        view_feat = view_feat.reshape(self.b, 1, -1).expand_as(motion_context_current)

        # Apply gated fusion with view features
        motion_context_fused = self.gated_fusion(motion_context_current, view_feat)

        # View decoder expects context dim = d_embed + n_joints*3; build to match (same as full forward)
        expected_ctx_dim = self.view_decoder.regressor.rnn.input_size - 6  # init is 6-dim
        d_motion_for_view = expected_ctx_dim - n_joints * 3
        motion_for_view = motion_context_fused[..., :d_motion_for_view]
        motion_context_with_kp3d_for_view = torch.cat(
            (motion_for_view, pred_kp3d_current.reshape(self.b, 1, -1)), dim=-1
        )
        if motion_context_with_kp3d_for_view.shape[-1] < expected_ctx_dim:
            motion_context_with_kp3d_for_view = F.pad(
                motion_context_with_kp3d_for_view, (0, expected_ctx_dim - motion_context_with_kp3d_for_view.shape[-1])
            )

        # Prepare full sequences for view decoder
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            # recent_context is from motion path (d_embed+51); ensure same dim for view
            recent_view_ctx = recent_context[..., :expected_ctx_dim] if recent_context.shape[-1] > expected_ctx_dim else recent_context
            if recent_view_ctx.shape[-1] < expected_ctx_dim:
                recent_view_ctx = F.pad(recent_view_ctx, (0, expected_ctx_dim - recent_view_ctx.shape[-1]))
            full_motion_context_with_kp3d_for_view = torch.cat([
                recent_view_ctx, motion_context_with_kp3d_for_view
            ], dim=1)
        else:
            full_motion_context_with_kp3d_for_view = motion_context_with_kp3d_for_view

        # Get previous SMPL params for decoders
        if prev_output is not None and 'pose' in prev_output:
            prev_smpl = prev_output['pose'][:, -1:].clone()  # Only last frame
        else:
            prev_smpl = init_smpl

        # Stage 4. View Decoder - predict global_orient and cam from view-fused context
        pred_global_orient, pred_cam, hidden_states['view_decoder'] = self.view_decoder.forward_step(
            full_motion_context_with_kp3d_for_view, prev_smpl, hidden_states['view_decoder']
        )

        # Stage 5. Apply dynamic projection to decouple motion from view
        motion_context_decoupled = self.dynamic_projection(motion_context_fused, view_feat)

        # Prepare motion_context_with_kp3d for motion decoder (after dynamic projection)
        motion_context_with_kp3d_for_motion = torch.cat(
            (motion_context_decoupled, pred_kp3d_current.reshape(self.b, 1, -1)), dim=-1
        )

        # Prepare full sequences for motion decoder
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            full_motion_context_with_kp3d_for_motion = torch.cat([
                recent_context, motion_context_with_kp3d_for_motion
            ], dim=1)
        else:
            full_motion_context_with_kp3d_for_motion = motion_context_with_kp3d_for_motion

        # Stage 6. Motion Decoder - predict body pose only (not including global_orient)
        pred_body_pose, pred_shape, pred_contact, hidden_states['motion_decoder'] = self.motion_decoder.forward_step(
            full_motion_context_with_kp3d_for_motion, prev_smpl, hidden_states['motion_decoder']
        )

        # Combine global_orient and body_pose to form complete pose
        pred_pose = torch.cat([pred_global_orient, pred_body_pose], dim=-1)

        # Update full motion context for return (use the decoupled version)
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            updated_full_motion_context = torch.cat([
                recent_context, motion_context_with_kp3d_for_motion
            ], dim=1)
            full_kp3d = torch.cat((recent_kp3d, pred_kp3d_current), dim=1)
        else:
            updated_full_motion_context = motion_context_with_kp3d_for_motion
            full_kp3d = pred_kp3d_current

        # Register predictions
        self.pred_kp3d = full_kp3d
        if prev_context is not None and prev_kp3d is not None and window_size > 1:
            self.pred_pose = torch.cat([self.pred_pose[:, -window_size+1:], pred_pose], dim=1)
            self.pred_shape = torch.cat([self.pred_shape[:, -window_size+1:], pred_shape], dim=1)
            self.pred_cam = torch.cat([self.pred_cam[:, -window_size+1:], pred_cam], dim=1)
            self.pred_contact = torch.cat([self.pred_contact[:, -window_size+1:], pred_contact], dim=1)
            self.pred_vel = torch.cat([self.pred_vel[:, -window_size+1:], pred_vel], dim=1)
            self.pred_root = torch.cat([self.pred_root[:, -window_size:], pred_root], dim=1)
        else:
            self.pred_kp3d = pred_kp3d_current
            self.pred_pose = pred_pose
            self.pred_shape = pred_shape
            self.pred_cam = pred_cam
            self.pred_contact = pred_contact
            self.pred_root = torch.cat((prev_root, pred_root), dim=1)
            self.pred_vel = pred_vel

        # Build SMPL (slice bbox to current window so full_cam shape matches pred_cam [B, T, 3])
        smpl_kwargs = dict(kwargs)
        smpl_kwargs['cam_intrinsics'] = cam_intrinsics
        smpl_kwargs['res'] = res
        if bbox is not None:
            smpl_kwargs['bbox'] = bbox[:, -self.f:] if (bbox.dim() == 3 and bbox.shape[1] > self.f) else bbox
        else:
            smpl_kwargs['bbox'] = bbox
        output = self.forward_smpl(**smpl_kwargs)

        output = self.rollout(output, self.pred_root, self.pred_vel, return_y_up)

        # Return output, updated hidden states, and current context for history buffer
        return output, hidden_states, updated_full_motion_context, full_kp3d