#movid.py
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
from torch import nn
import numpy as np
from lib.utils.imutils import avg_preds
from configs import constants as _C
from lib.models.layers import (MotionEncoder, MotionDecoder, TrajectoryDecoder, TrajectoryRefiner, Integrator, GatedFusion, EnhancedViewEncoder,LightweightMLP, CrossAttentionFusion,CLIPGatedFusion, ImprovedDynamicProjection,DynamicProjection,IMUProjection,ViewDecoder,MinimalViewEncoder,
                               rollout_global_motion, reset_root_velocity, compute_camera_motion)
from lib.utils.transforms import axis_angle_to_matrix
from lib.utils.kp_utils import root_centering
import math
from lib.utils import transforms
from lib.models.optimized_stream import StreamInference
import torch.nn.functional as F
import random
def adaptive_contrastive_loss(anchor, positive, negatives):
    """
    anchor: [B,T,d] motion features
    positive: [B,T,d] pose features from the same sample (positive sample)
    negatives: [B,T,d] inferred negative samples
    """
    # Normalize features
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    # Compute similarity
    pos_sim = (anchor * positive).sum(dim=-1).mean(dim=-1)  # [B]
    neg_sim = (anchor * negatives).sum(dim=-1).mean(dim=-1)  # [B]
    
    # Dynamic temperature coefficient (based on sample difficulty)
    with torch.no_grad():
        hardness = (pos_sim - neg_sim).abs().mean()
        temperature = torch.clamp(0.1 + hardness * 0.5, min=0.01, max=0.5)
    
    # Compute loss
    loss = -torch.log(
        torch.exp(pos_sim / temperature) / 
        (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature))
    ).mean()
    
    return loss, temperature.item()  # Return the temperature coefficient for monitoring

def safe_normalize(x, dim=1, eps=1e-6):
    """
    Safe normalization function to avoid division by zero
    
    Args:
    - x: input tensor
    - dim: normalization dimension
    - eps: small value to avoid division by zero
    
    Returns:
    - normalized tensor
    """
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)

def debug_contrastive_loss(pose_feat, motion_feat, temperature=0.1):
    """
    Contrastive loss function with debug information
    
    Args:
    - pose_feat: pose features
    - motion_feat: motion features
    - temperature: temperature parameter
    
    Returns:
    - contrastive loss
    """
    # Safe normalization
    pose_feat = safe_normalize(pose_feat, dim=1)
    motion_feat = safe_normalize(motion_feat, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(pose_feat, motion_feat.t()) / temperature
    
    labels = torch.arange(pose_feat.size(0)).to(pose_feat.device)
    
    # Safe cross-entropy computation
    try:
        loss_1 = F.cross_entropy(similarity_matrix, labels)
        loss_2 = F.cross_entropy(similarity_matrix.t(), labels)
        loss = (loss_1 + loss_2) / 2
    except Exception as e:
        print("Loss Calculation Error:", e)
        loss = torch.tensor(0.0, device=pose_feat.device)
    
    return loss

def safe_contrastive_loss(pose_feat, motion_feat, temperature=0.1, margin=None):
    """
    More robust contrastive loss function
    
    Args:
    - pose_feat: pose features
    - motion_feat: motion features
    - temperature: temperature parameter
    - margin: compatibility parameter, not actually used
    
    Returns:
    - contrastive loss
    """
    # Remove dimensions of size 1
    pose_feat = pose_feat.squeeze()
    motion_feat = motion_feat.squeeze()
    
    # Ensure the features are 2D
    if pose_feat.dim() == 1:
        pose_feat = pose_feat.unsqueeze(0)
    if motion_feat.dim() == 1:
        motion_feat = motion_feat.unsqueeze(0)
    
    # Ensure feature dimensions match
    min_len = min(pose_feat.size(0), motion_feat.size(0))
    pose_feat = pose_feat[:min_len]
    motion_feat = motion_feat[:min_len]
    
    # Safe normalization
    pose_feat = F.normalize(pose_feat, p=2, dim=1)
    motion_feat = F.normalize(motion_feat, p=2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.matmul(pose_feat, motion_feat.t()) / temperature
    
    # Create labels
    labels = torch.arange(pose_feat.size(0)).to(pose_feat.device)
    
    # Compute loss
    try:
        loss_1 = F.cross_entropy(similarity_matrix, labels)
        loss_2 = F.cross_entropy(similarity_matrix.t(), labels)
        loss = (loss_1 + loss_2) / 2
    except Exception as e:
        print(f"Loss calculation error: {e}")
        loss = torch.tensor(0.0, device=pose_feat.device)
    
    return loss


def enhanced_orthogonal_loss_1(motion, view, lambda_=0.1, eps=1e-8):
    """
    Args:
        motion: [B,T,d] motion features
        view: [B,T,d] view features
        lambda_: nonlinear term weight
        eps: numerical stability term
    """
    # 1. Normalize features
    motion_norm = F.normalize(motion, p=2, dim=-1)  # [B,T,d]
    view_norm = F.normalize(view, p=2, dim=-1)      # [B,T,d]
    
    # 2. Linear orthogonality term
    linear_term = torch.mean((motion_norm * view_norm).sum(dim=-1)**2)
    
    # 3. Nonlinear orthogonality term (optional)
    delta = 1e-3
    view_perturb = view_norm + delta * torch.randn_like(view_norm)
    perturbed_motion = motion_norm + (view_perturb - view_norm)
    jacobian = (perturbed_motion - motion_norm) / delta  # finite-difference approximation
    nonlin_term = torch.norm(jacobian.transpose(-1,-2) @ jacobian, p='fro')**2
    
    return linear_term + lambda_ * nonlin_term

def enhanced_orthogonal_loss(motion, view, lambda_=0.01):
    # Normalize
    motion = F.normalize(motion, dim=-1)
    view = F.normalize(view, dim=-1)
    
    # Linear orthogonality term
    cos_sim = (motion * view).sum(dim=-1)  # [B,T]
    linear_term = torch.mean(cos_sim**2)
    
    # Nonlinear regularization term (avoids explicit Jacobian computation)
    nonlin_term = torch.mean((motion - view.detach()).norm(dim=-1)**2)
    
    return linear_term + lambda_ * nonlin_term

def orthogonal_loss(motion_context, view_feat):
    inner_product = torch.sum(motion_context * view_feat, dim=-1)
    ortho_loss = torch.mean(inner_product ** 2)
    return ortho_loss

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
        
        # Replace the view encoder
        self.view_encoder = MinimalViewEncoder(joint_dim=3, d_embed=d_embed)
        self.dynamic_projection = DynamicProjection(d_model= view_dim)
        #self.view_encoder = ViewEncoder(in_dim=3, d_embed=512)
        
        # Enhance the motion encoder
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
        self.bone_proj = nn.Sequential(
            nn.Linear(33, constrast_dim),
            nn.ReLU(),
            nn.Linear(constrast_dim, constrast_dim)
        )
        self.motion_proj = nn.Sequential(
            nn.Linear(view_dim, constrast_dim),
            nn.ReLU(),
            nn.Linear(constrast_dim, constrast_dim)
        )
        self.lightweight_mlp = LightweightMLP(d_context, output_dim=42)
        # CLIP feature projection and fusion layers (new)
        self.clip_proj = nn.Linear(d_feat, view_dim)  # Project CLIP features to the d_embed dimension

        self.clip_gated_fusion = GatedFusion(view_dim)       # Gate and fuse CLIP features
        # Module 3. Feature Integrator
        self.integrator = Integrator(in_channel=d_feat + d_context, 
                                     out_channel=d_context)
        # Keep the module for checkpoint compatibility, but disable it in all runtime paths.
        self.use_integrator = False

        # Module 4. Motion Decoder - Predict body pose from motion features (excluding root)
        self.motion_decoder = MotionDecoder(
            d_embed=d_embed + n_joints * 3,
            rnn_type=rnn_type,
            n_layers=n_layers
        )
        
        # View Decoder - Predict only global_orient and cam
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
    
    def compute_global_feet(self, root_world, trans):
        # # Compute world-coordinate motion
        cam_R, cam_T = compute_camera_motion(self.output, self.pred_pose[:, :, :6], root_world, trans, self.pred_cam)
        feet_cam = self.output.feet.reshape(self.b, self.f, -1, 3) + self.output.full_cam.reshape(self.b, self.f, 1, 3)
        feet_world = (cam_R.mT @ (feet_cam - cam_T.unsqueeze(-2)).mT).mT
        
        return feet_world, cam_R
    
    def debug_contrastive_loss(self, pose_feat, motion_feat, temperature=0.1):
        """
        Contrastive loss function with debug information
        
        Args:
        - pose_feat: pose features
        - motion_feat: motion features
        - temperature: temperature parameter
        
        Returns:
        - contrastive loss
        """
        # Safe normalization
        pose_feat = safe_normalize(pose_feat, dim=1)
        motion_feat = safe_normalize(motion_feat, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(pose_feat, motion_feat.t()) / temperature
        
        labels = torch.arange(pose_feat.size(0)).to(pose_feat.device)
        
        # Safe cross-entropy computation
        try:
            loss_1 = F.cross_entropy(similarity_matrix, labels)
            loss_2 = F.cross_entropy(similarity_matrix.t(), labels)
            loss = (loss_1 + loss_2) / 2
        except Exception as e:
            print("Loss Calculation Error:", e)
            loss = torch.tensor(0.0, device=pose_feat.device)
        
        return loss
    
    def orthogonal_loss(self, motion_context, view_feat):
        inner_product = torch.sum(motion_context * view_feat, dim=-1)
        ortho_loss = torch.mean(inner_product ** 2)
        return ortho_loss
    
    def contrastive_loss(self, pose_feat, motion_feat, temperature=0.1):
        """
        Standard contrastive learning (InfoNCE) loss
        - pose_feat: [N, D]
        - motion_feat: [N, D]
        """
        # squeeze and ensure dimensions are correct
        pose_feat = pose_feat.squeeze()
        motion_feat = motion_feat.squeeze()
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if motion_feat.dim() == 1:
            motion_feat = motion_feat.unsqueeze(0)

        # Ensure batch alignment
        N = min(pose_feat.size(0), motion_feat.size(0))
        pose_feat = pose_feat[:N]
        motion_feat = motion_feat[:N]

        # L2 normalize
        pose_feat = F.normalize(pose_feat, p=2, dim=1)
        motion_feat = F.normalize(motion_feat, p=2, dim=1)

        # Concatenate the two modalities: [2N, D]
        features = torch.cat([pose_feat, motion_feat], dim=0)  # [2N, D]

        # Similarity matrix [2N, 2N]
        sim_matrix = torch.matmul(features, features.t()) / temperature

        # Avoid self-contrast (mask the diagonal)
        mask = torch.eye(2*N, dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Build labels
        labels = torch.arange(N, device=sim_matrix.device)
        labels = torch.cat([labels + N, labels], dim=0)  # [2N], positive sample indices

        # cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def forward_smpl(self, **kwargs):
        self.output = self.smpl(self.pred_pose, 
                                self.pred_shape,
                                cam=self.pred_cam,
                                return_full_pose=not self.training,
                                **kwargs,
                                )
        
        # # Feet location in global coordinate
        # root_world, trans = rollout_global_motion(self.pred_root, self.pred_vel)
        # feet_world, cam_R = self.compute_global_feet(root_world, trans)
        
        # Return output
        output = {'contact': self.pred_contact,
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
                'poses_body': self.output.body_pose,
            })
        else:
            output.update({
                'kp3d': self.output.joints,
                'kp3d_nn': self.pred_kp3d,
                'full_kp2d': self.output.full_joints2d,
                'poses_root_r6d': self.pred_root,
                'trans_cam': self.output.full_cam,
                'poses_body': self.output.body_pose})
        
        return output     


    def safe_contrastive_loss(self, pose_feat, motion_feat, temperature=0.1, margin=None):
        """
        More robust contrastive loss function
        
        Args:
        - pose_feat: pose features
        - motion_feat: motion features
        - temperature: temperature parameter
        - margin: compatibility parameter, not actually used
        
        Returns:
        - contrastive loss
        """
        # Remove dimensions of size 1
        pose_feat = pose_feat.squeeze()
        motion_feat = motion_feat.squeeze()
        
        # Ensure the features are 2D
        if pose_feat.dim() == 1:
            pose_feat = pose_feat.unsqueeze(0)
        if motion_feat.dim() == 1:
            motion_feat = motion_feat.unsqueeze(0)
        
        # Ensure feature dimensions match
        min_len = min(pose_feat.size(0), motion_feat.size(0))
        pose_feat = pose_feat[:min_len]
        motion_feat = motion_feat[:min_len]
        
        # Safe normalization
        pose_feat = F.normalize(pose_feat, p=2, dim=1)
        motion_feat = F.normalize(motion_feat, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(pose_feat, motion_feat.t()) / temperature
        
        # Create labels
        labels = torch.arange(pose_feat.size(0)).to(pose_feat.device)
        
        # Compute loss
        try:
            loss_1 = F.cross_entropy(similarity_matrix, labels)
            loss_2 = F.cross_entropy(similarity_matrix.t(), labels)
            loss = (loss_1 + loss_2) / 2
        except Exception as e:
            print(f"Loss calculation error: {e}")
            loss = torch.tensor(0.0, device=pose_feat.device)
        
        return loss
   
    def generate_negative_views(self, motion_feat, pose_feat, threshold=0.7):
        """
        Args:
            motion_feat: [B,T,d] current motion features
            pose_feat: [B,T,d] corresponding pose features (used as pseudo-labels)
            threshold: similarity threshold; values above it are treated as potential same-action samples
        Returns:
            negative_samples: [B,T,d] negative samples (guaranteed to differ from the anchor action)
        """
        B, T, d = motion_feat.shape
        
        # 1. Compute the inter-sample similarity matrix [B, B]
        with torch.no_grad():
            # Average over the time dimension and normalize
            pose_centers = F.normalize(pose_feat.mean(dim=1), dim=-1)  # [B,d]
            sim_matrix = torch.mm(pose_centers, pose_centers.t())  # [B,B]
            
            # Exclude self-comparisons
            sim_matrix.fill_diagonal_(-1) 
        
        # 2. Automatically infer negative-sample indices
        neg_mask = sim_matrix < threshold  # [B,B]
        valid_neg_counts = neg_mask.sum(dim=1)  # [B]
        
        # 3. Dynamically choose negative samples
        negatives = []
        for i in range(B):
            if valid_neg_counts[i] > 0:
                # Select the most similar "negative sample" (hard sample)
                neg_indices = torch.where(neg_mask[i])[0]
                candidates = pose_feat[neg_indices]  # [K,T,d]
                candidate_sim = sim_matrix[i, neg_indices]  # [K]
                hardest_idx = candidate_sim.argmax()
                negatives.append(candidates[hardest_idx])
            else:
                # Generate adversarial samples when no valid negative sample exists
                noise = torch.randn_like(motion_feat[i]) * 0.1
                negatives.append(motion_feat[i] + noise)
        
        return torch.stack(negatives, dim=0)  # [B,T,d]
    

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

    def _decode_prediction_branch(self, pred_kp3d, motion_context, view_feat, init_smpl,
                                  init_root, cam_angvel, img_features=None,
                                  use_img_features=False):
        motion_context_with_kp3d = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        old_motion_context = motion_context_with_kp3d.detach()

        if self.use_integrator and use_img_features and img_features is not None and self.integrator is not None:
            motion_context_with_kp3d = self.integrator(motion_context_with_kp3d, img_features)

        pred_root, pred_vel = self.trajectory_decoder(motion_context_with_kp3d, init_root, cam_angvel)

        fused_motion_context = motion_context
        if use_img_features and img_features is not None:
            clip_feat = self.clip_proj(img_features)
            fused_motion_context = self.clip_gated_fusion(fused_motion_context, clip_feat)

        fused_motion_context = self.gated_fusion(fused_motion_context, view_feat)

        view_context_with_kp3d = torch.cat((fused_motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        pred_global_orient, pred_cam = self.view_decoder(view_context_with_kp3d, init_smpl)

        projected_motion_context = self.dynamic_projection(fused_motion_context, view_feat)
        ortho_loss = self.orthogonal_loss(projected_motion_context, view_feat)

        motion_context_with_kp3d = torch.cat(
            (projected_motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1
        )
        pred_body_pose, pred_shape, pred_contact = self.motion_decoder(
            motion_context_with_kp3d,
            init_smpl
        )

        return {
            'old_motion_context': old_motion_context,
            'pred_root': pred_root,
            'pred_vel': pred_vel,
            'pred_global_orient': pred_global_orient,
            'pred_cam': pred_cam,
            'motion_context': projected_motion_context,
            'motion_context_with_kp3d': motion_context_with_kp3d,
            'pred_body_pose': pred_body_pose,
            'pred_shape': pred_shape,
            'pred_contact': pred_contact,
            'pred_pose': torch.cat([pred_global_orient, pred_body_pose], dim=-1),
            'ortho_loss': ortho_loss,
            'pred_kp3d': pred_kp3d,
        }

    def forward(self, x, gt, inits, img_features=None, atten=True, mask=None, init_root=None, cam_angvel=None,
                cam_intrinsics=None, bbox=None, res=None, return_y_up=False, refine_traj=True, **kwargs):

        x = self.preprocess(x, mask)
        init_kp, init_smpl = inits

        # Stage 1. Encode motion - for body-pose prediction
        pred_kp3d, motion_context = self.motion_encoder(x, init_kp)
        view_feat = self.view_encoder(pred_kp3d)
        use_img_features = img_features is not None

        main_branch = self._decode_prediction_branch(
            pred_kp3d, motion_context, view_feat, init_smpl, init_root, cam_angvel,
            img_features=img_features, use_img_features=use_img_features
        )

        self.old_motion_context = main_branch['old_motion_context']
        self.motion_context_with_kp3d = main_branch['motion_context_with_kp3d']
        self.motion_context = main_branch['motion_context']
        self.view_feat = view_feat

        # --------- Register predictions --------- #
        self.pred_kp3d = main_branch['pred_kp3d']
        self.pred_root = main_branch['pred_root']
        self.global_orient = main_branch['pred_global_orient']
        self.pred_vel = main_branch['pred_vel']
        self.pred_pose = main_branch['pred_pose']
        self.pred_shape = main_branch['pred_shape']
        self.pred_cam = main_branch['pred_cam']
        self.pred_contact = main_branch['pred_contact']
        # --------- #
        
        # --------- Build SMPL --------- #
        output = self.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)

        # --------- #
        
        # --------- Refine trajectory --------- #
        # if refine_traj:
        #     output = self.refine_trajectory(output, cam_angvel, return_y_up)
        # else:
        #     output = self.rollout(output, self.pred_root, self.pred_vel, return_y_up)
        # --------- #
        if self.training:
            b, f = gt['pose'].shape[:2]
            gt_pose_flat = gt['pose'].reshape(b, f, -1)
            
            epsilon = 1e-9
            valid_mask = torch.all(torch.abs(gt_pose_flat) > epsilon, dim=-1)
            
            # Existing pose-motion contrastive learning
            gt_pose_filtered = gt_pose_flat[valid_mask]
            motion_context_reshaped = self.motion_context.reshape(b, f, -1)
            motion_context_filtered = motion_context_reshaped[valid_mask]
            
            if gt_pose_filtered.shape[0] > 0:
                # 3. Compute features
                pose_feat = self.pose_proj(gt_pose_filtered[:, 6:])
                motion_feat = self.motion_proj(motion_context_filtered)

                # 4. Compute contrastive loss
                # pose_feat and motion_feat are already shaped as [N, feature_dim], so no reshape is needed
                contrastive_loss = self.contrastive_loss(
                    pose_feat,
                    motion_feat
                )
            else:
                contrastive_loss = torch.tensor(0.0, device=output['pose'].device)

            output['ortho_loss'] = main_branch['ortho_loss']
            output['contrastive_loss'] = contrastive_loss

            if 'bone_vectors' in gt and gt['bone_vectors'] is not None:
                has_imu = gt['has_imu']
                imu_size = has_imu.sum()
                
                if imu_size > 0:
                    bone_feat = self.bone_proj(gt['bone_vectors'][has_imu].reshape(-1, 11*3))
                    # Cross-view consistency loss
                    imu_motion_feat = motion_feat[has_imu]
                    
                    # 1. IMU-to-Motion consistency
                    imu_contrast_loss = self.safe_contrastive_loss(
                        bone_feat.reshape(imu_size*self.f,-1),
                        imu_motion_feat.reshape(imu_size*self.f,-1))
                    # 2. Motion-to-IMU consistency
                    motion_contrast_loss = self.debug_contrastive_loss(
                        imu_motion_feat.reshape(imu_size*self.f,-1),
                        bone_feat.reshape(imu_size*self.f,-1))

                    
                    output['contrastive_loss'] += imu_contrast_loss + motion_contrast_loss
        
            output['contrastive_loss'] *= 0.05


        return output

    def stream_inference(self, x, inits, img_features=None, mask=None, init_root=None, cam_angvel=None,
                        cam_intrinsics=None, bbox=None, res=None, return_y_up=False, window_size=10, refine_traj=True,
                        hidden_states=None, prev_context=None, prev_kp3d=None, prev_output=None, flip_eval=False, 
                        use_optimized=True, **kwargs):
        """
        Optimized streaming inference method supporting both legacy and optimized modes
        
        Args:
            use_optimized (bool): whether to use the optimized circular-buffer mode (recommended: True)
            other parameters are the same as in the original method
        
        Returns:
            tuple: (output, hidden_states, current_context, current_kp3d, avg_output)
        """
        
        # ============ Optimized mode: use OptimizedStreamInference ============
        if use_optimized:
            # Initialize the optimized inference helper on the first call
            if not hasattr(self, '_stream_optimizer'):
                from configs import constants as _C
                # Get the actual d_embed dimension from motion_encoder
                d_embed = self.motion_encoder.embed_layer.out_features
                self._stream_optimizer = StreamInference(
                    network=self,
                    window_size=window_size,
                    device=x.device,
                    d_embed=d_embed,  # automatically get d_embed from the network
                    n_joints=_C.KEYPOINTS.NUM_JOINTS
                )
            
            # Use optimized single-frame processing
            output, hidden_states = self._stream_optimizer.process_frame(
                x=x,
                inits=inits,
                img_features=img_features,
                mask=mask,
                init_root=init_root,
                cam_angvel=cam_angvel,
                hidden_states=hidden_states,
                prev_output=prev_output,
                cam_intrinsics=cam_intrinsics,
                bbox=bbox,
                res=res,
                **kwargs
            )
            
            # Return the current frame's context and kp3d (from the optimizer buffer)
            motion_seq, kp3d_seq = self._stream_optimizer.state_manager.get_windowed_features()
            current_context = torch.cat([
                motion_seq[:, -1:], 
                kp3d_seq[:, -1:]
            ], dim=-1)  # [B, 1, 563]
            current_kp3d = kp3d_seq[:, -1:]  # [B, 1, 51]
            
            # Handle flip evaluation
            avg_output = None
            if flip_eval:
                avg_output = self._handle_flip_eval(output, cam_intrinsics, bbox, res)
            
            return output, hidden_states, current_context, current_kp3d, avg_output
        
        # ============ Legacy mode: keep the original logic (backward compatible) ============
        else:
            return self._stream_inference_legacy(
                x, inits, img_features, mask, init_root, cam_angvel,
                cam_intrinsics, bbox, res, return_y_up, window_size, refine_traj,
                hidden_states, prev_context, prev_kp3d, prev_output, flip_eval, **kwargs
            )


    def _handle_flip_eval(self, output, cam_intrinsics, bbox, res):
        """Helper method for handling flip evaluation"""
        if output['pose'].shape[0] != 2:
            return None
        
        normal_pose = output['pose'][0:1].squeeze(0)
        normal_shape = output['betas'][0:1].squeeze(0)
        flipped_pose = output['pose'][1:2].squeeze(0)
        flipped_shape = output['betas'][1:2].squeeze(0)
        
        normal_pose_reshaped = normal_pose.reshape(-1, 24, 6)
        flipped_pose_reshaped = flipped_pose.reshape(-1, 24, 6)
        
        from lib.utils.imutils import avg_preds
        avg_pose, avg_shape = avg_preds(
            normal_pose_reshaped, normal_shape, 
            flipped_pose_reshaped, flipped_shape
        )
        
        with torch.no_grad():
            avg_output = self.smpl(
                avg_pose.reshape(1, 1, 144),
                avg_shape.reshape(1, 1, 10),
                cam=self.pred_cam[0:1, -1:],
                return_full_pose=not self.training,
                cam_intrinsics=cam_intrinsics[0:1] if cam_intrinsics is not None else None, 
                bbox=bbox[0:1] if bbox is not None else None, 
                res=res[0:1] if res is not None else None,
            )
        
        return avg_output


    def _stream_inference_legacy(self, x, inits, img_features=None, mask=None, init_root=None, cam_angvel=None,
                                cam_intrinsics=None, bbox=None, res=None, return_y_up=False, window_size=10,
                                refine_traj=True, hidden_states=None, prev_context=None, prev_kp3d=None,
                                prev_output=None, flip_eval=False, **kwargs):
        """
        Edge streaming inference - strictly follows the forward pipeline without using feature/SLAM

        Forward pipeline:
        1. motion_encoder -> pred_kp3d, motion_context
        2. cat(motion_context, kp3d) -> trajectory_decoder -> pred_root, pred_vel
        3. clip_proj + clip_gated_fusion -> motion_context (CLIP fusion)
        4. view_encoder(pred_kp3d) -> view_feat
        5. gated_fusion(motion_context, view_feat) -> motion_context
        6. cat(motion_context, kp3d) -> view_decoder -> pred_global_orient, pred_cam
        7. dynamic_projection(motion_context, view_feat) -> motion_context
        8. cat(motion_context, kp3d) -> motion_decoder -> pred_body_pose, pred_shape, pred_contact
        9. cat(pred_global_orient, pred_body_pose) -> pred_pose
        """
        self.b = x.shape[0]

        # Initialize hidden states if None
        if hidden_states is None:
            hidden_states = {
                'motion_encoder': None,
                'trajectory_decoder': None,
                'motion_decoder': None,
                'view_decoder': None,
                'trajectory_refiner': None
            }
        # Ensure view_decoder key exists (backward compat)
        if 'view_decoder' not in hidden_states:
            hidden_states['view_decoder'] = None

        # --- Step 1: Preprocess current frame ---
        x_current = x[:, -1:] if x.shape[1] > 1 else x
        mask_current = mask[:, -1:] if mask is not None and mask.shape[1] > 1 else mask
        x_current_processed = self.preprocess(x_current, mask_current)
        init_kp, init_smpl = inits

        # --- Step 2: Get previous frame kp3d ---
        if prev_output is not None and 'kp3d_nn' in prev_output:
            prev_kp3d_single = prev_output['kp3d_nn'][:, -1:].clone()
        else:
            if init_kp.dim() == 2:
                prev_kp3d_single = init_kp.unsqueeze(1)
            else:
                prev_kp3d_single = init_kp[:, -1:] if init_kp.shape[1] > 0 else init_kp

        # --- Step 3: Motion Encoder ---
        pred_kp3d_current, motion_context_current, hidden_states['motion_encoder'] = \
            self.motion_encoder.forward_step(
                x_current_processed,
                prev_kp3d_single.reshape(self.b, 1, -1),
                hidden_states['motion_encoder']
            )

        # --- Step 4: cat(motion_context, kp3d) for trajectory decoder ---
        # Key point: trajectory_decoder uses the original motion_context before CLIP fusion
        motion_with_kp_original = torch.cat([
            motion_context_current,
            pred_kp3d_current.reshape(self.b, 1, -1)
        ], dim=-1)
        self.old_motion_context = motion_with_kp_original.detach()

        # --- Step 5: Get previous root ---
        if prev_output is not None and 'poses_root_r6d' in prev_output:
            prev_root = prev_output['poses_root_r6d'][:, -1:].clone()
        else:
            prev_root = init_root if init_root is not None else torch.zeros(self.b, 1, 6, device=x.device)

        # --- Step 6: Trajectory Decoder ---
        pred_root, pred_vel, hidden_states['trajectory_decoder'] = \
            self.trajectory_decoder.forward_step(
                motion_with_kp_original,
                prev_root,
                cam_angvel,
                hidden_states['trajectory_decoder']
            )

        # --- Step 7: CLIP feature fusion (after the trajectory decoder) ---
        if img_features is not None:
            clip_feat = self.clip_proj(img_features[:, -1:])
            motion_context_current = self.clip_gated_fusion(motion_context_current, clip_feat)

        # --- Step 8: View encoding ---
        view_feat = self.view_encoder(pred_kp3d_current)

        # --- Step 9: Gated fusion ---
        motion_context_fused = self.gated_fusion(motion_context_current, view_feat)

        # --- Step 10: cat(motion_context_fused, kp3d) for view_decoder ---
        motion_with_kp_for_view = torch.cat([
            motion_context_fused,
            pred_kp3d_current.reshape(self.b, 1, -1)
        ], dim=-1)

        # --- Step 11: Prepare init_smpl_view [B, 1, 24, 6] ---
        if prev_output is not None and 'pose' in prev_output:
            prev_smpl = prev_output['pose'][:, -1:].clone()
        else:
            if init_smpl.dim() == 2:
                prev_smpl = init_smpl.unsqueeze(1)
            else:
                prev_smpl = init_smpl[:, -1:] if init_smpl.shape[1] > 0 else init_smpl

        if prev_smpl.shape[-1] == 144:
            init_smpl_view = prev_smpl.reshape(self.b, 1, 24, 6)
        elif prev_smpl.shape[-1] == 6 and prev_smpl.shape[-2] == 24:
            init_smpl_view = prev_smpl
        else:
            init_smpl_view = torch.zeros(self.b, 1, 24, 6, device=x.device)
            if prev_output is not None and 'pose' in prev_output:
                prev_pose = prev_output['pose'][:, -1:].clone()
                if prev_pose.shape[-1] >= 6:
                    init_smpl_view[:, :, 0, :] = prev_pose[:, :, :6]

        # --- Step 12: View Decoder - pred_global_orient, pred_cam ---
        pred_global_orient, pred_cam, hidden_states['view_decoder'] = \
            self.view_decoder.forward_step(
                motion_with_kp_for_view,
                init_smpl_view,
                hidden_states['view_decoder']
            )

        # --- Step 13: Dynamic projection ---
        motion_context_projected = self.dynamic_projection(motion_context_fused, view_feat)

        # --- Step 14: cat(motion_context_projected, kp3d) for motion_decoder ---
        motion_with_kp_for_pose = torch.cat([
            motion_context_projected,
            pred_kp3d_current.reshape(self.b, 1, -1)
        ], dim=-1)

        # --- Step 15: Motion Decoder - pred_body_pose, pred_shape, pred_contact ---
        pred_body_pose, pred_shape, pred_contact, hidden_states['motion_decoder'] = \
            self.motion_decoder.forward_step(
                motion_with_kp_for_pose,
                init_smpl_view,
                hidden_states['motion_decoder']
            )

        # --- Step 16: Combine pose = [global_orient(6) + body_pose(138)] ---
        pred_pose = torch.cat([pred_global_orient, pred_body_pose], dim=-1)

        # --- Step 17: Register predictions ---
        self.pred_pose = pred_pose
        self.pred_shape = pred_shape
        self.pred_cam = pred_cam
        self.pred_contact = pred_contact
        self.pred_vel = pred_vel
        self.pred_root = pred_root
        self.pred_kp3d = pred_kp3d_current
        self.global_orient = pred_global_orient

        # --- Step 18: SMPL forward ---
        output = self.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)

        # Ensure continuity keys
        output.setdefault('poses_root_r6d', self.pred_root)
        output.setdefault('vel', self.pred_vel)
        output.setdefault('contact', self.pred_contact)

        # --- Step 19: Handle flip evaluation ---
        avg_output = self._handle_flip_eval(output, cam_intrinsics, bbox, res) if flip_eval else None

        return output, hidden_states, motion_with_kp_for_pose, pred_kp3d_current, avg_output


    def reset_stream_state(self):
        """Reset streaming inference state (for the start of a new sequence)"""
        if hasattr(self, '_stream_optimizer'):
            self._stream_optimizer.reset()
            print("Stream inference state reset")


    def print_stream_stats(self):
        """Print streaming inference performance statistics"""
        if hasattr(self, '_stream_optimizer'):
            self._stream_optimizer.print_stats()
