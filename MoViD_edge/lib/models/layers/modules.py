from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import numpy as np
from torch import nn
from configs import constants as _C
from .utils import rollout_global_motion
from lib.utils.transforms import axis_angle_to_matrix
import torch.nn.functional as F
import math
import tqdm


class DynamicProjection(nn.Module):
    def __init__(self, d_model, K=3):
        super().__init__()
        self.K = K
        self.base_gen = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, K*d_model)
        )
        
    def forward(self, motion, view):
        """
        Args:
            motion: [B, T, d] input motion features
            view: [B, T, d_view] view features
        """
        B, T, d = motion.shape
        
        # 1. Generate basis vectors while preserving batch and sequence dimensions
        bases = self.base_gen(view).view(B, T, self.K, d)  # [B,T,K,d]
        
        # 2. Initialize the projection result
        proj = motion.clone()  # [B,T,d]
        
        # 3. Improved Gram-Schmidt orthogonalization
        for k in range(self.K):
            v = bases[:,:,k,:]  # [B,T,d]
            
            # Compute projection coefficients while keeping dimensions aligned
            coef = (proj * v).sum(dim=-1, keepdim=True)  # [B,T,1]
            
            # Orthogonalize the projection
            proj = proj - coef * v / (v.norm(dim=-1, keepdim=True)**2 + 1e-6)
        
        return proj  # [B,T,d]


class ImprovedDynamicProjection(nn.Module):
    def __init__(self, d_model, K=4):
        super().__init__()
        self.K = K
        # Improved base generator with residual connections
        self.base_gen = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*d_model, K*d_model)
        )
        
        # Adaptive projection strength
        self.projection_gate = nn.Sequential(
            nn.Linear(d_model*2, 1),  # Changed to output a single value
            nn.Sigmoid()
        )
        
    def forward(self, motion, view):
        """
        Args:
            motion: [B,T,d] Input motion features
            view: [B,T,d_view] View features
        """
        B, T, d = motion.shape
        
        # 1. Generate orthogonal bases (keeping batch and sequence dimensions)
        bases = self.base_gen(view).view(B, T, self.K, d)  # [B,T,K,d]
        
        # 2. Compute adaptive projection strength (scalar gate)
        adaptive_gate = self.projection_gate(
            torch.cat([motion, view], dim=-1)
        )  # [B,T,1]
        
        # 3. Initialize projection result
        proj = motion.clone()  # [B,T,d]
        
        # 4. Improved iterative Gram-Schmidt with adaptive strength
        for k in range(self.K):
            v = bases[:,:,k,:]  # [B,T,d]
            v = F.normalize(v, dim=-1)  # Normalize basis vectors
            
            # Compute projection coefficients (keeping dimensions aligned)
            coef = (proj * v).sum(dim=-1, keepdim=True)  # [B,T,1]
            
            # Apply orthogonal projection with adaptive strength
            proj = proj - adaptive_gate * coef * v
            
            # Re-normalize after each step to maintain numerical stability
            if k < self.K - 1:  # Don't normalize after final step
                proj_norm = proj.norm(dim=-1, keepdim=True)
                proj = F.normalize(proj, dim=-1) * proj_norm
        
        # Apply residual connection
        proj = motion + (proj - motion) * adaptive_gate
        
        return proj  # [B,T,d]

class EnhancedViewEncoder(nn.Module):
    """Multi-scale biomechanical feature encoder"""
    def __init__(self, joint_dim=4, d_embed=512):
        super().__init__()
        # Biomechanical feature extraction
        self.bio_feat = nn.Sequential(
            nn.Linear(18, 128),  # hip/shoulder spatial relationships
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256))
        
        # Multi-scale temporal convolution
        self.tconvs = nn.ModuleList([
            nn.Conv1d(256, 256, kernel_size=3, dilation=2**i, padding=2**i) 
            for i in range(3)
        ])
        # Attention aggregation
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.final_fc = nn.Linear(256, d_embed)

    def forward(self, kp3d):
        """Input shape: [B, T, J, 3]"""
        # Biomechanical features
        hips = kp3d[:, :, [11,12]]  # hip joints
        shoulders = kp3d[:, :, [5,6]]  # shoulder joints
        
        # Compute spatial relationship features
        spatial_feat = torch.cat([
            hips.mean(2) - shoulders.mean(2),        # torso vector
            hips.std(2),                            # hip stability
            hips[:,:,0]-hips[:,:,1],        # hip range of motion
            shoulders.max(2)[0] - shoulders.min(2)[0],  # shoulder range of motion
            #shoulders[:,:,0]-shoulders[:,:,1],  # shoulder range of motion
            kp3d[:, :, [0]].expand(-1,-1,2,-1).flatten(2)  # root joint position
        ], dim=-1)  # [B, T, 16]
        
        bio_feat = self.bio_feat(spatial_feat)  # [B, T, 256]
        # return self.final_fc(bio_feat)  # [B, T, d_embed]
        # Multi-scale temporal convolution
        t_feat = bio_feat.transpose(1,2)  # [B, 256, T]
        for conv in self.tconvs:
            t_feat = F.gelu(conv(t_feat))
        t_feat = t_feat.transpose(1,2)  # [B, T, 256]
        
        # Temporal attention aggregation
        attn_out, _ = self.attn(t_feat, t_feat, t_feat)  # [B, T, 256]
        pooled = F.adaptive_avg_pool1d(attn_out.transpose(1,2), 1).squeeze(-1)

        return self.final_fc(pooled)  # [B, d_embed]


class MultiScaleMotionEncoder(nn.Module):
    """Multi-scale motion encoder"""
    def __init__(self, in_dim, d_embed=512):
        super().__init__()
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, 256, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.BatchNorm1d(256)  # use BatchNorm1d instead of LayerNorm
            ) for _ in range(3)
        ])

        
        self.spatial_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.fusion = nn.Linear(256*3, d_embed)
        
    def forward(self, x):
        """Input shape [B, T, D]"""
        B, T, D = x.shape
        x = x.transpose(1,2)  # [B, D, T]
        print("Input to branches:", x.shape)
        
        # Multi-scale feature extraction
        features = []
        for branch in self.temporal_branches:
            feat = branch(x)  # [B, 256, T//2]
            print("After conv:", feat.shape)
            feat = feat.transpose(1,2)  # [B, T//2, 256]
            
            # Spatial attention
            attn_feat, _ = self.spatial_attn(feat, feat, feat)
            features.append(F.adaptive_max_pool1d(attn_feat.transpose(1,2), 1).squeeze(-1))
        
        # Multi-scale feature fusion
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)  # [B, d_embed]
class ViewEncoder(nn.Module):
    """View feature encoder"""
    def __init__(self, in_dim=3, d_embed=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.LayerNorm(d_embed)
        )
        
    def forward(self, x):
        return self.mlp(x)

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion module (with dimensional projection)"""
    def __init__(self, d_model=512, n_head=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        
    def forward(self, motion_feat, view_feat):
        # Project to aligned dimensions
        
        # Expand view features
        #view_feat = view_feat.unsqueeze(1).expand(-1, motion_feat.size(1), -1)
        
        # Attention computation
        attn_output, _ = self.cross_attn(
            query=motion_feat,
            key=view_feat,
            value=view_feat,
            need_weights=False
        )
        motion_feat = self.norm(motion_feat + attn_output)
        return motion_feat #self.proj2(motion_feat)

class LightweightMLP(nn.Module):
    """Lightweight MLP for efficient refinement of 3D keypoints."""
    
    def __init__(self, keypoint_dim, hidden_dims=[128, 64], output_dim=72):
        super(LightweightMLP, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(keypoint_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Initialize weights for better gradient flow
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        
        # No activation on the output layer
        return self.layers[-1](x)


class GatedFusion(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_embed*2, d_embed),
            nn.Sigmoid()
        )
    
    def forward(self, motion_feat, view_feat):
        # motion_feat: [B, T, D]
        # view_feat: [B, T, D] or [B, D] (will be broadcasted)
        # Ensure view_feat has the same shape as motion_feat
        if view_feat.dim() == 2:
            # [B, D] -> [B, 1, D] -> [B, T, D]
            view_feat = view_feat.unsqueeze(1).expand_as(motion_feat)
        fusion_gate = self.gate(torch.cat([motion_feat, view_feat], -1))
        return motion_feat * fusion_gate + view_feat * (1 - fusion_gate)

class MinimalViewEncoder(nn.Module):
    """
    Minimal view encoder - extracts only basic geometric cues for body orientation
    
    Features include:
    1. hip_left - hip_right (hip-width vector, 3D)
    2. shoulder_left - shoulder_right (shoulder-width vector, 3D)
    3. Depth values (hip and shoulder z coordinates, 4D)
    
    Total: 10 basic features -> 512D embedding
    """
    def __init__(self, joint_dim=3, d_embed=512):
        super().__init__()
        
        # Minimal feature extraction: use only a lightweight MLP
        # Input: 10D (two 3D vectors + four depth values)
        self.encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, d_embed),
        )
        
    def forward(self, kp3d):
        """
        Input: [B, T, J, 3] - 3D joint positions
        Output: [B, T, d_embed] - view features
        
        Joint indices (SMPL convention):
        - 11: left_hip
        - 12: right_hip  
        - 5: left_shoulder
        - 6: right_shoulder
        """
        B, T = kp3d.shape[:2]
        
        # Extract key joints
        left_hip = kp3d[:, :, 11]        # [B, T, 3]
        right_hip = kp3d[:, :, 12]       # [B, T, 3]
        left_shoulder = kp3d[:, :, 5]    # [B, T, 3]
        right_shoulder = kp3d[:, :, 6]   # [B, T, 3]
        
        # 1. Hip-width vector (primary cue for body orientation)
        hip_vector = left_hip - right_hip  # [B, T, 3]
        
        # 2. Shoulder-width vector (secondary cue for body orientation)
        shoulder_vector = left_shoulder - right_shoulder  # [B, T, 3]
        
        # 3. Depth values (distance relative to the camera)
        left_hip_depth = left_hip[:, :, 2:3]      # [B, T, 1] - z coordinate
        right_hip_depth = right_hip[:, :, 2:3]    # [B, T, 1]
        left_shoulder_depth = left_shoulder[:, :, 2:3]   # [B, T, 1]
        right_shoulder_depth = right_shoulder[:, :, 2:3] # [B, T, 1]
        
        # Concatenate all features: [B, T, 10]
        view_features = torch.cat([
            hip_vector,              # 3D
            shoulder_vector,         # 3D
            left_hip_depth,          # 1D
            right_hip_depth,         # 1D
            left_shoulder_depth,     # 1D
            right_shoulder_depth,    # 1D
        ], dim=-1)
        
        # Encode with a simple MLP
        output = self.encoder(view_features)  # [B, T, d_embed]
        
        return output


# class GatedFusion(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(dim * 2, dim),
#             nn.ReLU(),
#             nn.Linear(dim, dim)
#         )
        
#     def forward(self, motion, view):
#         combined = torch.cat([motion, view], dim=-1)
#         decoupled = self.mlp(combined) - view
#         return decoupled



class CLIPGatedFusion(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_embed*2, d_embed),
            nn.Sigmoid()
        )
    
    def forward(self, motion_feat, video_feat):
        # motion_feat: [B, T, D]
        # view_feat: [B, D]
        fusion_gate = self.gate(torch.cat([motion_feat, video_feat], -1))
        return motion_feat * fusion_gate + video_feat * (1 - fusion_gate)

class Regressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dims, init_dim, layer='LSTM', n_layers=2, n_iters=1):
        super().__init__()
        self.n_outs = len(out_dims)

        self.rnn = getattr(nn, layer.upper())(
            in_dim + init_dim, hid_dim, n_layers, 
            bidirectional=False, batch_first=True, dropout=0.3)

        for i, out_dim in enumerate(out_dims):
            setattr(self, 'declayer%d'%i, nn.Linear(hid_dim, out_dim))
            nn.init.xavier_uniform_(getattr(self, 'declayer%d'%i).weight, gain=0.01)

    def forward(self, x, inits, h0):
        xc = torch.cat([x, *inits], dim=-1)
        xc, h0 = self.rnn(xc, h0)

        preds = []
        for j in range(self.n_outs):
            out = getattr(self, 'declayer%d'%j)(xc)
            preds.append(out)

        return preds, xc, h0
    
    
class NeuralInitialization(nn.Module):
    def __init__(self, in_dim, hid_dim, layer, n_layers):
        super().__init__()

        out_dim = hid_dim
        self.n_layers = n_layers
        self.num_inits = int(layer.upper() == 'LSTM') + 1
        out_dim *= self.num_inits * n_layers

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim * self.n_layers)
        self.linear3 = nn.Linear(hid_dim * self.n_layers, out_dim)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        b = x.shape[0]

        out = self.linear3(self.relu2(self.linear2(self.relu1(self.linear1(x)))))
        out = out.view(b, self.num_inits, self.n_layers, -1).permute(1, 2, 0, 3).contiguous()

        if self.num_inits == 2:
            return tuple([_ for _ in out])
        return out[0]


class Integrator(nn.Module):
    def __init__(self, in_channel, out_channel, hid_channel=1024):
        super().__init__()
        
        self.layer1 = nn.Linear(in_channel, hid_channel)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)
        
        self.layer2 = nn.Linear(hid_channel, hid_channel)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)
        
        self.layer3 = nn.Linear(hid_channel, out_channel)
        
        
    def forward(self, x, feat):
        res = x
        mask = (feat != 0).all(dim=-1).all(dim=-1)
        
        out = torch.cat((x, feat), dim=-1)
        out = self.layer1(out)
        out = self.relu1(out)
        out = self.dr1(out)
        
        out = self.layer2(out)
        out = self.relu2(out)
        out = self.dr2(out)
        
        out = self.layer3(out)
        out[mask] = out[mask] + res[mask]
        
        return out


class MotionEncoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 d_embed,
                 pose_dr,
                 rnn_type,
                 n_layers,
                 n_joints):
        super().__init__()
        
        self.n_joints = n_joints
        
        self.embed_layer = nn.Linear(in_dim, d_embed)
        self.pos_drop = nn.Dropout(pose_dr)
        
        # Keypoints initializer
        self.neural_init = NeuralInitialization(n_joints * 3 + in_dim, d_embed, rnn_type, n_layers)
        
        # 3d keypoints regressor
        self.regressor = Regressor(
            d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)
        
    def forward(self, x, init):
        """ Forward pass of motion encoder.
        """
        
        self.b, self.f = x.shape[:2]
        x = self.embed_layer(x.reshape(self.b, self.f, -1))
        x = self.pos_drop(x)
        
        h0 = self.neural_init(init)
        pred_list = [init[..., :self.n_joints * 3]]
        motion_context_list = []
        
        for i in range(self.f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            motion_context_list.append(motion_context)
            pred_list.append(pred_kp3d)
            
        pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
        motion_context = torch.cat(motion_context_list, dim=1)
        
        return pred_kp3d, motion_context

    def forward_step(self, x, init_kp, hidden_state=None):
        """
        Process a single frame through the motion encoder
        
        Args:
            x (tensor): Input keypoints for the current frame [B, 1, n_features]
            init_kp (tensor): Initial keypoints
            hidden_state (tuple, optional): Previous hidden state for RNN
            
        Returns:
            tensor: Predicted 3D keypoints
            tensor: Motion context
            tuple: Updated hidden state for next frame
        """
        batch_size, frame_size = x.shape[:2]
        
        # Process input features
        x = self.embed_layer(x.reshape(batch_size, frame_size, -1))
        x = self.pos_drop(x)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.neural_init(init_kp)
        
        # Run regressor for one step
        (pred_kp3d,), motion_context, updated_hidden = self.regressor(
            x[:,[-1]], [init_kp[..., :self.n_joints * 3]], hidden_state
        )
        
        # Reshape 3D keypoints
        pred_kp3d = pred_kp3d.view(batch_size, 1, -1, 3)
        
        return pred_kp3d, motion_context, updated_hidden

class TrajectoryDecoder(nn.Module):
    def __init__(self, 
                 d_embed,
                 rnn_type,
                 n_layers):
        super().__init__()
        
        # Trajectory regressor
        self.regressor = Regressor(
            d_embed, d_embed, [3, 6], 12, rnn_type, n_layers, )
        
    def forward(self, x, root, cam_a, h0=None):
        """ Forward pass of trajectory decoder.
        """
        
        b, f = x.shape[:2]
        pred_root_list, pred_vel_list = [root[:, :1]], []
        
        for i in range(f):
            # Global coordinate estimation
            (pred_rootv, pred_rootr), _, h0 = self.regressor(
                x[:, [i]], [pred_root_list[-1], cam_a[:, [i]]], h0)
            
            pred_root_list.append(pred_rootr)
            pred_vel_list.append(pred_rootv)
        
        pred_root = torch.cat(pred_root_list, dim=1).view(b, f + 1, -1)
        pred_vel = torch.cat(pred_vel_list, dim=1).view(b, f, -1)
        
        return pred_root, pred_vel
    def forward_step(self, x, root=None, cam_angvel=None, hidden_state=None):
        """
        Process a single frame through the trajectory decoder
        
        Args:
            x (tensor): Motion context features for current frame [B, 1, D]
            root (tensor, optional): Initial root orientation
            cam_angvel (tensor, optional): Camera angular velocity
            hidden_state (tuple, optional): Previous hidden state for RNN
            
        Returns:
            tensor: Predicted root orientation (r6d)
            tensor: Predicted root velocity
            tuple: Updated hidden state for next frame
        """
        b,f = x.shape[:2]
        
        # Run regressor for one step
        (pred_rootv, pred_rootr), _, updated_hidden = self.regressor(
            x[:,[-1]], [root, cam_angvel[:,[-1]]], hidden_state
        )
        
        return pred_rootr, pred_rootv, updated_hidden
            
class IMUProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, imu_data):
        return self.net(imu_data)
    
class MotionDecoder(nn.Module):
    def __init__(self,
                 d_embed,
                 rnn_type,
                 n_layers):  # new parameter: whether to predict root
        super().__init__()


        # Decoupled version: predict only body pose (23 joints)
        self.n_pose = 23
        pose_dim = self.n_pose * 6  # 138

        # SMPL pose initialization
        body_joints_indices = _C.BMODEL.MAIN_JOINTS[1:]  # exclude root
        self.neural_init = NeuralInitialization(
            len(body_joints_indices) * 6, d_embed, rnn_type, n_layers
        )

        # Regressor
        self.regressor = Regressor(
            d_embed, d_embed, [pose_dim, 10, 4], pose_dim, rnn_type, n_layers)

    def forward(self, x, init):
        """ Forward pass of motion decoder.
        """
        b, f = x.shape[:2]


        # Decoupled version: initialize using only body joints
        # init shape: [B, 1, 24, 6], take body joints (indices 1-23)
        init_joints = [j-1 for j in _C.BMODEL.MAIN_JOINTS if j > 0]
        h0 = self.neural_init(init[:, :, 1:][..., init_joints, :].reshape(b, 1, -1))
        init_pose = init[:, :, 1:].reshape(b, 1, -1)  # use only the body part

        # Recursive prediction
        pred_pose_list = [init_pose]
        pred_shape_list, pred_contact_list = [], []

        for i in range(f):
            # Camera coordinate estimation
            (pred_pose, pred_shape, pred_contact), _, h0 = self.regressor(
                x[:, [i]], pred_pose_list[-1:], h0)
            pred_pose_list.append(pred_pose)
            pred_shape_list.append(pred_shape)

            pred_contact_list.append(pred_contact)

        pred_pose = torch.cat(pred_pose_list[1:], dim=1).view(b, f, -1)
        pred_shape = torch.cat(pred_shape_list, dim=1).view(b, f, -1)
        pred_contact = torch.cat(pred_contact_list, dim=1).view(b, f, -1)

        return pred_pose, pred_shape, pred_contact
    def forward_step(self, x, init_smpl=None, hidden_state=None):
        """
        Process a single frame through the motion decoder
        
        Args:
            x (tensor): Motion context features for current frame [B, 1, D]
            init_smpl (tensor): Initial SMPL parameters [B, 1, D_pose]
            hidden_state (tuple, optional): Previous hidden state for RNN
            
        Returns:
            tensor: Predicted SMPL pose parameters (body only, no root)
            tensor: Predicted SMPL shape parameters
            tensor: Predicted contact probabilities
            tuple: Updated hidden state for next frame
        """
        b,f = x.shape[:2]
        
        # Initialize hidden state if not provided
        if hidden_state is None and init_smpl is not None:
            # Extract body joints (excluding root) from init_smpl
            init_joints = [j-1 for j in _C.BMODEL.MAIN_JOINTS if j > 0]
            body_joints = init_smpl[:, :, 1:][..., init_joints, :].reshape(b, 1, -1)
            hidden_state = self.neural_init(body_joints)
        
        # Prepare init_pose (body only)
        if init_smpl is not None:
            init_pose = init_smpl[:, :, 1:].reshape(b, 1, -1)  # Exclude root
        else:
            init_pose = torch.zeros(b, 1, self.n_pose * 6, device=x.device)

        # Run regressor for one step
        (pred_pose, pred_shape, pred_contact), _, updated_hidden = self.regressor(
            x[:,[-1]], [init_pose], hidden_state
        )
        
        return pred_pose, pred_shape, pred_contact, updated_hidden


class ViewDecoder(nn.Module):
    """Decode view-dependent features: global_orient and camera parameters"""
    def __init__(self,
                 d_embed,
                 rnn_type,
                 n_layers):
        super().__init__()
        
        # View decoder predicts only global_orient (6D) and cam (3D)
        
        # Global orientation initialization - use only the root joint (index 0)
        # If MAIN_JOINTS contains 0, use it; otherwise initialize directly with 6D values
        if 0 in _C.BMODEL.MAIN_JOINTS:
            init_joints = [0]  # use only the root joint
            self.neural_init = NeuralInitialization(
                len(init_joints) * 6, d_embed, rnn_type, n_layers
            )
        else:
            # Initialize global_orient directly with 6D values
            self.neural_init = NeuralInitialization(
                6, d_embed, rnn_type, n_layers
            )
        
        # Regressor: output [global_orient(6), cam(3)]
        self.regressor = Regressor(
            d_embed, d_embed, 
            [6, 3],  # [global_orient, cam]
            6,       # initialization dimension
            rnn_type, n_layers
        )
        
    def forward(self, x, init):
        """
        Args:
            x: [B, T, d_embed] - view-context features
            init: [B, 1, 24, 6] - initial SMPL pose
        Returns:
            pred_global_orient: [B, T, 6]
            pred_cam: [B, T, 3]
        """
        b, f = x.shape[:2]
        
        # Use only global_orient (root joint, index 0)
        init_global = init[:, :, 0]  # [B, 1, 6]
        h0 = self.neural_init(init_global.reshape(b, 1, -1))
        
        # Recursive prediction
        pred_global_list = [init_global.reshape(b, 1, -1)]
        pred_cam_list = []
        
        for i in range(f):
            (pred_global, pred_cam), _, h0 = self.regressor(
                x[:, [i]], pred_global_list[-1:], h0
            )
            pred_global_list.append(pred_global)
            pred_cam_list.append(pred_cam)
        
        pred_global_orient = torch.cat(pred_global_list[1:], dim=1).view(b, f, -1)
        pred_cam = torch.cat(pred_cam_list, dim=1).view(b, f, -1)
        
        return pred_global_orient, pred_cam

    def forward_step(self, x, init_smpl=None, hidden_state=None):
        """
        Process a single frame through the view decoder.

        Args:
            x: [B, T, d_embed] - view context features (typically T=1)
            init_smpl: [B, 1, 24, 6] - initial SMPL pose (used for init if no hidden_state)
            hidden_state: Previous hidden state for RNN

        Returns:
            pred_global_orient: [B, 1, 6]
            pred_cam: [B, 1, 3]
            updated_hidden: Updated hidden state
        """
        b = x.shape[0]

        # Initialize hidden state from global_orient if not provided
        if hidden_state is None and init_smpl is not None:
            init_global = init_smpl[:, :, 0]  # [B, 1, 6]
            hidden_state = self.neural_init(init_global.reshape(b, 1, -1))

        # Prepare init_global for regressor
        if init_smpl is not None:
            init_global = init_smpl[:, :, 0].reshape(b, 1, -1)  # [B, 1, 6]
        else:
            init_global = torch.zeros(b, 1, 6, device=x.device)

        # Run regressor for one step
        (pred_global, pred_cam), _, updated_hidden = self.regressor(
            x[:, [-1]], [init_global], hidden_state
        )

        return pred_global, pred_cam, updated_hidden

class TrajectoryRefiner(nn.Module):
    def __init__(self,
                 d_embed,
                 d_hidden, 
                 rnn_type,
                 n_layers):
        super().__init__()
        
        d_input = d_embed + 12
        self.refiner = Regressor(
            d_input, d_hidden, [6, 3], 9, rnn_type, n_layers)

    def forward(self, context, pred_vel, output, cam_angvel, return_y_up):
        b, f = context.shape[:2]
        
        # Register values
        pred_root = output['poses_root_r6d'].clone().detach()
        feet = output['feet'].clone().detach()
        contact = output['contact'].clone().detach()
        
        feet_vel = torch.cat((torch.zeros_like(feet[:, :1]), feet[:, 1:] - feet[:, :-1]), dim=1) * 30   # Normalize to 30 times
        feet = (feet_vel * contact.unsqueeze(-1)).reshape(b, f, -1)  # Velocity input
        inpt_feat = torch.cat([context, feet], dim=-1)
        
        (delta_root, delta_vel), _, _ = self.refiner(inpt_feat, [pred_root[:, 1:], pred_vel], h0=None)
        pred_root[:, 1:] = pred_root[:, 1:] + delta_root
        pred_vel = pred_vel + delta_vel

        output.update({
            'poses_root_r6d_refined': pred_root,
            'vel_root_refined': pred_vel,
        })
        
        return output
    def forward_step(self, context, pred_vel, output, cam_angvel=None, return_y_up=False, hidden_state=None):
        """
        Process a single frame through the trajectory refiner
        
        Args:
            context (tensor): Motion context features [B, T, D]
            pred_vel (tensor): Predicted root velocity
            output (dict): Current output dictionary
            cam_angvel (tensor, optional): Camera angular velocity
            return_y_up (bool): Whether to return y-up coordinate system
            hidden_state (tuple, optional): Previous hidden state for RNN
            
        Returns:
            dict: Updated output dictionary with refined trajectory
            tuple: Updated hidden state for next frame
        """
        batch_size = context.shape[0]
        
        # Extract data from output
        pred_root = output['poses_root_r6d'].clone().detach()
        feet = output['feet'].clone().detach()
        contact = output['contact'].clone().detach()
        
        # Calculate feet velocity
        # For streaming we're only looking at the current frame
        feet_vel = torch.zeros_like(feet)
        if 'prev_feet' in output:
            # Use previous feet position if available
            feet_vel = (feet - output['prev_feet']) * 30  # Normalize to 30 times
        
        # Apply contact mask
        feet = (feet_vel * contact.unsqueeze(-1)).reshape(batch_size, 1, -1)
        
        # Combine context and feet features
        inpt_feat = torch.cat([context[:, -1:], feet], dim=-1)
        
        # Run refiner for one step
        (delta_root, delta_vel), _, updated_hidden = self.refiner(
            inpt_feat, [pred_root[:, -1:], pred_vel], hidden_state
        )
        
        # Apply deltas
        refined_root = pred_root.clone()
        refined_root[:, -1:] = refined_root[:, -1:] + delta_root
        refined_vel = pred_vel + delta_vel
        
        # Update output with refined trajectory
        output.update({
            'poses_root_r6d_refined': refined_root,
            'vel_root_refined': refined_vel,
            'prev_feet': feet.clone()  # Store for next frame
        })
        
        return output, updated_hidden