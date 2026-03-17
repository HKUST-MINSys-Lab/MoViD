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
            motion: [B,T,d] 输入运动特征
            view: [B,T,d_view] 视角特征
        """
        B, T, d = motion.shape
        
        # 1. 生成基向量 (保持batch和序列维度)
        bases = self.base_gen(view).view(B, T, self.K, d)  # [B,T,K,d]
        
        # 2. 初始化投影结果
        proj = motion.clone()  # [B,T,d]
        
        # 3. 改进的Gram-Schmidt正交化
        for k in range(self.K):
            v = bases[:,:,k,:]  # [B,T,d]
            
            # 计算投影系数 (保持维度对齐)
            coef = (proj * v).sum(dim=-1, keepdim=True)  # [B,T,1]
            
            # 正交化投影
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
    """多尺度生物力学特征编码器"""
    def __init__(self, joint_dim=4, d_embed=512):
        super().__init__()
        # 生物力学特征提取
        self.bio_feat = nn.Sequential(
            nn.Linear(18, 128),  # 髋/肩空间关系
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256))
        
        # 多尺度时间卷积
        self.tconvs = nn.ModuleList([
            nn.Conv1d(256, 256, kernel_size=3, dilation=2**i, padding=2**i) 
            for i in range(3)
        ])
        # 注意力聚合
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        self.final_fc = nn.Linear(256, d_embed)

    def forward(self, kp3d):
        """输入形状：[B, T, J, 3]"""
        # 生物力学特征
        hips = kp3d[:, :, [11,12]]  # 髋关节
        shoulders = kp3d[:, :, [5,6]]  # 肩关节
        
        # 计算空间关系特征
        spatial_feat = torch.cat([
            hips.mean(2) - shoulders.mean(2),        # 躯干向量
            hips.std(2),                            # 髋部稳定性
            hips[:,:,0]-hips[:,:,1],        # 髋部活动范围
            shoulders.max(2)[0] - shoulders.min(2)[0],  # 肩部活动范围
            #shoulders[:,:,0]-shoulders[:,:,1],  # 肩部活动范围
            kp3d[:, :, [0]].expand(-1,-1,2,-1).flatten(2)  # 根节点位置
        ], dim=-1)  # [B, T, 16]
        
        bio_feat = self.bio_feat(spatial_feat)  # [B, T, 256]
        # return self.final_fc(bio_feat)  # [B, T, d_embed]
        # 多尺度时间卷积
        t_feat = bio_feat.transpose(1,2)  # [B, 256, T]
        for conv in self.tconvs:
            t_feat = F.gelu(conv(t_feat))
        t_feat = t_feat.transpose(1,2)  # [B, T, 256]
        
        # 时序注意力聚合
        attn_out, _ = self.attn(t_feat, t_feat, t_feat)  # [B, T, 256]
        pooled = F.adaptive_avg_pool1d(attn_out.transpose(1,2), 1).squeeze(-1)

        return self.final_fc(pooled)  # [B, d_embed]


class MultiScaleMotionEncoder(nn.Module):
    """多尺度运动编码器"""
    def __init__(self, in_dim, d_embed=512):
        super().__init__()
        self.temporal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, 256, kernel_size=5, stride=1, padding=2),
                nn.GELU(),
                nn.BatchNorm1d(256)  # 用 BatchNorm1d 代替 LayerNorm
            ) for _ in range(3)
        ])

        
        self.spatial_attn = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.fusion = nn.Linear(256*3, d_embed)
        
    def forward(self, x):
        """输入形状 [B, T, D]"""
        B, T, D = x.shape
        x = x.transpose(1,2)  # [B, D, T]
        print("Input to branches:", x.shape)
        
        # 多尺度特征提取
        features = []
        for branch in self.temporal_branches:
            feat = branch(x)  # [B, 256, T//2]
            print("After conv:", feat.shape)
            feat = feat.transpose(1,2)  # [B, T//2, 256]
            
            # 空间注意力
            attn_feat, _ = self.spatial_attn(feat, feat, feat)
            features.append(F.adaptive_max_pool1d(attn_feat.transpose(1,2), 1).squeeze(-1))
        
        # 多尺度特征融合
        fused = torch.cat(features, dim=-1)
        return self.fusion(fused)  # [B, d_embed]
class ViewEncoder(nn.Module):
    """视角特征编码器"""
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
    """交叉注意力融合模块（带维度投影）"""
    def __init__(self, d_model=512, n_head=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        
    def forward(self, motion_feat, view_feat):
        # 投影对齐维度
        
        # 扩展视角特征
        #view_feat = view_feat.unsqueeze(1).expand(-1, motion_feat.size(1), -1)
        
        # 注意力计算
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
        # view_feat: [B, D]
        fusion_gate = self.gate(torch.cat([motion_feat, view_feat], -1))
        return motion_feat * fusion_gate + view_feat * (1 - fusion_gate)
    
class MinimalViewEncoder(nn.Module):
    """
    极简视角编码器 - 只提取身体朝向的基本几何特征
    
    特征包括：
    1. hip_left - hip_right (髋部宽度向量, 3维)
    2. shoulder_left - shoulder_right (肩部宽度向量, 3维)
    3. 深度信息 (髋部和肩部的z坐标, 4维)
    
    总共: 10维基本特征 → 512维嵌入
    """
    def __init__(self, joint_dim=3, d_embed=512):
        super().__init__()
        
        # 极简特征提取：只用简单的MLP
        # 输入: 10维 (2个3D向量 + 4个深度值)
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
        输入: [B, T, J, 3] - 3D关节位置
        输出: [B, T, d_embed] - 视角特征
        
        关节索引（SMPL约定）:
        - 11: left_hip
        - 12: right_hip  
        - 5: left_shoulder
        - 6: right_shoulder
        """
        B, T = kp3d.shape[:2]
        
        # 提取关键关节
        left_hip = kp3d[:, :, 11]        # [B, T, 3]
        right_hip = kp3d[:, :, 12]       # [B, T, 3]
        left_shoulder = kp3d[:, :, 5]    # [B, T, 3]
        right_shoulder = kp3d[:, :, 6]   # [B, T, 3]
        
        # 1. 髋部宽度向量 (身体朝向的主要指示)
        hip_vector = left_hip - right_hip  # [B, T, 3]
        
        # 2. 肩部宽度向量 (身体朝向的辅助指示)
        shoulder_vector = left_shoulder - right_shoulder  # [B, T, 3]
        
        # 3. 深度信息 (相对于相机的距离)
        left_hip_depth = left_hip[:, :, 2:3]      # [B, T, 1] - z坐标
        right_hip_depth = right_hip[:, :, 2:3]    # [B, T, 1]
        left_shoulder_depth = left_shoulder[:, :, 2:3]   # [B, T, 1]
        right_shoulder_depth = right_shoulder[:, :, 2:3] # [B, T, 1]
        
        # 组合所有特征: [B, T, 10]
        view_features = torch.cat([
            hip_vector,              # 3维
            shoulder_vector,         # 3维
            left_hip_depth,          # 1维
            right_hip_depth,         # 1维
            left_shoulder_depth,     # 1维
            right_shoulder_depth,    # 1维
        ], dim=-1)
        
        # 通过简单MLP编码
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


# class MotionEncoder(nn.Module):
#     def __init__(self, 
#                  in_dim, 
#                  d_embed,
#                  pose_dr,
#                  rnn_type,
#                  n_layers,
#                  n_joints):
#         super().__init__()
        
#         self.n_joints = n_joints
        
#         self.embed_layer = nn.Linear(in_dim, d_embed)
#         self.pos_drop = nn.Dropout(pose_dr)
        
#         # Keypoints initializer
#         self.neural_init = NeuralInitialization(n_joints * 3 + in_dim, d_embed, rnn_type, n_layers)
        
#         # 3d keypoints regressor
#         self.regressor = Regressor(
#             d_embed, d_embed, [n_joints * 3], n_joints * 3, rnn_type, n_layers)
        
#     def forward(self, x, init):
#         """ Forward pass of motion encoder.
#         """
        
#         self.b, self.f = x.shape[:2]
#         x = self.embed_layer(x.reshape(self.b, self.f, -1))
#         x = self.pos_drop(x)
        
#         h0 = self.neural_init(init)
#         pred_list = [init[..., :self.n_joints * 3]]
#         motion_context_list = []
        
#         for i in range(self.f):
#             (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
#             motion_context_list.append(motion_context)
#             pred_list.append(pred_kp3d)
            
#         pred_kp3d = torch.cat(pred_list[1:], dim=1).view(self.b, self.f, -1, 3)
#         motion_context = torch.cat(motion_context_list, dim=1)
        
#         # Merge 3D keypoints with motion context
#         # motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
#         return pred_kp3d, motion_context

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

        # Merge 3D keypoints with motion context
        #motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        return pred_kp3d, motion_context

    def forward_step(self, x, prev_kp3d, h0=None):
        """
        Single step forward for streaming inference.

        Args:
            x: [B, T, n_joints*2+3] - input keypoints (can be window)
            prev_kp3d: [B, 1, n_joints*3] - previous 3D keypoints
            h0: previous hidden state

        Returns:
            pred_kp3d: [B, 1, n_joints, 3] - predicted 3D keypoints for current frame
            motion_context: [B, 1, d_embed] - motion context for current frame
            h0: updated hidden state
        """
        b, f = x.shape[:2]
        x_raw = x.reshape(b, f, -1)  # [B, T, in_dim] - raw keypoints
        x = self.embed_layer(x_raw)
        x = self.pos_drop(x)

        # Initialize hidden state if None (neural_init expects n_joints*3 + in_dim, use raw x not embedded)
        if h0 is None:
            init = torch.cat([prev_kp3d, x_raw[:, :1]], dim=-1)
            h0 = self.neural_init(init)

        # Only process the last frame with RNN (regressor expects 3D [B, 1, D], so flatten if 4D)
        prev_flat = prev_kp3d.reshape(b, 1, -1)[..., :self.n_joints * 3]
        pred_list = [prev_flat]

        for i in range(f):
            (pred_kp3d, ), motion_context, h0 = self.regressor(x[:, [i]], pred_list[-1:], h0)
            pred_list.append(pred_kp3d)

        # Return only the last frame's prediction
        pred_kp3d = pred_list[-1].view(b, 1, -1, 3)

        return pred_kp3d, motion_context, h0


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.head_dim = embed_dim // num_heads
        
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        
        query = self.query_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        key = self.key_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        value = self.value_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_probs, value)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        output = self.out_proj(context)
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class MultiScaleMotionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_heads=4, num_scales=3):
        super(MultiScaleMotionEncoder, self).__init__()
        
        # Multi-scale feature extraction with padding to maintain consistent dimensions
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1, stride=1) 
            for _ in range(num_scales)
        ])
        
        # Residual blocks for each scale
        self.scale_residuals = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        
        # Self-attention mechanism
        self.self_attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Fusion layer
        self.fusion_layer = nn.Conv1d(hidden_dim * num_scales, hidden_dim, kernel_size=1)
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        # Ensure input is in the right shape: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Multi-scale feature extraction with consistent dimensions
        scale_features = []
        for conv, residual in zip(self.scale_convs, self.scale_residuals):
            scale_feat = conv(x)
            scale_feat = residual(scale_feat)
            scale_features.append(scale_feat)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_features, dim=1)
        
        # Fuse multi-scale features
        fused_features = self.fusion_layer(multi_scale_features)
        
        # Transpose for self-attention
        fused_features = fused_features.transpose(1, 2)
        
        # Apply self-attention
        context_features = self.self_attention(fused_features)
        
        # Final projection
        motion_context = self.final_proj(context_features)
        motion_context += x.transpose(1, 2)
        
        return motion_context

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

    def forward_step(self, x, prev_root, cam_a, h0=None):
        """
        Single step forward for streaming inference.

        Args:
            x: [B, T, d_embed] - motion context (can be window)
            prev_root: [B, 1, 6] - previous root orientation
            cam_a: [B, T, 6] - camera angular velocity (can be window)
            h0: previous hidden state

        Returns:
            pred_root: [B, 1, 6] - predicted root for current frame
            pred_vel: [B, 1, 3] - predicted velocity for current frame
            h0: updated hidden state
        """
        b, f = x.shape[:2]

        # Process all frames in the window
        pred_root_list = [prev_root]
        pred_vel_list = []

        for i in range(f):
            (pred_rootv, pred_rootr), _, h0 = self.regressor(
                x[:, [i]], [pred_root_list[-1], cam_a[:, [i]]], h0)
            pred_root_list.append(pred_rootr)
            pred_vel_list.append(pred_rootv)

        # Return only the last frame's prediction
        pred_root = pred_root_list[-1]  # [B, 1, 6]
        pred_vel = pred_vel_list[-1]    # [B, 1, 3]

        return pred_root, pred_vel, h0


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
    
# class MotionDecoder(nn.Module):
#     def __init__(self, 
#                  d_embed,
#                  rnn_type,
#                  n_layers):
#         super().__init__()
        
#         self.n_pose = 24
        
#         # SMPL pose initialization
#         self.neural_init = NeuralInitialization(len(_C.BMODEL.MAIN_JOINTS) * 6, d_embed, rnn_type, n_layers)
        
#         # 3d keypoints regressor
#         self.regressor = Regressor(
#             d_embed, d_embed, [self.n_pose * 6, 10, 3, 4], self.n_pose * 6, rnn_type, n_layers)
        
#     def forward(self, x, init):
#         """ Forward pass of motion decoder.
#         """
#         b, f = x.shape[:2]
        
#         h0 = self.neural_init(init[:, :, _C.BMODEL.MAIN_JOINTS].reshape(b, 1, -1))
        
#         # Recursive prediction of SMPL parameters
#         pred_pose_list = [init.reshape(b, 1, -1)]
#         #pred_shape_list, pred_cam_list, pred_contact_list, pred_imu_list = [], [], [], []
#         pred_shape_list, pred_cam_list, pred_contact_list = [], [], []
        
#         for i in range(f):
#             # Camera coordinate estimation
#             #(pred_pose, pred_shape, pred_cam, pred_contact, pred_imu), _, h0 = self.regressor(x[:, [i]], pred_pose_list[-1:], h0)
#             (pred_pose, pred_shape, pred_cam, pred_contact), _, h0 = self.regressor(x[:, [i]], pred_pose_list[-1:], h0)

#             pred_pose_list.append(pred_pose)
#             pred_shape_list.append(pred_shape)
#             pred_cam_list.append(pred_cam)
#             pred_contact_list.append(pred_contact)
#             #pred_imu_list.append(pred_imu)
            
#         pred_pose = torch.cat(pred_pose_list[1:], dim=1).view(b, f, -1)
#         pred_shape = torch.cat(pred_shape_list, dim=1).view(b, f, -1)
#         pred_cam = torch.cat(pred_cam_list, dim=1).view(b, f, -1)
#         pred_contact = torch.cat(pred_contact_list, dim=1).view(b, f, -1)
#         #pred_imu = torch.cat(pred_imu_list, dim=1).view(b, f, -1)
        
#         return pred_pose, pred_shape, pred_cam, pred_contact#, pred_imu

class MotionDecoder(nn.Module):
    def __init__(self,
                 d_embed,
                 rnn_type,
                 n_layers):  # 新增参数：是否预测root
        super().__init__()


        # 解耦版本：只预测body pose (23关节)
        self.n_pose = 23
        pose_dim = self.n_pose * 6  # 138

        # SMPL pose initialization
        body_joints_indices = _C.BMODEL.MAIN_JOINTS[1:]  # 排除root
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


        # 解耦版本：只用body joints初始化
        # init shape: [B, 1, 24, 6], 取body joints (索引1-23)
        init_joints = [j-1 for j in _C.BMODEL.MAIN_JOINTS if j > 0]
        h0 = self.neural_init(init[:, :, 1:][..., init_joints, :].reshape(b, 1, -1))
        init_pose = init[:, :, 1:].reshape(b, 1, -1)  # 只用body部分

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

    def forward_step(self, x, init, h0=None):
        """
        Single step forward for streaming inference.
        Only predicts body pose (not including global_orient).

        Args:
            x: [B, T, d_embed] - motion context (can be window)
            init: [B, 1, 24, 6] - initial SMPL pose
            h0: previous hidden state

        Returns:
            pred_body_pose: [B, 1, 138] - predicted body pose for current frame
            pred_shape: [B, 1, 10] - predicted shape for current frame
            pred_contact: [B, 1, 4] - predicted contact for current frame
            h0: updated hidden state
        """
        b, f = x.shape[:2]

        # Initialize hidden state if None
        if h0 is None:
            init_joints = [j-1 for j in _C.BMODEL.MAIN_JOINTS if j > 0]
            h0 = self.neural_init(init[:, :, 1:][..., init_joints, :].reshape(b, 1, -1))

        # Use body part of init as previous pose
        init_pose = init[:, :, 1:].reshape(b, 1, -1)  # Only body part [B, 1, 138]
        pred_pose_list = [init_pose]
        pred_shape_list, pred_contact_list = [], []

        for i in range(f):
            (pred_pose, pred_shape, pred_contact), _, h0 = self.regressor(
                x[:, [i]], pred_pose_list[-1:], h0)
            pred_pose_list.append(pred_pose)
            pred_shape_list.append(pred_shape)
            pred_contact_list.append(pred_contact)

        # Return only the last frame's prediction
        pred_body_pose = pred_pose_list[-1]  # [B, 1, 138]
        pred_shape = pred_shape_list[-1]     # [B, 1, 10]
        pred_contact = pred_contact_list[-1] # [B, 1, 4]

        return pred_body_pose, pred_shape, pred_contact, h0


class ViewDecoder(nn.Module):
    """解码view-dependent特征: global_orient和camera参数"""
    def __init__(self,
                 d_embed,
                 rnn_type,
                 n_layers):
        super().__init__()
        
        # View decoder只预测global_orient(6维)和cam(3维)
        
        # Global orientation initialization - 只用root joint (索引0)
        # 如果MAIN_JOINTS包含0，则用它；否则直接用6维
        if 0 in _C.BMODEL.MAIN_JOINTS:
            init_joints = [0]  # 只用root joint
            self.neural_init = NeuralInitialization(
                len(init_joints) * 6, d_embed, rnn_type, n_layers
            )
        else:
            # 直接用6维global_orient初始化
            self.neural_init = NeuralInitialization(
                6, d_embed, rnn_type, n_layers
            )
        
        # Regressor: 输出[global_orient(6), cam(3)]
        self.regressor = Regressor(
            d_embed, d_embed, 
            [6, 3],  # [global_orient, cam]
            6,       # 初始化维度
            rnn_type, n_layers
        )
        
    def forward(self, x, init):
        """
        Args:
            x: [B, T, d_embed] - view context特征
            init: [B, 1, 24, 6] - 初始SMPL pose
        Returns:
            pred_global_orient: [B, T, 6]
            pred_cam: [B, T, 3]
        """
        b, f = x.shape[:2]
        
        # 只用global_orient (root joint, 索引0)
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

    def forward_step(self, x, init, h0=None):
        """
        Single step forward for streaming inference.
        Predicts global_orient and cam.

        Args:
            x: [B, T, d_embed] - view context (can be window)
            init: [B, 1, 24, 6] - initial SMPL pose
            h0: previous hidden state

        Returns:
            pred_global_orient: [B, 1, 6] - predicted global orientation for current frame
            pred_cam: [B, 1, 3] - predicted camera parameters for current frame
            h0: updated hidden state
        """
        b, f = x.shape[:2]

        # Initialize hidden state if None
        if h0 is None:
            init_global = init[:, :, 0]  # [B, 1, 6]
            h0 = self.neural_init(init_global.reshape(b, 1, -1))

        # Use global_orient as previous prediction
        init_global = init[:, :, 0].reshape(b, 1, -1)  # [B, 1, 6]
        pred_global_list = [init_global]
        pred_cam_list = []

        for i in range(f):
            (pred_global, pred_cam), _, h0 = self.regressor(
                x[:, [i]], pred_global_list[-1:], h0
            )
            pred_global_list.append(pred_global)
            pred_cam_list.append(pred_cam)

        # Return only the last frame's prediction
        pred_global_orient = pred_global_list[-1]  # [B, 1, 6]
        pred_cam = pred_cam_list[-1]               # [B, 1, 3]

        return pred_global_orient, pred_cam, h0


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

        # root_world, trans_world = rollout_global_motion(pred_root, pred_vel)
        
        # if return_y_up:
        #     yup2ydown = axis_angle_to_matrix(torch.tensor([[np.pi, 0, 0]])).float().to(root_world.device)
        #     root_world = yup2ydown.mT @ root_world
        #     trans_world = (yup2ydown.mT @ trans_world.unsqueeze(-1)).squeeze(-1)
            
        output.update({
            'poses_root_r6d_refined': pred_root,
            'vel_root_refined': pred_vel,
            # 'poses_root_world': root_world,
            # 'trans_world': trans_world,
        })
        
        return output

class DenoiserNetwork(nn.Module):
    def __init__(self, input_dim=93, cond_dim=512, time_dim=128, smooth_dim=51):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim))
        
        # 合并运动特征和平滑关键点
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + smooth_dim, time_dim),
            nn.LayerNorm(time_dim))
        
        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim*2, 512),  # 时间嵌入和条件各占time_dim
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Linear(512, input_dim))
    
    def forward(self, x, t_normalized, condition_with_smooth):
        """ 
        x: [B*T, J*3]
        t_normalized: [B*T,] 归一化到0~1
        condition_with_smooth: [B*T, cond_dim + smooth_dim]
        """
        t_embed = self.time_embed(t_normalized.unsqueeze(-1))
        cond_embed = self.cond_proj(condition_with_smooth)
        x_in = torch.cat([x, t_embed, cond_embed], dim=-1)
        return self.main(x_in)
    
class DiffusionWrapper(nn.Module):
    def __init__(self, denoiser, timesteps=200,
                 bones=None, 
                 joint_triples=None, angle_limits=None,
                 collision_pairs=None,
                 lambda_accel=0.1, lambda_bone=0.1, 
                 lambda_angle=0.1, lambda_collision=0.1,
                 collision_threshold=0.1):
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps)
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
        
        # 运动学约束参数
        self.bones = bones  # 骨骼连接对列表 [(j1, j2), ...]
        self.joint_triples = joint_triples  # 关节角度三元组 [(j0, j1, j2), ...]
        self.angle_limits = angle_limits  # 角度限制字典 {(j0,j1,j2): (min,max), ...}
        self.collision_pairs = collision_pairs  # 碰撞检测对列表 [(j1, j2), ...]
        self.collision_threshold = collision_threshold
        
        # 约束权重系数
        self.lambda_accel = lambda_accel
        self.lambda_bone = lambda_bone
        self.lambda_angle = lambda_angle
        self.lambda_collision = lambda_collision
        
    def compute_bone_lengths(self, x_start):
        """计算骨骼长度"""
        B, T, J, _ = x_start.shape
        bone_lengths = []
        for j1, j2 in self.bones:
            bone_lengths.append(torch.norm(x_start[:, :, j1] - x_start[:, :, j2], dim=-1))
        return torch.stack(bone_lengths, dim=-1)

    def acceleration_smoothness_loss(self, x_start):
        """加速度平滑性损失（二阶差分约束）"""
        B, T, J, _ = x_start.shape
        if T < 3:
            return torch.tensor(0.0, device=x_start.device)
        
        # 计算加速度：x(t+1) - 2x(t) + x(t-1)
        accel = x_start[:, 2:] - 2 * x_start[:, 1:-1] + x_start[:, :-2]
        return torch.mean(accel ** 2)

    def bone_length_constraint_loss(self, x_start, bone_target_lengths):
        """骨骼长度约束损失"""

        total_loss = 0.0
        for idx, (j1, j2) in enumerate(self.bones):
            # 计算当前骨骼长度
            current_lengths = torch.norm(x_start[:, :, j1] - x_start[:, :, j2], dim=-1)
            
            # 获取预计算的骨骼目标长度
            target_length = bone_target_lengths[:,:,idx]
            
            # 计算长度约束损失
            loss = F.mse_loss(current_lengths, target_length)
            total_loss += loss
            
        return total_loss / len(self.bones)

    def joint_angle_constraint_loss(self, x_start):
        """关节角度约束损失"""
        if self.joint_triples is None or self.angle_limits is None:
            return torch.tensor(0.0, device=x_start.device)
        
        total_loss = 0.0
        for triple in self.joint_triples:
            j0, j1, j2 = triple
            min_angle, max_angle = self.angle_limits[triple]
            
            # 获取关节坐标
            p0 = x_start[:, :, j0]  # [B, T, 3]
            p1 = x_start[:, :, j1]
            p2 = x_start[:, :, j2]
            
            # 计算向量
            v1 = p0 - p1  # 父关节到当前关节向量
            v2 = p2 - p1  # 当前关节到子关节向量
            
            # 计算夹角余弦值
            cos_theta = torch.sum(v1 * v2, dim=-1) / (
                torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + 1e-6)
            
            # 转换为角度（弧度）
            theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
            
            # 计算超出限制的惩罚项
            lower_violation = torch.relu(min_angle - theta)
            upper_violation = torch.relu(theta - max_angle)
            total_loss += (lower_violation + upper_violation).mean()
            
        return total_loss / len(self.joint_triples)

    def collision_detection_loss(self, x_start):
        """关节点碰撞检测损失"""
        if self.collision_pairs is None:
            return torch.tensor(0.0, device=x_start.device)
        
        total_loss = 0.0
        for j1, j2 in self.collision_pairs:
            # 计算关节点间距离
            dist = torch.norm(x_start[:, :, j1] - x_start[:, :, j2], dim=-1)
            
            # 对小于阈值的距离施加惩罚
            violation = torch.relu(self.collision_threshold - dist)
            total_loss += violation.mean()
            
        return total_loss / len(self.collision_pairs)


    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    # forward diffusion (using the nice property): q(x_t | x_0)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    # Get the mean and variance of q(x_t | x_0).
    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # compute x_0 from x_t and pred noise: the reverse of `q_sample`
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    # compute predicted mean and variance of p(x_{t-1} | x_t)
    def p_mean_variance(self, model, x_t, t, clip_denoised=True):
        # predict noise using model
        pred_noise = model(x_t, t)
        # get the predicted x_0: different from the algorithm2 in the paper
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1., max=1.)
        model_mean, posterior_variance, posterior_log_variance = \
                    self.q_posterior_mean_variance(x_recon, x_t, t)
        return model_mean, posterior_variance, posterior_log_variance
        
    # denoise_step: sample x_{t-1} from x_t and pred_noise
    @torch.no_grad()
    def p_sample(self, model, x_t, t, clip_denoised=True):
        # predict mean and variance
        model_mean, _, model_log_variance = self.p_mean_variance(model, x_t, t,
                                                    clip_denoised=clip_denoised)
        noise = torch.randn_like(x_t)
        # no noise when t == 0
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        # compute x_{t-1}
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred_img
    
    # denoise: reverse diffusion
    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        batch_size = shape[0]
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs
    
    # sample new images
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=8, channels=3):
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

    def p_losses(self, x_start, condition_with_smooth, conf):
        """整合所有损失的训练损失计算"""
        B, T, J, _ = x_start.shape
        x_flat = x_start.reshape(B*T, -1)
        bone_target_lengths = self.compute_bone_lengths(x_start)
        cond_flat = condition_with_smooth.reshape(B*T, -1)
        conf_flat = conf.reshape(B*T, J)
        
        # 基础扩散损失
        t = torch.randint(0, self.timesteps, (B*T,), device=x_start.device)
        t_normalized = t.float() / (self.timesteps - 1)
        noise = torch.randn_like(x_flat)
        x_noisy = self.q_sample(x_flat, t, noise)
        pred_noise = self.denoiser(x_noisy, t_normalized, cond_flat)

        denoised_x = self.predict_start_from_noise(x_noisy, t, pred_noise)
        denoised_x = denoised_x.reshape(B, T, J, 3)

        point_wise_loss = F.mse_loss(
            pred_noise.reshape(B*T, J, 3), 
            noise.reshape(B*T, J, 3), 
            reduction='none'
        ).mean(dim=-1)
        weighted_loss = (point_wise_loss * conf_flat).sum() / conf_flat.sum()
        
        total_loss = weighted_loss


        # 加速度平滑性约束
        accel_loss = self.acceleration_smoothness_loss(denoised_x)
        total_loss += self.lambda_accel * accel_loss
        
        # 骨骼长度约束
        bone_loss = self.bone_length_constraint_loss(denoised_x, bone_target_lengths)
        total_loss += self.lambda_bone * bone_loss
        
        # 关节角度约束
        angle_loss = self.joint_angle_constraint_loss(denoised_x)
        total_loss += self.lambda_angle * angle_loss
        
        # 碰撞检测约束
        collision_loss = self.collision_detection_loss(denoised_x)
        total_loss += self.lambda_collision * collision_loss
        
        return total_loss

    @torch.no_grad()
    def refine(self, init, condition_with_smooth, steps=50):
        """ 
        init: [B, T, J, 3] 初始平滑关键点
        condition_with_smooth: [B, T, cond_dim + smooth_dim] 合并后的条件
        """
        B, T, J, _ = init.shape
        device = init.device
        x = init.reshape(B*T, -1)
        cond_flat = condition_with_smooth.reshape(B*T, -1)
        
        # 生成时间步索引（等间隔采样训练时的时间步）
        step_indices = torch.linspace(0, self.timesteps-1, steps, dtype=torch.long)
        for t_idx in reversed(step_indices):
            t = torch.full((B*T,), t_idx.item(), device=device)
            
            # 1. 预测噪声
            t_normalized = t.float() / (self.timesteps - 1)
            pred_noise = self.denoiser(x, t_normalized, cond_flat)
            
            
            # 计算x0预测值
            x_recon = self.predict_start_from_noise(x, t, pred_noise)
            
            # 计算mu系数
            posterior_mean_coef1 = self._extract(self.posterior_mean_coef1, t, x.shape)
            posterior_mean_coef2 = self._extract(self.posterior_mean_coef2, t, x.shape)
            
            # 计算mu = coef1 * x0 + coef2 * x_t
            mu = posterior_mean_coef1 * x_recon + posterior_mean_coef2 * x
            
            # 3. 计算方差并采样
            posterior_variance = self._extract(self.posterior_variance, t, x.shape)
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mu + torch.sqrt(posterior_variance) * noise
            else:
                x = mu  # 最后一步不加噪声
                
        return x.reshape(B, T, J, 3)


    # @torch.no_grad()
    # def refine(self, init, condition_with_smooth, steps=50):
    #     """ 
    #     init: [B, T, J, 3] 初始平滑关键点
    #     condition_with_smooth: [B, T, cond_dim + smooth_dim] 合并后的条件
    #     """
    #     B, T, J, _ = init.shape
    #     x = init.reshape(B*T, -1)
    #     cond_flat = condition_with_smooth.reshape(B*T, -1)
        
    #     # 生成时间步索引（等间隔采样训练时的时间步）
    #     step_indices = torch.linspace(0, self.timesteps-1, steps, dtype=torch.long)
    #     for t_idx in reversed(step_indices):
    #         t = torch.full((B*T,), t_idx.item(), device=x.device)
    #         t_normalized = t.float() / (self.timesteps - 1)
            
    #         pred_noise = self.denoiser(x, t_normalized, cond_flat)
            
    #         # 计算当前时间步参数
    #         alpha_bar = self.alpha_bars[t_idx]
    #         alpha_bar_prev = self.alpha_bars[t_idx-1] if t_idx > 0 else 1.0
    #         beta_t = 1 - (alpha_bar / alpha_bar_prev)
            
    #         # 计算mu
    #         mu = (x - (beta_t / torch.sqrt(1 - alpha_bar)) * pred_noise) / torch.sqrt(alpha_bar / alpha_bar_prev)
            
    #         # 添加噪声
    #         if t_idx > 0:
    #             sigma_t = torch.sqrt(beta_t)
    #             noise = torch.randn_like(x)
    #             x = mu + sigma_t * noise
    #         else:
    #             x = mu
        
    #     return x.reshape(B, T, J, 3)