import torch
import torch.nn as nn
import time
import numpy as np

class GatingModule(nn.Module):
    """
    Gating module that directly decides whether to use the full pipeline or lightweight MLP
    based on view information, motion context, and computational/latency budgets.
    """
    def __init__(self, 
                 motion_context_dim,
                 view_feat_dim=None,
                 latency_budget_ms=20.0,  # Maximum allowed latency in ms
                 computational_budget=0.7,  # Maximum computational resource usage (0-1)
                 adaptation_rate=0.9):  # Exponential moving average rate for resource tracking
        super().__init__()
        
        # Calculate input dimension
        input_dim = motion_context_dim
        if view_feat_dim is not None:
            input_dim += view_feat_dim
            
        # Add dimensions for budget information
        input_dim += 2  # latency_budget and computational_budget
        
        # Gating network to directly output binary decisions for each frame
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(device='cuda')
        
        # Default hyperparameters
        self.latency_budget_ms = latency_budget_ms
        self.computational_budget = computational_budget
        self.adaptation_rate = adaptation_rate
        
        # Runtime tracking variables for monitoring and adaptation
        self.avg_full_pipeline_time = None
        self.avg_fast_path_time = None
        self.full_pipeline_count = 0
        self.fast_path_count = 0
        self.total_frames = 0
        self.threshold = 1-computational_budget
    
    def update_runtime_stats(self, full_pipeline_time, fast_path_time, decision_mask):
        """
        Update runtime statistics based on the current batch
        """
        # Update average times with exponential moving average
        if self.avg_full_pipeline_time is None:
            self.avg_full_pipeline_time = full_pipeline_time
            self.avg_fast_path_time = fast_path_time
        else:
            self.avg_full_pipeline_time = self.adaptation_rate * self.avg_full_pipeline_time + \
                                         (1 - self.adaptation_rate) * full_pipeline_time
            self.avg_fast_path_time = self.adaptation_rate * self.avg_fast_path_time + \
                                     (1 - self.adaptation_rate) * fast_path_time
        
        # Update counters
        self.full_pipeline_count += decision_mask.sum().item()
        self.fast_path_count += (~decision_mask).sum().item()
        self.total_frames += decision_mask.numel()
    

    def get_decision_mask(self, motion_context, view_feat=None):
            """
            Generate a binary decision mask (0/1) for each sequence based on inputs
            Returns:
                - decision_mask: Boolean tensor indicating which sequences to process with full pipeline
                - gating_scores: Raw average scores for each sequence
            """
            batch_size, seq_len = motion_context.shape[0], motion_context.shape[1]
            
            # Create inputs for gating network
            if view_feat is not None:
                features = torch.cat([motion_context, view_feat], dim=-1)
            else:
                features = motion_context
                
            # Add budget information to each sequence's features
            latency_budget = torch.ones(batch_size, seq_len, 1, device=motion_context.device) * self.latency_budget_ms
            comp_budget = torch.ones(batch_size, seq_len, 1, device=motion_context.device) * self.computational_budget
            
            # Combine all features
            gating_input = torch.cat([features, latency_budget, comp_budget], dim=-1)
            
            # Compute scores for each frame
            frame_scores = self.gating_network(gating_input).squeeze(-1)
            
            # Calculate average score for each sequence
            gating_scores = frame_scores.mean(dim=1)
            
            # Sort scores and select top sequences based on computational budget
            sorted_scores, sorted_indices = torch.sort(gating_scores, descending=True)

            if batch_size == 1:
                decision_mask = (gating_scores > self.threshold)
            else: 
            # Calculate how many sequences to process based on computational budget
                k = int(batch_size * self.computational_budget)
                # Create decision mask
                decision_mask = torch.zeros(batch_size, dtype=torch.bool, device=motion_context.device)
                decision_mask[sorted_indices[:k]] = True
                self.threshold = sorted_scores[k]
            
            return decision_mask, gating_scores



class AdaptiveForwardModule(nn.Module):
    """
    Wrapper module that implements the adaptive forward pass using the gating module
    """
    def __init__(self, base_model, motion_context_dim, view_feat_dim=None, 
                 latency_budget_ms=20.0, computational_budget=0.7):
        super().__init__()
        self.base_model = base_model
        self.gating_module = GatingModule(
            motion_context_dim=motion_context_dim,
            view_feat_dim=view_feat_dim,
            latency_budget_ms=latency_budget_ms,
            computational_budget=computational_budget
        )
    
    def forward(self, x, gt, inits, img_features=None, atten=True, mask=None, 
                init_root=None, cam_angvel=None, cam_intrinsics=None, bbox=None, 
                res=None, return_y_up=False, refine_traj=True, **kwargs):
        """
        Adaptive forward pass that switches between full pipeline and fast path
        based on the gating module's decisions
        """
        # Start tracking time
        start_time = time.time()
        
        # Initial preprocessing and motion encoding (always done)
        x = self.base_model.preprocess(x, mask)
        init_kp, init_smpl = inits
        pred_kp3d, motion_context = self.base_model.motion_encoder(x, init_kp)
        self.b, self.f = self.base_model.b, self.base_model.f

        # Create fast motion context and compute fast keypoints (always done)
        fast_motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        fast_kp3d = self.base_model.lightweight_mlp(fast_motion_context.reshape(self.b, self.f, -1))
        fast_kp3d = fast_kp3d.reshape(self.b, self.f, -1, 3)
        
        # Record time for fast path
        fast_path_time = (time.time() - start_time) * 1000  # convert to ms
        
        # Get view features if available (for making decision)
        view_feat = None
        if hasattr(self.base_model, 'view_encoder') and self.base_model.view_encoder is not None:
            view_feat = self.base_model.view_encoder(pred_kp3d)
            view_feat = view_feat.unsqueeze(1).expand_as(motion_context)
        
        # Get binary decision mask from gating module
        use_full_pipeline, gating_scores = self.gating_module.get_decision_mask(
            motion_context=motion_context,
            view_feat=view_feat,
        )
        
        # If all frames use fast path, skip the full pipeline
        if not use_full_pipeline.any() and not self.base_model.training:
            # Create minimal output using just the fast path results
            output = {
                'fast_kp3d': fast_kp3d,
                'gating_mask': use_full_pipeline,
                'gating_scores': gating_scores
            }
                       
            # Record end time for fast path only
            end_time = time.time()
            fast_path_total_time = (end_time - start_time) * 1000  # convert to ms
            
            # Update runtime statistics
            self.gating_module.update_runtime_stats(
                full_pipeline_time=fast_path_total_time * 2,  # Estimate full pipeline time
                fast_path_time=fast_path_total_time,
                decision_mask=use_full_pipeline
            )
            
            # Return the fast path output
            return output
        
        # If we need full pipeline for any frames, continue with the regular process
        # Save the original motion context for later
        self.base_model.old_motion_context = fast_motion_context.detach().clone()
        
        # Process with attention if needed
        ortho_loss = 0
        
        if img_features is not None and hasattr(self.base_model, 'clip_proj'):
            clip_feat = self.base_model.clip_proj(img_features)
            motion_context = self.base_model.clip_gated_fusion(motion_context, clip_feat)
        
        if view_feat is not None:
            motion_context = self.base_model.gated_fusion(motion_context, view_feat)
            motion_context = self.base_model.dynamic_projection(motion_context, view_feat)
            ortho_loss = self.base_model.orthogonal_loss(motion_context, view_feat)
        
        # Continue with the full pipeline
        motion_context = torch.cat((motion_context, pred_kp3d.reshape(self.b, self.f, -1)), dim=-1)
        
        # Decode global trajectory
        pred_root, pred_vel = self.base_model.trajectory_decoder(motion_context, init_root, cam_angvel)
        self.base_model.pred_root = pred_root
        self.base_model.pred_vel = pred_vel
        
        # Integrate features if available
        if img_features is not None and hasattr(self.base_model, 'integrator') and self.base_model.integrator is not None:
            motion_context = self.base_model.integrator(motion_context, img_features)
        
        # Decode SMPL motion
        pred_pose, pred_shape, pred_cam, pred_contact = self.base_model.motion_decoder(motion_context, init_smpl)
        
        # Register predictions
        self.base_model.pred_kp3d = pred_kp3d
        self.base_model.pred_pose = pred_pose
        self.base_model.pred_shape = pred_shape
        self.base_model.pred_cam = pred_cam
        self.base_model.pred_contact = pred_contact
        
        # Build SMPL
        output = self.base_model.forward_smpl(cam_intrinsics=cam_intrinsics, bbox=bbox, res=res)
        output['fast_kp3d'] = fast_kp3d
        output['gating_mask'] = use_full_pipeline
        output['gating_scores'] = gating_scores
        
        # Refine trajectory or rollout
        if refine_traj:
            output = self.base_model.refine_trajectory(output, cam_angvel, return_y_up)
        else:
            output = self.base_model.rollout(output, self.base_model.pred_root, self.base_model.pred_vel, return_y_up)
        
        # Training losses
        if self.base_model.training and atten and gt is not None:
            pose_feat = self.base_model.pose_proj(output['pose'][:,:,6:])
            motion_feat = self.base_model.motion_proj(motion_context.reshape(self.b, self.f, -1))
            
            contrastive_loss_safe = self.base_model.safe_contrastive_loss(
                pose_feat.reshape(self.b * self.f, -1), 
                motion_feat.reshape(self.b * self.f, -1)
            )
            
            contrast_loss = self.base_model.debug_contrastive_loss(
                pose_feat.reshape(self.b * self.f, -1), 
                motion_feat.reshape(self.b * self.f, -1)
            )
            
            output['ortho_loss'] = ortho_loss
            output['contrastive_loss'] = contrast_loss + contrastive_loss_safe
            
            output['contrastive_loss'] *= 0.05
        
        # Record end time for full pipeline
        end_time = time.time()
        full_pipeline_time = (end_time - start_time) * 1000  # convert to ms
        
        # Update runtime statistics
        self.gating_module.update_runtime_stats(
            full_pipeline_time=full_pipeline_time,
            fast_path_time=fast_path_time,
            decision_mask=use_full_pipeline
        )
        
        return output


def convert_to_adaptive_model(base_model, motion_context_dim, view_feat_dim=None, 
                              latency_budget_ms=20.0, computational_budget=0.7):
    """
    Convert a base model to use adaptive computation with the new gating module
    """
    return AdaptiveForwardModule(
        base_model=base_model,
        motion_context_dim=motion_context_dim,
        view_feat_dim=view_feat_dim,
        latency_budget_ms=latency_budget_ms,
        computational_budget=computational_budget
    )