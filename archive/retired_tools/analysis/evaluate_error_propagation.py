# evaluate_error_propagation.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import constants as _C
from lib.eval.eval_utils import (
    compute_accel,
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import pprint
import random
from lib.utils import transforms
from configs.config import parse_args
from lib.utils.utils import prepare_output_dir, create_logger
from lib.data.dataloader import setup_dloaders
from lib.models import build_network, build_body_model
from lib.utils.utils import prepare_batch

# ---- Style configuration (global) ----
# Set this at the top of the script so every plot follows the same style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
})


def setup_seed(seed):
    """ Setup seed for reproducibility """
    # ... (function body unchanged) ...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def procrustes_alignment(predicted, target):
    """ Aligns the predicted 3D pose to the target 3D pose using Procrustes analysis. """
    # ... (function body unchanged) ...
    mu_pred = predicted.mean(axis=1, keepdims=True)
    mu_target = target.mean(axis=1, keepdims=True)

    x0 = predicted - mu_pred
    y0 = target - mu_target

    ss_x = (x0**2).sum(axis=(1, 2))
    ss_y = (y0**2).sum(axis=(1, 2))

    norm_x = np.sqrt(ss_x)
    norm_y = np.sqrt(ss_y)

    x0 /= norm_x[:, None, None]
    y0 /= norm_y[:, None, None]

    A = np.einsum('bij,bkj->bik', x0, y0)
    U, s, V = np.linalg.svd(A)
    T = np.einsum('bij,bkj->bik', V.transpose(0, 2, 1), U.transpose(0, 2, 1))
    
    predicted_aligned = norm_y[:, None, None] * np.einsum('bik,bjk->bij', T, x0) + mu_target
    
    return predicted_aligned

def compute_mpjpe(predicted, target, align=False):
    """ Computes the Mean Per Joint Position Error (MPJPE). """
    # ... (function body unchanged) ...
    assert predicted.shape == target.shape
    
    if align:
        predicted_np = predicted.cpu().numpy()
        target_np = target.cpu().numpy()
        predicted_aligned = procrustes_alignment(predicted_np, target_np)
        predicted = torch.tensor(predicted_aligned, device=target.device)

    error = torch.sqrt(((predicted - target) ** 2).sum(dim=-1)).mean(dim=-1)
    return error.mean() * 1000

# ---- New plotting function ----
def plot_results(results_dict, output_dir, logger):
    """
    Plots the error propagation results in a single figure with three subplots.
    """
    noise_mm = [n * 1000 for n in results_dict.keys()]
    mpjpe_vals = [v['mpjpe'] for v in results_dict.values()]
    pa_mpjpe_vals = [v['pa_mpjpe'] for v in results_dict.values()]
    accel_err_vals = [v['accel_err'] for v in results_dict.values()]
    
    # Plot data and styling
    plot_data = [mpjpe_vals, pa_mpjpe_vals, accel_err_vals]
    titles = ['MPJPE', 'PA-MPJPE', 'Acceleration Error']
    y_labels = ['Error (mm)', 'Error (mm)', 'Error (mm/s²)']
    
    COLORS = ['#4E79A7', '#F28E2B', '#59A14F']  # Blue, Orange, Green
    MARKERS = ['o', 's', '^']
    LINESTYLES = ['-', '--', '-.']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Impact of Noise on Pose Estimation Metrics', fontweight='bold')

    for i, ax in enumerate(axes):
        ax.plot(noise_mm, plot_data[i], 
                marker=MARKERS[i], 
                linestyle=LINESTYLES[i], 
                color=COLORS[i])
        
        ax.set_title(titles[i], fontweight='bold')
        ax.set_xlabel('Input Noise Std (mm)', fontweight='bold')
        ax.set_ylabel(y_labels[i], fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust the layout to fit the main title
    
    save_path = os.path.join(output_dir, 'error_propagation_combined.png')
    plt.savefig(save_path)
    logger.info(f"Saved combined results plot to {save_path}")
    plt.show()

def run_error_propagation_analysis(cfg):
    """
    Main function to run the error propagation analysis.
    """
    # --- (existing setup and loading code unchanged) ---
    if cfg.SEED_VALUE >= 0:
        setup_seed(cfg.SEED_VALUE)
    logger = create_logger(cfg.LOGDIR, phase='eval_error_prop')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(pprint.pformat(cfg))
    data_loaders = setup_dloaders(cfg, cfg.TRAIN.DATASET_EVAL, 'val')
    val_loader = data_loaders[1]
    logger.info(f'Validation dataset loaded')
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT, map_location=cfg.DEVICE)
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if 'smpl.' not in k}
        network.load_state_dict(model_state_dict, strict=False)
        logger.info(f"Loaded checkpoint from {cfg.TRAIN.CHECKPOINT}")
    else:
        logger.error(f"No checkpoint found at {cfg.TRAIN.CHECKPOINT}")
        return
    network.to(cfg.DEVICE)
    network.eval()
    
    # --- (evaluation loop unchanged) ---
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    results = {level: {'mpjpe': [], 'pa_mpjpe': [], 'accel_err': []} for level in noise_levels}
    bar = Bar('Error Propagation Analysis', fill='#', max=len(val_loader))

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if batch is None: continue
            
            x, inits, features, kwargs, gt = prepare_batch(batch, cfg.DEVICE)
            
            x_processed = network.preprocess(x, kwargs.get('mask'))
            init_kp, init_smpl = inits
            pred_kp3d_clean, motion_context_clean = network.motion_encoder(x_processed, init_kp)

            for noise_std in noise_levels:
                noise = torch.randn_like(pred_kp3d_clean) * noise_std
                pred_kp3d_noisy = pred_kp3d_clean + noise
                
                b, f = x.shape[:2]
                view_feat = network.view_encoder(pred_kp3d_noisy)
                view_feat = view_feat.unsqueeze(1).expand_as(motion_context_clean)
                
                motion_context = motion_context_clean
                if features is not None:
                    clip_feat = network.clip_proj(features)
                    motion_context = network.clip_gated_fusion(motion_context, clip_feat)

                motion_context = network.gated_fusion(motion_context, view_feat)
                motion_context = network.dynamic_projection(motion_context, view_feat)
                motion_context = torch.cat((motion_context, pred_kp3d_noisy.reshape(b, f, -1)), dim=-1)

                pred_root, pred_vel = network.trajectory_decoder(motion_context, kwargs.get('init_root'), kwargs.get('cam_angvel'))
                
                if features is not None and network.integrator is not None:
                    motion_context = network.integrator(motion_context, features)

                pred_pose, pred_shape, pred_cam, pred_contact = network.motion_decoder(motion_context, init_smpl)
                
                rotmat = transforms.rotation_6d_to_matrix(pred_pose.reshape(*pred_pose.shape[:2], -1, 6)
        ).reshape(-1, 24, 3, 3)
                output = smpl.get_output(
                    body_pose=rotmat[:, 1:],
                    global_orient=rotmat[:, :1],
                    betas=pred_shape.view(-1, 10),
                    pose2rot=False
                )
                gt_output = smpl.get_output(
                    body_pose=transforms.rotation_6d_to_matrix(gt['pose'][0, :, 1:]),
                    global_orient=transforms.rotation_6d_to_matrix(gt['pose'][0, :, :1]),
                    betas=gt['betas'][0],
                    pose2rot=False
                )
                J_regressor_eval=torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(cfg.DEVICE)
                
                pred_j3d = torch.matmul(J_regressor_eval, output.vertices).cpu()
                target_j3d = torch.matmul(J_regressor_eval, gt_output.vertices).cpu()
                pred_verts = output.vertices.cpu()
                target_verts = gt_output.vertices.cpu()
                
                pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
                    [pred_j3d, target_j3d, pred_verts, target_verts], [2, 3]
                )
                
                errors = torch.sqrt(((pred_j3d - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
                errors_pa = torch.sqrt(((S1_hat - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

                m2mm = 1000
                accel_err = np.mean(compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d)) * m2mm
                mpjpe = np.mean(errors) * m2mm
                pa_mpjpe = np.mean(errors_pa) * m2mm

                results[noise_std]['mpjpe'].append(mpjpe)
                results[noise_std]['pa_mpjpe'].append(pa_mpjpe)
                results[noise_std]['accel_err'].append(accel_err)
            
            summary_string = f'({i + 1}/{len(val_loader)}) | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.suffix = summary_string
            bar.next()
    bar.finish()

    # --- Aggregate and print results (mostly unchanged) ---
    logger.info("="*30)
    logger.info("Error Propagation Analysis Results")
    logger.info("="*30)
    avg_results = {}
    for noise_std, errors in results.items():
        avg_mpjpe = np.mean(errors['mpjpe']) if errors['mpjpe'] else 0
        avg_pa_mpjpe = np.mean(errors['pa_mpjpe']) if errors['pa_mpjpe'] else 0
        avg_accel_err = np.mean(errors['accel_err']) if errors['accel_err'] else 0

        avg_results[noise_std] = {
            'mpjpe': avg_mpjpe,
            'pa_mpjpe': avg_pa_mpjpe,
            'accel_err': avg_accel_err
        }
        logger.info(
            f"Noise Std: {noise_std*1000:.0f}mm | "
            f"MPJPE: {avg_mpjpe:.2f}mm | "
            f"PA-MPJPE: {avg_pa_mpjpe:.2f}mm | "
            f"Accel-Err: {avg_accel_err:.2f}mm/s²"
        )
        
    # --- Plotting call after unifying the format ---
    plot_results(avg_results, cfg.LOGDIR, logger)


if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    
    if not cfg.TRAIN.CHECKPOINT:
        raise ValueError("Please provide a checkpoint file path using the --checkpoint argument.")
        
    run_error_propagation_analysis(cfg)
