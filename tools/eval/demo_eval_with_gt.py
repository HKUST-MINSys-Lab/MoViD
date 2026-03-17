import numpy as np
import pickle
import joblib
import os
import os.path as osp
import sys
import torch
from scipy.spatial.transform import Rotation as R
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import constants for J_regressor
try:
    from configs import constants as _C
    JOINTS_REGRESSOR_H36M = _C.BMODEL.JOINTS_REGRESSOR_H36M
    H36M_TO_J14 = _C.KEYPOINTS.H36M_TO_J14
except:
    # Fallback paths if imports fail
    JOINTS_REGRESSOR_H36M = 'data/body_models/J_regressor_h3m.npy'
    H36M_TO_J14 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]


def apply_aria01_rotation_to_3d(pose3d):
    """
    Apply rotation transformation for aria01_rgb camera to 3D coordinates
    
    This corresponds to the 2D image transformation used in aria01_rgb:
        rotated_x = height - y
        rotated_y = x
    
    The equivalent 3D rotation matrix is:
        R = [0, -1, 0]
            [1,  0, 0]
            [0,  0, 1]
    
    This transforms (X, Y, Z) -> (-Y, X, Z), which is a 90-degree 
    counter-clockwise rotation around the Z-axis (depth axis).
    
    Args:
        pose3d: (N, 3) or (N, V, 3) - 3D coordinates [X, Y, Z] in camera frame
    
    Returns:
        rotated_pose3d: Rotated 3D coordinates matching the rotated 2D image
    
    Example:
        # For vertices
        vertices_rotated = apply_aria01_rotation_to_3d(vertices)  # (N, 6890, 3)
        
        # For joints
        joints_rotated = apply_aria01_rotation_to_3d(joints)      # (N, 14, 3)
    """
    assert pose3d.shape[-1] == 3, f"Last dimension must be 3 (X, Y, Z), got shape {pose3d.shape}"
    
    # Rotation matrix: 90-degree CCW rotation around Z-axis
    rotation_matrix = np.array([
        [ 0, -1, 0],  # X_new = -Y_old (Y-axis points down, rotated points left)
        [ 1,  0, 0],  # Y_new = X_old  (X-axis points right, rotated points down)
        [ 0,  0, 1]   # Z_new = Z_old  (Depth unchanged)
    ], dtype=np.float64)
    
    original_shape = pose3d.shape
    
    # Reshape to (N, 3) for matrix multiplication
    if pose3d.ndim == 3:
        N, V, _ = pose3d.shape
        pose3d_flat = pose3d.reshape(-1, 3)
    else:
        pose3d_flat = pose3d.copy()
    
    # Apply rotation: (N, 3) @ (3, 3).T = (N, 3)
    rotated_pose3d = pose3d_flat @ rotation_matrix.T
    
    # Restore original shape
    rotated_pose3d = rotated_pose3d.reshape(original_shape)
    
    return rotated_pose3d


def load_j_regressor(device='cpu'):
    """
    Load J_regressor for H36M joints (14 joints)
    
    Returns:
        J_regressor: (14, 6890) numpy array for regressing joints from vertices
    """
    try:
        J_regressor_eval = torch.from_numpy(
            np.load(JOINTS_REGRESSOR_H36M)
        )[H36M_TO_J14, :].float()
        
        # Convert to numpy for our evaluation
        J_regressor = J_regressor_eval.numpy()
        
        print(f"Loaded J_regressor from {JOINTS_REGRESSOR_H36M}")
        print(f"J_regressor shape: {J_regressor.shape}")
        
        return J_regressor
    except Exception as e:
        print(f"Error loading J_regressor: {e}")
        print("Please ensure JOINTS_REGRESSOR_H36M path is correct")
        raise


def compute_joints_from_vertices(vertices, J_regressor):
    """
    Compute joints from SMPL vertices using J_regressor
    
    Args:
        vertices: (N, V, 3) or (V, 3) vertices array (numpy or torch)
        J_regressor: (J, V) joint regressor matrix (numpy or torch)
    
    Returns:
        joints: (N, J, 3) or (J, 3) joints array (numpy)
    """
    # Convert inputs to numpy if they're tensors
    if isinstance(J_regressor, torch.Tensor):
        J_regressor = J_regressor.cpu().numpy()
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    
    if vertices.ndim == 2:
        # Single frame: (V, 3)
        joints = np.matmul(J_regressor, vertices)  # (J, 3)
    elif vertices.ndim == 3:
        # Multiple frames: (N, V, 3)
        # Use einsum for batch matrix multiplication
        joints = np.einsum('jv,nvc->njc', J_regressor, vertices)
    else:
        raise ValueError(f"Invalid vertices shape: {vertices.shape}")
    
    return joints


def align_by_pelvis(joints, pelvis_idxs=[2, 3]):
    """
    Align joints by pelvis (root alignment)
    Args:
        joints: (N, J, 3) joints
        pelvis_idxs: indices of pelvis joints
    Returns:
        aligned_joints: (N, J, 3) pelvis-aligned joints
    """
    pelvis = joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    return joints - pelvis


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (N, 3) closest to a set of 3D points S2 (N, 3),
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert S1.shape[0] == S2.shape[0], (S1.shape, S2.shape)

    # 1. Remove mean
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform S1
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1, S2, return_params=False):
    """
    Compute similarity transform for batch processing or return parameters
    Args:
        S1: (N, 3) source points
        S2: (N, 3) target points
        return_params: if True, return (scale, R, t) instead of transformed points
    Returns:
        if return_params: (scale, R, t) - transformation parameters
        else: S1_transformed - transformed points
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    
    # 1. Remove mean
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation
    t = mu2 - scale * (R.dot(mu1))
    
    if return_params:
        return scale, R, t, mu1
    
    # 7. Transform S1
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_mpjpe(pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Mean Per Joint Position Error (MPJPE) after pelvis alignment
    Args:
        pred_joints: (N, J, 3) predicted 3D joints
        gt_joints: (N, J, 3) ground truth 3D joints
        pelvis_idxs: indices of pelvis joints for alignment
    Returns:
        mpjpe: mean per joint position error in mm
    """
    assert pred_joints.shape == gt_joints.shape
    
    # Align by pelvis
    pred_aligned = align_by_pelvis(pred_joints, pelvis_idxs)
    gt_aligned = align_by_pelvis(gt_joints, pelvis_idxs)
    
    # Compute MPJPE
    error = np.sqrt(np.sum((pred_aligned - gt_aligned) ** 2, axis=-1))  # (N, J)
    mpjpe = np.mean(error) * 1000  # convert to mm
    return mpjpe


def compute_pa_mpjpe(pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE)
    Args:
        pred_joints: (N, J, 3) predicted 3D joints
        gt_joints: (N, J, 3) ground truth 3D joints
        pelvis_idxs: indices of pelvis joints for alignment
    Returns:
        pa_mpjpe: procrustes aligned mean per joint position error in mm
    """
    assert pred_joints.shape == gt_joints.shape
    
    # Align by pelvis first
    pred_aligned = align_by_pelvis(pred_joints, pelvis_idxs)
    gt_aligned = align_by_pelvis(gt_joints, pelvis_idxs)
    
    N, J, _ = pred_aligned.shape
    errors = []
    
    for i in range(N):
        # Apply Procrustes alignment for each frame
        pred_proc = compute_similarity_transform(pred_aligned[i], gt_aligned[i])
        
        # Compute error
        error = np.sqrt(np.sum((pred_proc - gt_aligned[i]) ** 2, axis=-1))
        errors.append(np.mean(error))
    
    pa_mpjpe = np.mean(errors) * 1000  # convert to mm
    return pa_mpjpe


def compute_pve(pred_verts, gt_verts, pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Per Vertex Error (PVE) after pelvis alignment
    Args:
        pred_verts: (N, V, 3) predicted vertices
        gt_verts: (N, V, 3) ground truth vertices
        pred_joints: (N, J, 3) predicted joints (for pelvis alignment)
        gt_joints: (N, J, 3) ground truth joints (for pelvis alignment)
        pelvis_idxs: indices of pelvis joints
    Returns:
        pve: mean per vertex error in mm
    """
    assert pred_verts.shape == gt_verts.shape
    assert pred_joints.shape == gt_joints.shape
    
    # Compute pelvis position from joints (same as MPJPE)
    pred_pelvis = pred_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    gt_pelvis = gt_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    
    # Align vertices using pelvis from joints
    pred_verts_aligned = pred_verts - pred_pelvis
    gt_verts_aligned = gt_verts - gt_pelvis
    
    # Compute PVE
    error = np.sqrt(np.sum((pred_verts_aligned - gt_verts_aligned) ** 2, axis=-1))  # (N, V)
    pve = np.mean(error) * 1000  # convert to mm
    return pve


def compute_pa_pve(pred_verts, gt_verts, pred_joints, gt_joints, pelvis_idxs=[2, 3]):
    """
    Compute Procrustes Aligned Per Vertex Error (PA-PVE)
    IMPORTANT: Uses joint alignment parameters to align vertices (correct way)
    
    Args:
        pred_verts: (N, V, 3) predicted vertices
        gt_verts: (N, V, 3) ground truth vertices
        pred_joints: (N, J, 3) predicted joints (for computing alignment)
        gt_joints: (N, J, 3) ground truth joints (for computing alignment)
        pelvis_idxs: indices of pelvis joints
    Returns:
        pa_pve: procrustes aligned mean per vertex error in mm
    """
    assert pred_verts.shape == gt_verts.shape
    assert pred_joints.shape == gt_joints.shape
    
    # Compute pelvis position from joints
    pred_pelvis = pred_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    gt_pelvis = gt_joints[:, pelvis_idxs, :].mean(axis=1, keepdims=True)  # (N, 1, 3)
    
    # Align joints and vertices using pelvis from joints
    pred_joints_aligned = pred_joints - pred_pelvis
    gt_joints_aligned = gt_joints - gt_pelvis
    pred_verts_aligned = pred_verts - pred_pelvis
    gt_verts_aligned = gt_verts - gt_pelvis
    
    N, V, _ = pred_verts_aligned.shape
    errors = []
    
    for i in range(N):
        # ⭐ KEY FIX: Compute Procrustes transform using JOINTS (not vertices)
        # This ensures fair comparison - we use semantic landmarks to align
        scale, R, t, mu1 = compute_similarity_transform_batch(
            pred_joints_aligned[i], 
            gt_joints_aligned[i],
            return_params=True
        )
        
        # Apply the joint-based transformation to VERTICES
        pred_verts_centered = pred_verts_aligned[i].T - mu1
        pred_verts_proc = scale * R.dot(pred_verts_centered) + t
        pred_verts_proc = pred_verts_proc.T  # Back to (V, 3)
        
        # Compute error on transformed vertices
        error = np.sqrt(np.sum((pred_verts_proc - gt_verts_aligned[i]) ** 2, axis=-1))
        errors.append(np.mean(error))
    
    pa_pve = np.mean(errors) * 1000  # convert to mm
    return pa_pve


def load_gt_smpl_data_npz(root_dir, camera_name='aria01', apply_rotation=True):
    """
    Load all ground truth SMPL data from NPZ files
    
    Args:
        root_dir: Root directory containing processed_data
        camera_name: Camera name (e.g., 'aria01')
        apply_rotation: If True and camera is 'aria01', apply 3D rotation to match
                       the rotated 2D images used in visualization
    
    Returns:
        dict with structure:
        {
            'aria02': {
                'vertices': (N, V, 3),
                'frame_ids': (N,),
                'joints': (N, J, 3) or None
            },
            'aria03': {...},
            ...
        }
    """
    smpl_3d_dir = osp.join(root_dir, 'processed_data', 'smpl_3d_points', f'{camera_name}_rgb')
    
    if not osp.exists(smpl_3d_dir):
        print(f"Error: Directory not found: {smpl_3d_dir}")
        return {}
    
    gt_data = {}
    
    # Find all NPZ files (each represents a subject)
    npz_files = [f for f in os.listdir(smpl_3d_dir) if f.endswith('.npz')]
    
    print(f"Found {len(npz_files)} NPZ files in {smpl_3d_dir}")
    
    # Check if rotation should be applied
    should_rotate = apply_rotation and (camera_name == 'aria01')
    if should_rotate:
        print(f"  ⚠️  Will apply 3D rotation for {camera_name}_rgb camera")
        print(f"  This matches the 2D image rotation used in visualization")
    
    for npz_file in sorted(npz_files):
        subject_name = osp.splitext(npz_file)[0]  # e.g., 'aria02'
        npz_path = osp.join(smpl_3d_dir, npz_file)
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # Extract data from NPZ
            # The actual keys are: 'points_3d_cam', 'time_stamps', 'camera_id', 'human_name'
            vertices = data['points_3d_cam']  # (N, V, 3) - vertices in camera coordinates
            time_stamps = data['time_stamps']  # (N,) - timestamps (can be used as frame_ids)
            
            # Apply 3D rotation if needed (for aria01_rgb)
            if should_rotate:
                vertices = apply_aria01_rotation_to_3d(vertices)
            
            # Use time_stamps as frame_ids (they should be sequential integers)
            frame_ids = time_stamps.astype(np.int64)
            
            # No joints provided in NPZ, will compute from vertices using J_regressor
            joints = None
            
            gt_data[subject_name] = {
                'vertices': vertices,
                'frame_ids': frame_ids,
                'joints': joints
            }
            
            rotation_status = " (rotated)" if should_rotate else ""
            print(f"  Loaded {subject_name}: {len(frame_ids)} frames "
                  f"(timestamp {frame_ids[0]} to {frame_ids[-1]}){rotation_status}")
            print(f"    Vertices shape: {vertices.shape}")
            
        except Exception as e:
            print(f"  Error loading {npz_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return gt_data


def visualize_alignment(pred_verts, gt_verts, pred_joints, gt_joints, 
                       frame_idx=0, pelvis_idxs=[2, 3], output_path='alignment_check.png'):
    """
    Visualize predicted and GT alignment in 3D
    Shows: Original, Pelvis-aligned, and Procrustes-aligned views
    
    Args:
        pred_verts: (N, V, 3) predicted vertices
        gt_verts: (N, V, 3) ground truth vertices
        pred_joints: (N, J, 3) predicted joints
        gt_joints: (N, J, 3) ground truth joints
        frame_idx: frame index to visualize
        pelvis_idxs: pelvis joint indices
        output_path: path to save the visualization
    """
    fig = plt.figure(figsize=(18, 5))
    
    # Sample vertices for faster visualization (every 10th vertex)
    sample_rate = 10
    
    # Extract single frame
    pred_v = pred_verts[frame_idx]
    gt_v = gt_verts[frame_idx]
    pred_j = pred_joints[frame_idx]
    gt_j = gt_joints[frame_idx]
    
    # 1. Original coordinates
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pred_v[::sample_rate, 0], pred_v[::sample_rate, 1], pred_v[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred')
    ax1.scatter(gt_v[::sample_rate, 0], gt_v[::sample_rate, 1], gt_v[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    # Plot joints
    ax1.scatter(pred_j[:, 0], pred_j[:, 1], pred_j[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8, label='Pred Joints')
    ax1.scatter(gt_j[:, 0], gt_j[:, 1], gt_j[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8, label='GT Joints')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Coordinates')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.view_init(elev=20, azim=45)
    
    # 2. Pelvis-aligned
    pred_pelvis = pred_j[pelvis_idxs, :].mean(axis=0, keepdims=True)
    gt_pelvis = gt_j[pelvis_idxs, :].mean(axis=0, keepdims=True)
    
    pred_v_aligned = pred_v - pred_pelvis
    gt_v_aligned = gt_v - gt_pelvis
    pred_j_aligned = pred_j - pred_pelvis
    gt_j_aligned = gt_j - gt_pelvis
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_v_aligned[::sample_rate, 0], pred_v_aligned[::sample_rate, 1], pred_v_aligned[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred')
    ax2.scatter(gt_v_aligned[::sample_rate, 0], gt_v_aligned[::sample_rate, 1], gt_v_aligned[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    ax2.scatter(pred_j_aligned[:, 0], pred_j_aligned[:, 1], pred_j_aligned[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8)
    ax2.scatter(gt_j_aligned[:, 0], gt_j_aligned[:, 1], gt_j_aligned[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Pelvis-Aligned')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.view_init(elev=20, azim=45)
    
    # 3. Procrustes-aligned
    # Compute Procrustes transform using joints
    pred_j_aligned_T = pred_j_aligned.T
    gt_j_aligned_T = gt_j_aligned.T
    
    # Center
    mu1 = pred_j_aligned_T.mean(axis=1, keepdims=True)
    mu2 = gt_j_aligned_T.mean(axis=1, keepdims=True)
    X1 = pred_j_aligned_T - mu1
    X2 = gt_j_aligned_T - mu2
    
    # Compute scale and rotation
    var1 = np.sum(X1**2)
    K = X1.dot(X2.T)
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    R_mat = V.dot(Z.dot(U.T))
    scale = np.trace(R_mat.dot(K)) / var1
    t = mu2 - scale * (R_mat.dot(mu1))
    
    # Apply transform to vertices
    pred_v_centered = pred_v_aligned.T - mu1
    pred_v_proc = scale * R_mat.dot(pred_v_centered) + t
    pred_v_proc = pred_v_proc.T
    
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(pred_v_proc[::sample_rate, 0], pred_v_proc[::sample_rate, 1], pred_v_proc[::sample_rate, 2], 
                c='red', marker='o', s=1, alpha=0.6, label='Pred (aligned)')
    ax3.scatter(gt_v_aligned[::sample_rate, 0], gt_v_aligned[::sample_rate, 1], gt_v_aligned[::sample_rate, 2], 
                c='blue', marker='^', s=1, alpha=0.6, label='GT')
    
    # Apply transform to joints for visualization
    pred_j_centered = pred_j_aligned.T - mu1
    pred_j_proc = scale * R_mat.dot(pred_j_centered) + t
    pred_j_proc = pred_j_proc.T
    
    ax3.scatter(pred_j_proc[:, 0], pred_j_proc[:, 1], pred_j_proc[:, 2], 
                c='darkred', marker='o', s=30, alpha=0.8)
    ax3.scatter(gt_j_aligned[:, 0], gt_j_aligned[:, 1], gt_j_aligned[:, 2], 
                c='darkblue', marker='^', s=30, alpha=0.8)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Procrustes-Aligned')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Saved alignment visualization to {output_path}")


def match_pred_to_gt_batch(pred_joints_batch, pred_verts_batch, frame_ids, 
                           gt_data_all, J_regressor, pelvis_idxs=[2, 3]):
    """
    Match predicted track to GT person across multiple frames
    Uses PA-MPJPE as the matching criterion (based on semantic joint landmarks)
    
    Args:
        pred_joints_batch: (N, J, 3) predicted joints for N frames
        pred_verts_batch: (N, V, 3) predicted vertices for N frames (or None)
        frame_ids: (N,) array of frame IDs
        gt_data_all: dict of all GT data (output from load_gt_smpl_data_npz)
        J_regressor: (J, V) joint regressor matrix (torch tensor)
        pelvis_idxs: pelvis joint indices
    
    Returns:
        best_match_person: string, name of matched GT person (e.g., 'aria02')
        metrics: dict with computed metrics for the best match
    """
    
    if len(gt_data_all) == 0:
        return None, None
    
    # Convert frame_ids to numpy array
    if not isinstance(frame_ids, np.ndarray):
        frame_ids = np.array(frame_ids)
    
    # Convert J_regressor to numpy if needed
    if isinstance(J_regressor, torch.Tensor):
        J_regressor_np = J_regressor.squeeze(0).cpu().numpy()
    else:
        J_regressor_np = J_regressor
    
    # Try matching with each GT person
    best_pa_mpjpe = float('inf')
    best_match_person = None
    best_metrics = None
    
    for person_name, person_gt_data in gt_data_all.items():
        gt_frame_ids = person_gt_data['frame_ids']
        gt_vertices_all = person_gt_data['vertices']  # (N_gt, V, 3)
        
        # Compute or get GT joints
        if person_gt_data['joints'] is not None:
            gt_joints_all = person_gt_data['joints']  # (N_gt, J, 3)
        else:
            # Compute joints from vertices using J_regressor
            gt_joints_all = compute_joints_from_vertices(gt_vertices_all, J_regressor_np)
        
        # Find matching frames
        # Create a mapping from frame_id to index in GT data
        gt_frame_to_idx = {fid: idx for idx, fid in enumerate(gt_frame_ids)}
        
        # Find valid frame indices
        valid_pred_indices = []
        valid_gt_indices = []
        
        for pred_idx, fid in enumerate(frame_ids):
            if fid in gt_frame_to_idx:
                valid_pred_indices.append(pred_idx)
                valid_gt_indices.append(gt_frame_to_idx[fid])
        
        if len(valid_pred_indices) == 0:
            continue
        
        # Extract matching frames
        pred_joints_subset = pred_joints_batch[valid_pred_indices]
        gt_joints_subset = gt_joints_all[valid_gt_indices]
        gt_verts_subset = gt_vertices_all[valid_gt_indices]
        
        # Ensure same number of joints (take minimum)
        min_joints = min(pred_joints_subset.shape[1], gt_joints_subset.shape[1])
        pred_joints_subset = pred_joints_subset[:, :min_joints, :]
        gt_joints_subset = gt_joints_subset[:, :min_joints, :]
        
        try:
            # Compute PA-MPJPE for matching (joint-based, stable)
            pa_mpjpe = compute_pa_mpjpe(pred_joints_subset, gt_joints_subset, pelvis_idxs)
            mpjpe = compute_mpjpe(pred_joints_subset, gt_joints_subset, pelvis_idxs)
            
            # Compute PVE metrics if vertices available
            pve = None
            pa_pve = None
            if pred_verts_batch is not None:
                pred_verts_subset = pred_verts_batch[valid_pred_indices]
                
                # Ensure same number of vertices
                min_verts = min(pred_verts_subset.shape[1], gt_verts_subset.shape[1])
                pred_verts_subset = pred_verts_subset[:, :min_verts, :]
                gt_verts_subset_trimmed = gt_verts_subset[:, :min_verts, :]
                
                pve = compute_pve(
                    pred_verts_subset, gt_verts_subset_trimmed,
                    pred_joints_subset, gt_joints_subset,
                    pelvis_idxs
                )
                
                pa_pve = compute_pa_pve(
                    pred_verts_subset, gt_verts_subset_trimmed,
                    pred_joints_subset, gt_joints_subset,
                    pelvis_idxs
                )
            
            # Update best match based on PA-MPJPE
            if pa_mpjpe < best_pa_mpjpe:
                best_pa_mpjpe = pa_mpjpe
                best_match_person = person_name
                
                best_metrics = {
                    'mpjpe': mpjpe,
                    'pa_mpjpe': pa_mpjpe,
                    'pve': pve,
                    'pa_pve': pa_pve,
                    'num_frames': len(valid_pred_indices),
                    'valid_frame_indices': valid_pred_indices,
                    'matched_frame_ids': frame_ids[valid_pred_indices]
                }
        
        except Exception as e:
            print(f"    Error matching with person {person_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return best_match_person, best_metrics


def evaluate_wham_output(wham_pkl_path, root_dir, camera_name='aria01',
                         pelvis_idxs=[2, 3], j_regressor_path=None,
                         apply_rotation=True):
    """
    Evaluate WHAM output against ground truth SMPL data (NPZ format)
    
    Args:
        wham_pkl_path: Path to wham_output.pkl
        root_dir: Root directory containing processed_data/smpl_3d_points/
        camera_name: Name of the aria camera perspective (e.g., 'aria01')
        pelvis_idxs: Indices of pelvis joints for alignment (default [2, 3] for H36M-14)
        j_regressor_path: Path to J_regressor file (optional)
        apply_rotation: If True, apply 3D rotation for aria01_rgb camera to match
                       rotated 2D images (default: True)
    """
    # Load J_regressor
    print("Loading J_regressor for computing joints from vertices...")
    try:
        from configs import constants as _C
        J_regressor = torch.from_numpy(
            np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
        )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float()
    except:
        if j_regressor_path and osp.exists(j_regressor_path):
            J_regressor = torch.from_numpy(
                np.load(j_regressor_path)
            ).unsqueeze(0).float()
        else:
            print("Error: Could not load J_regressor")
            return None
    
    print(f"J_regressor shape: {J_regressor.shape}")
    print(f"Using pelvis joint indices: {pelvis_idxs} (for 14-joint H36M format)\n")
    
    # Load all GT data
    print(f"Loading GT data from {root_dir}")
    print(f"Camera perspective: {camera_name}")
    gt_data_all = load_gt_smpl_data_npz(root_dir, camera_name, apply_rotation=apply_rotation)
    
    if len(gt_data_all) == 0:
        print("Error: No GT data loaded!")
        return None
    
    print(f"\nLoaded GT data for {len(gt_data_all)} subjects")
    print()
    
    # Load WHAM predictions
    print(f"Loading WHAM predictions from {wham_pkl_path}")
    wham_results = joblib.load(wham_pkl_path)
    
    # Initialize metrics storage
    all_mpjpe = []
    all_pa_mpjpe = []
    all_pve = []
    all_pa_pve = []
    
    match_statistics = {}
    track_results = []
    
    print(f"\nEvaluating {len(wham_results)} track predictions...")
    print(f"Matching criterion: PA-MPJPE (joint-based)")
    print(f"PA-PVE: Uses joint alignment parameters (correct method)\n")
    
    for track_id, pred_data in wham_results.items():
        frame_ids = pred_data['frame_ids']
        
        # Convert to numpy array if needed
        if not isinstance(frame_ids, np.ndarray):
            frame_ids = np.array(frame_ids).flatten()
        
        # --- MODIFICATION START ---
        # Get prediction data from 'verts' as 'joints3d' may not be present.
        pred_verts = pred_data.get('verts')
        
        if pred_verts is None:
            print(f"Warning: No 'verts' data found for track {track_id}. Skipping.")
            continue
            
        # Ensure pred_verts is a numpy array and has a batch dimension
        if isinstance(pred_verts, torch.Tensor):
            pred_verts = pred_verts.cpu().numpy()
        if pred_verts.ndim == 2:
            pred_verts = pred_verts[np.newaxis, ...] # from (V, 3) to (1, V, 3)

        # Always compute joints from vertices
        J_regressor_np = J_regressor.squeeze(0).cpu().numpy()
        pred_joints = compute_joints_from_vertices(pred_verts, J_regressor_np)
        # --- MODIFICATION END ---
        
        print(f"{'='*60}")
        print(f"Track {track_id}: {len(frame_ids)} frames (frame {frame_ids[0]} to {frame_ids[-1]})")
        print(f"  Pred verts shape: {pred_verts.shape}")
        print(f"  Computed pred joints shape: {pred_joints.shape}")
        
        # Match prediction to GT person
        best_match_person, metrics = match_pred_to_gt_batch(
            pred_joints, pred_verts, frame_ids, gt_data_all, J_regressor, pelvis_idxs
        )
        
        if best_match_person is None or metrics is None:
            print(f"  ⚠️  Could not match track {track_id} to any GT person")
            continue
        
        # Store results
        all_mpjpe.append(metrics['mpjpe'])
        all_pa_mpjpe.append(metrics['pa_mpjpe'])
        if metrics['pve'] is not None:
            all_pve.append(metrics['pve'])
        if metrics['pa_pve'] is not None:
            all_pa_pve.append(metrics['pa_pve'])
        
        # Track statistics per person
        if best_match_person not in match_statistics:
            match_statistics[best_match_person] = []
        match_statistics[best_match_person].append({
            'track_id': track_id,
            'mpjpe': metrics['mpjpe'],
            'pa_mpjpe': metrics['pa_mpjpe'],
            'pve': metrics['pve'],
            'pa_pve': metrics['pa_pve'],
            'num_frames': metrics['num_frames']
        })
        
        track_results.append({
            'track_id': track_id,
            'matched_person': best_match_person,
            'mpjpe': metrics['mpjpe'],
            'pa_mpjpe': metrics['pa_mpjpe'],
            'pve': metrics['pve'],
            'pa_pve': metrics['pa_pve'],
            'num_frames': metrics['num_frames']
        })
        
        print(f"  ✓ Matched to: {best_match_person}")
        print(f"    MPJPE:    {metrics['mpjpe']:.2f} mm")
        print(f"    PA-MPJPE: {metrics['pa_mpjpe']:.2f} mm")
        if metrics['pve'] is not None:
            print(f"    PVE:      {metrics['pve']:.2f} mm")
        if metrics['pa_pve'] is not None:
            print(f"    PA-PVE:   {metrics['pa_pve']:.2f} mm")
        print(f"    Valid frames: {metrics['num_frames']}/{len(frame_ids)}")
        
        # Visualize alignment for the first valid frame
        if pred_verts is not None and len(track_results) <= 3:  # Only visualize first 3 tracks
            # Get GT data for matched person
            person_gt_data = gt_data_all[best_match_person]
            gt_frame_ids = person_gt_data['frame_ids']
            gt_vertices_all = person_gt_data['vertices']
            
            # Compute GT joints if needed
            if person_gt_data['joints'] is not None:
                gt_joints_all = person_gt_data['joints']
            else:
                J_regressor_np = J_regressor.squeeze(0).cpu().numpy()
                gt_joints_all = compute_joints_from_vertices(gt_vertices_all, J_regressor_np)
            
            # Find matching frames
            gt_frame_to_idx = {fid: idx for idx, fid in enumerate(gt_frame_ids)}
            valid_pred_indices = []
            valid_gt_indices = []
            for pred_idx, fid in enumerate(frame_ids):
                if fid in gt_frame_to_idx:
                    valid_pred_indices.append(pred_idx)
                    valid_gt_indices.append(gt_frame_to_idx[fid])
            
            if len(valid_pred_indices) > 0:
                # Visualize the middle frame
                vis_idx = len(valid_pred_indices) // 2
                pred_verts_subset = pred_verts[valid_pred_indices]
                pred_joints_subset = pred_joints[valid_pred_indices]
                gt_verts_subset = gt_vertices_all[valid_gt_indices]
                gt_joints_subset = gt_joints_all[valid_gt_indices]
                
                # Ensure same number of vertices
                min_verts = min(pred_verts_subset.shape[1], gt_verts_subset.shape[1])
                pred_verts_subset = pred_verts_subset[:, :min_verts, :]
                gt_verts_subset = gt_verts_subset[:, :min_verts, :]
                
                min_joints = min(pred_joints_subset.shape[1], gt_joints_subset.shape[1])
                pred_joints_subset = pred_joints_subset[:, :min_joints, :]
                gt_joints_subset = gt_joints_subset[:, :min_joints, :]
                
                output_dir = osp.dirname(wham_pkl_path)
                vis_path = osp.join(output_dir, f'alignment_track_{track_id}_frame_{frame_ids[valid_pred_indices[vis_idx]]}.png')
                
                visualize_alignment(
                    pred_verts_subset, gt_verts_subset,
                    pred_joints_subset, gt_joints_subset,
                    frame_idx=vis_idx,
                    pelvis_idxs=pelvis_idxs,
                    output_path=vis_path
                )
        print()
    
    # Compute overall statistics
    if len(all_mpjpe) > 0:
        print("\n" + "="*60)
        print("OVERALL EVALUATION RESULTS")
        print("="*60)
        print(f"Total tracks evaluated: {len(all_mpjpe)}")
        print(f"\nMPJPE (Root-aligned):")
        print(f"  Mean:   {np.mean(all_mpjpe):.2f} mm")
        print(f"  Std:    {np.std(all_mpjpe):.2f} mm")
        print(f"  Median: {np.median(all_mpjpe):.2f} mm")
        print(f"  Min:    {np.min(all_mpjpe):.2f} mm")
        print(f"  Max:    {np.max(all_mpjpe):.2f} mm")
        
        print(f"\nPA-MPJPE (Procrustes-aligned):")
        print(f"  Mean:   {np.mean(all_pa_mpjpe):.2f} mm")
        print(f"  Std:    {np.std(all_pa_mpjpe):.2f} mm")
        print(f"  Median: {np.median(all_pa_mpjpe):.2f} mm")
        print(f"  Min:    {np.min(all_pa_mpjpe):.2f} mm")
        print(f"  Max:    {np.max(all_pa_mpjpe):.2f} mm")
        
        if len(all_pve) > 0:
            print(f"\nPVE (Per Vertex Error, Root-aligned):")
            print(f"  Mean:   {np.mean(all_pve):.2f} mm")
            print(f"  Std:    {np.std(all_pve):.2f} mm")
            print(f"  Median: {np.median(all_pve):.2f} mm")
            print(f"  Min:    {np.min(all_pve):.2f} mm")
            print(f"  Max:    {np.max(all_pve):.2f} mm")
        
        if len(all_pa_pve) > 0:
            print(f"\nPA-PVE (Procrustes-aligned Per Vertex Error):")
            print(f"  Mean:   {np.mean(all_pa_pve):.2f} mm")
            print(f"  Std:    {np.std(all_pa_pve):.2f} mm")
            print(f"  Median: {np.median(all_pa_pve):.2f} mm")
            print(f"  Min:    {np.min(all_pa_pve):.2f} mm")
            print(f"  Max:    {np.max(all_pa_pve):.2f} mm")
        
        print(f"\n{'='*60}")
        print("PER-PERSON STATISTICS")
        print("="*60)
        for person, stats_list in sorted(match_statistics.items()):
            person_mpjpe = [s['mpjpe'] for s in stats_list]
            person_pa_mpjpe = [s['pa_mpjpe'] for s in stats_list]
            total_frames = sum([s['num_frames'] for s in stats_list])
            
            print(f"\n{person} ({len(stats_list)} tracks, {total_frames} frames):")
            print(f"  MPJPE:    {np.mean(person_mpjpe):.2f} ± {np.std(person_mpjpe):.2f} mm")
            print(f"  PA-MPJPE: {np.mean(person_pa_mpjpe):.2f} ± {np.std(person_pa_mpjpe):.2f} mm")
            
            if all([s['pve'] is not None for s in stats_list]):
                person_pve = [s['pve'] for s in stats_list]
                print(f"  PVE:      {np.mean(person_pve):.2f} ± {np.std(person_pve):.2f} mm")
            
            if all([s['pa_pve'] is not None for s in stats_list]):
                person_pa_pve = [s['pa_pve'] for s in stats_list]
                print(f"  PA-PVE:   {np.mean(person_pa_pve):.2f} ± {np.std(person_pa_pve):.2f} mm")
        
        # Save results
        results_dict = {
            'overall': {
                'mpjpe': {
                    'mean': float(np.mean(all_mpjpe)),
                    'std': float(np.std(all_mpjpe)),
                    'median': float(np.median(all_mpjpe)),
                    'min': float(np.min(all_mpjpe)),
                    'max': float(np.max(all_mpjpe))
                },
                'pa_mpjpe': {
                    'mean': float(np.mean(all_pa_mpjpe)),
                    'std': float(np.std(all_pa_mpjpe)),
                    'median': float(np.median(all_pa_mpjpe)),
                    'min': float(np.min(all_pa_mpjpe)),
                    'max': float(np.max(all_pa_mpjpe))
                }
            },
            'per_track': track_results,
            'per_person': match_statistics,
        }
        
        if len(all_pve) > 0:
            results_dict['overall']['pve'] = {
                'mean': float(np.mean(all_pve)),
                'std': float(np.std(all_pve)),
                'median': float(np.median(all_pve)),
                'min': float(np.min(all_pve)),
                'max': float(np.max(all_pve))
            }
        
        if len(all_pa_pve) > 0:
            results_dict['overall']['pa_pve'] = {
                'mean': float(np.mean(all_pa_pve)),
                'std': float(np.std(all_pa_pve)),
                'median': float(np.median(all_pa_pve)),
                'min': float(np.min(all_pa_pve)),
                'max': float(np.max(all_pa_pve))
            }
        
        output_dir = osp.dirname(wham_pkl_path)
        results_path = osp.join(output_dir, 'evaluation_results.pkl')
        joblib.dump(results_dict, results_path)
        print(f"\nResults saved to {results_path}")
        
        # Save CSV
        csv_path = osp.join(output_dir, 'evaluation_results.csv')
        with open(csv_path, 'w') as f:
            f.write("track_id,matched_person,mpjpe,pa_mpjpe,pve,pa_pve,num_frames\n")
            for result in track_results:
                pve_str = f"{result['pve']:.4f}" if result['pve'] is not None else "N/A"
                pa_pve_str = f"{result['pa_pve']:.4f}" if result['pa_pve'] is not None else "N/A"
                f.write(f"{result['track_id']},{result['matched_person']},"
                       f"{result['mpjpe']:.4f},{result['pa_mpjpe']:.4f},"
                       f"{pve_str},{pa_pve_str},{result['num_frames']}\n")
        print(f"CSV results saved to {csv_path}")
        print("="*60)
        
        return results_dict
    else:
        print("\n❌ No valid matches found!")
        return None


if __name__ == "__main__":
    # Configuration
    #wham_pkl_path = "/home/yjliu/data0/wham_1/output/demo/output/wham_output.pkl"
    wham_pkl_path = "/home/yjliu/data0/wham_2/output/demo/output/wham_output.pkl"
    root_dir = "/data/yjliu/01_tagging/001_tagging"
    camera_name = "aria01"  # Camera perspective used for GT data
    
    # Pelvis joint indices for H36M-14 format
    pelvis_idxs = [2, 3]
    
    # Optional: specify custom J_regressor path
    j_regressor_path = None  # Will use default from constants
    # j_regressor_path = "data/body_models/J_regressor_h3m.npy"
    
    # 3D Rotation for aria01_rgb camera
    # Set to True if your WHAM predictions are based on rotated aria01 images
    # Set to False if predictions are based on original (non-rotated) images
    apply_rotation = True  # Default: True for aria01, False for other cameras
    
    print("="*60)
    print("WHAM Evaluation with GT SMPL Data (NPZ Format)")
    print("="*60)
    print(f"Camera perspective: {camera_name}")
    print(f"Using H36M-14 joint format")
    print(f"Pelvis joint indices: {pelvis_idxs}")
    print(f"Joints computed from vertices using J_regressor")
    print(f"GT data format: NPZ files per subject (aria02.npz, aria03.npz, etc.)")
    print(f"GT data location: {root_dir}/processed_data/smpl_3d_points/{camera_name}_rgb/")
    print(f"Matching: PA-MPJPE (joint-based)")
    print(f"PA-PVE: Uses joint alignment parameters (CORRECT METHOD)")
    print(f"Visualization: First 3 tracks will be visualized")
    if camera_name == 'aria01' and apply_rotation:
        print(f"⚠️  3D Rotation: ENABLED for {camera_name}_rgb")
        print(f"   GT coordinates will be rotated to match rotated 2D images")
    elif camera_name == 'aria01' and not apply_rotation:
        print(f"ℹ️  3D Rotation: DISABLED")
        print(f"   Using original GT coordinates (no rotation)")
    print("="*60 + "\n")
    
    results = evaluate_wham_output(
        wham_pkl_path, 
        root_dir,
        camera_name=camera_name,
        pelvis_idxs=pelvis_idxs,
        j_regressor_path=j_regressor_path,
        apply_rotation=apply_rotation
    )
