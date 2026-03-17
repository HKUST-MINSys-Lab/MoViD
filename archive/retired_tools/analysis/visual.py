import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import numpy as np

def visualize_skeleton(joints, bones, figsize=(10, 10), title="3D Skeleton Visualization",save_path='skeleton_visualization.png'):
    """
    Visualize a 3D skeleton with joints and bones.
    
    Args:
        joints: torch.Tensor or numpy array of shape (N, 3) containing joint coordinates
        bones: list of tuples containing joint indices to connect
        figsize: tuple of figure dimensions
        title: string for plot title
    """
    # Convert joints to numpy if it's a torch tensor
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy()
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot settings
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Plot joints
    ax.scatter(joints[:,0], joints[:,1], joints[:,2], 
              c='blue', marker='o', s=100, label='Joints')
    
    # Plot bones with validation
    valid_bones = []
    for (j1, j2) in bones:
        if j1 < len(joints) and j2 < len(joints):
            valid_bones.append((j1, j2))
            ax.plot([joints[j1,0], joints[j2,0]],
                   [joints[j1,1], joints[j2,1]],
                   [joints[j1,2], joints[j2,2]], 
                   'r-', linewidth=2)
    
    # Auto-adjust view
    joint_ranges = np.ptp(joints, axis=0)
    center = np.mean(joints, axis=0)
    max_range = joint_ranges.max()
    
    ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    # Add legend
    ax.legend()
    
    # Ensure correct bone connections

    
    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Skeleton visualization saved at {save_path}')
    
    # Report any invalid connections
    invalid_bones = set(bones) - set(valid_bones)
    if invalid_bones:
        print("\nWarning: The following bone connections were invalid:")
        for bone in invalid_bones:
            print(f"Joint indices {bone}")
    
    return fig, ax

# Example usage with the neutral pose
if __name__ == "__main__":

    corrected_bones= [(0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 11), (6, 12), (5, 7), (6, 8),
                    (7, 9), (8, 10),
                    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

    neutral_pose = torch.load('pred_kp3d.pth')[0][0]
    bones = corrected_bones  # Use the corrected bone connections
    fig, ax = visualize_skeleton(neutral_pose, bones,save_path='pred_kp3d.png')

    rotated_pose = torch.load('rotated_kp3d.pth')[0][0]
    fig, ax = visualize_skeleton(rotated_pose, bones,save_path='rotated_kp3d.png')
    
    # gt_kp3d = torch.load('gt_kp3d.pth')[0][0]
    # fig, ax = visualize_skeleton(gt_kp3d, bones,save_path='gt_kp3d.png')


    smpl_bones = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18),
        (18, 19), (19, 20), (20, 21), (21, 22), (22, 23)
    ]
    # smpl_pose = torch.load('init_kp3d_p000457_a000074_0.pth')[0]
    # fig, ax = visualize_skeleton(smpl_pose, bones,save_path='init_kp3d_p000457_a000074_0.png')

    # smpl_pose_gt = torch.load('gt_smpl_kp3d.pth')[0][0]
    # fig, ax = visualize_skeleton(smpl_pose_gt, bones,save_path='gt_smpl_kp3d.png')


