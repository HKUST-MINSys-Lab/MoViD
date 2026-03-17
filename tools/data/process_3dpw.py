# # import joblib
# # import os.path as osp
# # import torch
# # import os
# # import datetime

# # def log(msg):
# #     time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# #     print(f"\033[94m[{time_str}]\033[0m {msg}", flush=True)

# # def stack_view_datasets(output_dir, view_indices=[0, 1, 2], output_file='human36m_train_vit_combined.pth'):
# #     """
# #     Stack data from multiple views into a single dataset file.
    
# #     Args:
# #         output_dir: Directory containing view files and where output will be saved
# #         view_indices: List of view indices to combine (default: 0, 1, 2)
# #         output_file: Name of the output combined file
# #     """
# #     # Get the structure from the first view file to initialize
# #     log(f"Examining structure from view {view_indices[0]}...")
# #     first_view_file = osp.join(output_dir, f'humman_train_{view_indices[0]}_vit.pth')
    
# #     if not os.path.exists(first_view_file):
# #         log(f"Error: File {first_view_file} not found!")
# #         return
    
# #     # Get keys from the first dataset to initialize the structure
# #     sample_dataset = joblib.load(first_view_file)
# #     keys = list(sample_dataset.keys())
# #     log(f"Found {len(keys)} keys in the dataset: {keys}")
    
# #     # Initialize the combined dataset structure
# #     combined_dataset = {key: [] for key in keys}
# #     del sample_dataset  # Free memory
    
# #     total_sequences = 0
    
# #     # Process each view
# #     for view_idx in view_indices:
# #         view_file = osp.join(output_dir, f'humman_train_{view_idx}_vit.pth')
        
# #         if not os.path.exists(view_file):
# #             log(f"Warning: File {view_file} not found, skipping view {view_idx}")
# #             continue
            
# #         log(f"Loading view {view_idx} dataset...")
# #         dataset = joblib.load(view_file)
        
# #         # Count sequences in this view
# #         if len(keys) > 0 and keys[0] in dataset:
# #             num_sequences = len(dataset[keys[0]])
# #             log(f"View {view_idx} contains {num_sequences} sequences")
# #             total_sequences += num_sequences
# #         else:
# #             log(f"View {view_idx} appears to be empty or has different structure")
# #             continue
        
# #         # Append data from this view to the combined dataset
# #         for key in keys:
# #             if key in dataset:
# #                 combined_dataset[key].extend(dataset[key])
# #                 log(f"Added {len(dataset[key])} items for key '{key}' from view {view_idx}")
# #             else:
# #                 log(f"Warning: Key '{key}' not found in view {view_idx}")
        
# #         # Free memory
# #         del dataset
# #         log(f"Finished processing view {view_idx}")
    
# #     # Save the combined dataset
# #     output_path = osp.join(output_dir, output_file)
# #     log(f"Saving combined dataset with {total_sequences} total sequences to {output_path}...")
# #     joblib.dump(combined_dataset, output_path)
# #     log("Combined dataset saved successfully!")

# # if __name__ == "__main__":
# #     output_dir = '/home/yjliu/data0/movid_1/dataset/parsed_data/'
    
# #     # Stack the first three views (0, 1, 2)
# #     stack_view_datasets(
# #         output_dir=output_dir,
# #         view_indices=[1, 4, 6, 9],
# #         output_file='humman_train_vit_combined_views1469.pth'
# #     )



import joblib
import torch
import os
import datetime

# def log(msg):
#     time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"\033[94m[{time_str}]\033[0m {msg}", flush=True)

# def stack_or_concatenate_dataset(input_path, output_path, dim=-1):
#     """
#     Load a dataset, stack or concatenate tensors for each key along specified dimension,
#     and save the processed dataset.
    
#     Args:
#         input_path: Path to the input dataset file
#         output_path: Path to save the processed dataset
#         dim: Dimension along which to concatenate or stack tensors
#     """
#     log(f"Loading dataset from {input_path}")
#     dataset = joblib.load(input_path)
#     keys = list(dataset.keys())
#     log(f"Found {len(keys)} keys in dataset: {keys}")
    
#     # Process each key in the dataset
#     for key in keys:
#         try:
#             # Check if this is a list of tensors
#             if isinstance(dataset[key], list) and len(dataset[key]) > 0:
#                 # Get the first element to check its type
#                 first_elem = dataset[key][0]
                
#                 if isinstance(first_elem, torch.Tensor):
#                     log(f"Processing key '{key}' - original shape: {first_elem.shape}")
                    
#                     # Check if all tensors have the same shape except for the concatenation dimension
#                     shapes = [t.shape for t in dataset[key]]
#                     if not all(len(s) == len(shapes[0]) for s in shapes):
#                         log(f"Warning: Elements in '{key}' have different dimensions. Skipping...")
#                         continue
                    
#                     try:
#                         # Try to concatenate along the specified dimension
#                         dataset[key] = torch.cat(dataset[key], dim=dim)
#                         log(f"Concatenated '{key}' along dim={dim}, new shape: {dataset[key].shape}")
#                     except RuntimeError as e:
#                         # If concatenation fails, try stacking instead
#                         log(f"Concatenation failed for '{key}', trying stack operation...")
#                         try:
#                             dataset[key] = torch.stack(dataset[key], dim=0)
#                             log(f"Stacked '{key}', new shape: {dataset[key].shape}")
#                         except RuntimeError:
#                             log(f"Warning: Both concatenate and stack failed for '{key}', keeping as list")
#                 else:
#                     log(f"Key '{key}' contains non-tensor elements, skipping")
#             else:
#                 log(f"Key '{key}' is not a list of tensors, skipping")
#         except Exception as e:
#             log(f"Error processing key '{key}': {e}")
    
#     # Save the processed dataset
#     log(f"Saving processed dataset to {output_path}")
#     joblib.dump(dataset, output_path)
#     log("Dataset saved successfully!")

# if __name__ == "__main__":
#     input_path = '/home/yjliu/data0/movid_1/dataset/parsed_data/humman_train_vit_combined_views1469.pth'
#     output_path = '/home/yjliu/data0/movid_1/dataset/parsed_data/humman_train_vit_combined_views1469_stacked.pth'
    
#     # Process the dataset, concatenating along the last dimension
#     stack_or_concatenate_dataset(input_path, output_path, dim=0)
import torch
import numpy as np
from collections import defaultdict
import joblib
from lib.utils import transforms
def transform_global_orient(global_orient, extrinsics):
    """
    Transform the global orientation using camera extrinsics
    
    Args:
        global_orient (np.ndarray): original global orientation (3,)
        extrinsics (np.ndarray): camera extrinsics matrix (4x4)
    
    Returns:
        np.ndarray: transformed global orientation
    """
    # Convert axis-angle to a rotation matrix
    global_orient_matrix = transforms.axis_angle_to_matrix(global_orient.clone().detach())
    
    # Extract the rotational part of the camera matrix (top-left 3x3 block)
    camera_rot_matrix = extrinsics[:3, :3]
    
    # Compose rotations
    new_global_orient_matrix = camera_rot_matrix @ global_orient_matrix
    
    # Convert the rotation matrix back to axis-angle
    new_global_orient = transforms.matrix_to_axis_angle(new_global_orient_matrix.clone().detach()).numpy()
    
    return new_global_orient
# Load dataset
# Initialize new_dataset
new_dataset = defaultdict(list)  # Use list as the default container
tt = lambda x: torch.from_numpy(x).float()
for view in [1,4,6,9]:

    dataset = joblib.load(f'dataset/parsed_data/humman_train_{view}_vit.pth')

    for key, value in dataset.items():
        if key == 'vid':
            temp = defaultdict(list)
            for i in range(len(value)):
                # Append the result to new_dataset[key]
                temp[key].append(torch.tensor(int(i+len(value)*view)).unsqueeze(0).repeat(len(dataset['kp2d'][i]),1))
            if view == 1:
                new_dataset[key] = torch.cat(temp[key], dim=0)
            else:   
                new_dataset[key] = torch.cat((new_dataset[key],torch.cat(temp[key])))
        elif key == 'gender':
            temp = defaultdict(list)
            for i in range(len(value)):
                temp[key].append(torch.tensor(int(0)).unsqueeze(0).repeat(len(dataset['kp2d'][i]),1))
            if view == 1:
                new_dataset[key] = torch.cat(temp[key], dim=0)
            else:
                new_dataset[key] = torch.cat((new_dataset[key],torch.cat(temp[key])))

        # elif key == 'res':
        #     for i in range(len(value)):
        #         new_dataset2[key].append(value[i].unsqueeze(0).repeat(len(new_dataset['kp2d'][i]),1))
        #     new_dataset2[key] = torch.cat(new_dataset2[key], dim=0)
        elif key == 'frame_id':
            if view == 1:
                new_dataset[key] = value
            else:
                new_dataset[key].append(value)
        elif key == 'intrinsics':
            continue
        else:
            if view == 1:
                new_dataset[key] = torch.from_numpy(np.concatenate(value, axis=0))
            else:
                temp = torch.from_numpy(np.concatenate(value, axis=0))
                new_dataset[key] = torch.from_numpy(np.concatenate((new_dataset[key], temp), axis=0))
    

# Save the new dataset
joblib.dump(new_dataset, 'dataset/parsed_data/humman_train1469_vit.pth')