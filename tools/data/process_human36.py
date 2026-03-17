import joblib
import torch
import os.path as osp
import os

import datetime

# def log(msg):
#     time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"\033[94m[{time_str}]\033[0m {msg}", flush=True)


# def split_dataset_low_mem(input_path, output_dir):
#     # Load metadata first to find indices
#     metadata = joblib.load(input_path)
#     frame_id = list(metadata['frame_id'])
#     indices = [i for i, f in enumerate(frame_id) if torch.equal(f, torch.tensor(0))]
#     n_sequences = len(indices)
#     del metadata  # Free memory
    
#     # Initialize file handles for each view
#     view_files = {
#         0: osp.join(output_dir, 'human36m_train_vit_view0.pth'),
#         1: osp.join(output_dir, 'human36m_train_vit_view1.pth'),
#         2: osp.join(output_dir, 'human36m_train_vit_view2.pth'), 
#         3: osp.join(output_dir, 'human36m_train_vit_view3.pth')
#     }
#     for view_file in view_files:
#         joblib.dump({}, view_files[view_file])

#     # Process data in chunks
#     chunk_size = 30  # Adjust based on your memory constraints
#     for chunk_start in range(0, n_sequences, chunk_size):
#         chunk_end = min(chunk_start + chunk_size, n_sequences)
        
#         # Load only the current chunk
#         full_data = joblib.load(input_path)
#         keys = list(full_data.keys())
        
#         # Process each sequence in chunk
#         for seq_idx in range(chunk_start, chunk_end):
#             start = indices[seq_idx]
#             end = indices[seq_idx + 1] if seq_idx + 1 < n_sequences else len(frame_id)
            
#             view_idx = seq_idx % 4
#             seq_data = joblib.load(view_files[view_idx])
#             for key in keys:
#                 # Append the data for the current sequence
#                 if key not in seq_data:
#                     seq_data[key] = []
#                 seq_data[key].append(full_data[key][start:end])
#             # seq_data = {key: full_data[key][start:end] for key in keys}
            
#             # Write immediately to disk
#             joblib.dump(seq_data, view_files[view_idx])
#             log(f"Processed sequence {seq_idx + 1}/{n_sequences} for view {view_idx}")
#             del seq_data
        
#         del full_data  # Free memory after each chunk

# if __name__ == "__main__":
#     input_path = '/data/yjliu/wham/dataset/parsed_data/human36m_train_vit.pth'
#     output_dir = '/data/yjliu/wham/dataset/parsed_data/'
#     split_dataset_low_mem(input_path, output_dir)



# dataset = joblib.load('/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_train_vit_view0_temp.pth')
# # dataset = joblib.load('/home/yjliu/data0/wham_1/dataset/parsed_data/3dpw_test_vit.pth')
# frame_id = list(dataset['frame_id'])

# indices = [i for i, f in enumerate(frame_id) if torch.equal(f, torch.tensor(0))]
# n = len(indices)

# all_keys = dataset.keys()
# # 复制原数据以保留结构
# new_dataset = {key: [] for key in all_keys}

# # 添加结尾索引用于切片
# indices.append(len(dataset['frame_id']))

# # 遍历每段区间
# for i in range(len(indices) - 1):
#     start = indices[i]
#     end = indices[i + 1]
    
#     for key in all_keys:
#         # 按段切片，每段作为 list 元素
#         new_dataset[key].append(dataset[key][start:end])
# # 保存为 joblib 文件（list of dicts）
# joblib.dump(new_dataset, '/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_test_vit.pth')



import os
import joblib
import torch
import datetime
from tqdm import tqdm
import torch.multiprocessing as mp

from configs import constants as _C
from lib.models.smpl import SMPL
from lib.utils import transforms

def log(msg):
    time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\033[94m[{time_str}]\033[0m {msg}", flush=True)

def init_smpl_models():
    """Initialize SMPL models with different genders"""
    log("Initializing SMPL models...")
    smpl_models = {
        'neutral': SMPL(model_path=_C.BMODEL.FLDR),
        'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'),
        'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
    }
    return smpl_models

def process_batch(batch_data, smpl_model, device):
    """Process a batch of data with SMPL model"""
    with torch.no_grad():  # Explicitly disable gradient tracking
        batch_pose = torch.stack([item[0] for item in batch_data['pose']]).to(device)
        batch_betas = torch.stack([item[0] for item in batch_data['betas']]).to(device)
        
        batch_size = batch_pose.shape[0]
        
        # Convert poses to rotation matrices
        init_poses = transforms.axis_angle_to_matrix(batch_pose)
        
        # Reshape for SMPL
        init_global_orients = init_poses[:, 0].reshape(batch_size, 1, 3, 3)
        init_body_poses = init_poses[:, 1:].reshape(batch_size, -1, 3, 3)
        init_shapes = batch_betas.reshape(batch_size, 10)
        
        # Forward pass through SMPL
        pred_outputs = smpl_model.get_output(
            global_orient=init_global_orients.cpu(),
            body_pose=init_body_poses.cpu(),
            betas=init_shapes.cpu(),
            pose2rot=False
        )
        
        # Extract joints and detach from computation graph
        batch_init_kp3d = pred_outputs.joints.detach().cpu()
        batch_init_pose = batch_pose.detach().cpu()
        
    return batch_init_kp3d, batch_init_pose

def process_chunk_and_save(chunk_idx, start_idx, end_idx, data_path, save_path, gpu_id):
    """Process a chunk of the dataset and save directly to file instead of using queue"""
    try:
        # Set device
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        log(f"Chunk {chunk_idx}: Using device {device}")
        
        # Load SMPL model
        smpl_models = init_smpl_models()
        
        # Load dataset
        dataset = joblib.load(data_path)
        chunk_size = end_idx - start_idx
        log(f"Chunk {chunk_idx}: Processing frames {start_idx} to {end_idx-1} ({chunk_size} frames)")
        
        # Initialize outputs for this chunk
        chunk_results = {
            'init_kp3d': [None] * chunk_size,
            'init_pose': [None] * chunk_size
        }
        
        # Process in smaller batches to manage memory
        batch_size = 32  # Adjust based on GPU memory
        for batch_idx, batch_start in enumerate(range(start_idx, end_idx, batch_size)):
            batch_end = min(batch_start + batch_size, end_idx)
            local_start = batch_start - start_idx
            local_end = batch_end - start_idx
            
            # Prepare batch data
            batch_data = {
                'pose': dataset['pose'][batch_start:batch_end],
                'betas': dataset['betas'][batch_start:batch_end]
            }
            
            # Process batch with no gradient tracking
            batch_kp3d, batch_pose = process_batch(batch_data, smpl_models['neutral'], device)
            
            # Store results (in local chunk indices)
            for i, (kp3d, pose) in enumerate(zip(batch_kp3d, batch_pose)):
                if local_start + i < len(chunk_results['init_kp3d']):
                    chunk_results['init_kp3d'][local_start + i] = kp3d.reshape(1, 31, 3)
                    chunk_results['init_pose'][local_start + i] = pose.reshape(1, 24, 3)
            
            if batch_idx % 10 == 0 and batch_start > start_idx:
                log(f"Chunk {chunk_idx}: Processed {batch_start - start_idx}/{chunk_size} frames")
        
        # Save chunk results to a temporary file
        chunk_save_path = f"{save_path}.chunk{chunk_idx}"
        joblib.dump(chunk_results, chunk_save_path)        
        log(f"Chunk {chunk_idx}: Saved results to {chunk_save_path}")
        
        return True
    
    except Exception as e:
        log(f"Error in chunk {chunk_idx}: {e}")
        return False

def process_dataset(data_path, save_path, num_gpus=8):
    """Process entire dataset using multiple GPUs"""
    # Load dataset to get size
    log(f"Loading dataset from {data_path}")
    dataset = joblib.load(data_path)
    total_frames = len(dataset['frame_id'])
    log(f"Dataset contains {total_frames} frames")
    
    # Divide work among GPUs
    frames_per_gpu = total_frames // num_gpus
    chunks = []
    for i in range(num_gpus):
        start_idx = i * frames_per_gpu
        end_idx = (i + 1) * frames_per_gpu if i < num_gpus - 1 else total_frames
        chunks.append((i, start_idx, end_idx))
    
    # Use multiprocessing to distribute work across GPUs
    processes = []
    
    for chunk_idx, start_idx, end_idx in chunks:
        p = mp.Process(
            target=process_chunk_and_save,
            args=(chunk_idx, start_idx, end_idx, data_path, save_path, chunk_idx % num_gpus)
        )
        processes.append(p)
    
    # Start processes
    log(f"Starting {len(processes)} processes on {num_gpus} GPUs")
    for p in processes:
        p.start()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Combine results from all chunks
    log("All processes completed. Combining results...")
    
    # Reload the original dataset for modification
    dataset = joblib.load(data_path)
    
    # Initialize output fields if they don't exist
    if 'init_kp3d' not in dataset:
        dataset['init_kp3d'] = [None] * total_frames
    if 'init_pose' not in dataset:
        dataset['init_pose'] = [None] * total_frames
    
    # Load each chunk and merge
    for chunk_idx, start_idx, end_idx in chunks:
        chunk_save_path = f"{save_path}.chunk{chunk_idx}"
        if os.path.exists(chunk_save_path):
            try:
                chunk_results = joblib.load(chunk_save_path)
                
                # Copy chunk results to the main dataset
                for i in range(len(chunk_results['init_kp3d'])):
                    global_idx = start_idx + i
                    if global_idx < len(dataset['init_kp3d']):
                        dataset['init_kp3d'][global_idx] = chunk_results['init_kp3d'][i]
                        dataset['init_pose'][global_idx] = chunk_results['init_pose'][i]
                
                log(f"Merged chunk {chunk_idx} (frames {start_idx}-{end_idx-1})")
                
                # Clean up temporary file
                os.remove(chunk_save_path)
            except Exception as e:
                log(f"Error loading chunk {chunk_idx}: {e}")
                log(f"Frames {start_idx}-{end_idx-1} may be missing in the final dataset")
    
    # Save final dataset
    log(f"Saving processed dataset to {save_path}")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    joblib.dump(dataset, save_path)
    log(f"Dataset saved successfully to {save_path}")

if __name__ == "__main__":
    # Try to set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set
    
    data_path = '/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_view3_test_vit.pth'
    save_path = '/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_test_vit.pth'
    
    # Process dataset using all available GPUs
    process_dataset(data_path, save_path, num_gpus=8)

    

# import joblib
# import os.path as osp
# import torch
# import os
# import datetime

# def log(msg):
#     time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     print(f"\033[94m[{time_str}]\033[0m {msg}", flush=True)

# def stack_view_datasets(output_dir, view_indices=[0, 1, 2], output_file='human36m_train_vit_combined.pth'):
#     """
#     Stack data from multiple views into a single dataset file.
    
#     Args:
#         output_dir: Directory containing view files and where output will be saved
#         view_indices: List of view indices to combine (default: 0, 1, 2)
#         output_file: Name of the output combined file
#     """
#     # Get the structure from the first view file to initialize
#     log(f"Examining structure from view {view_indices[0]}...")
#     first_view_file = osp.join(output_dir, f'human36m_train_vit_view{view_indices[0]}_temp.pth')
    
#     if not os.path.exists(first_view_file):
#         log(f"Error: File {first_view_file} not found!")
#         return
    
#     # Get keys from the first dataset to initialize the structure
#     sample_dataset = joblib.load(first_view_file)
#     keys = list(sample_dataset.keys())
#     log(f"Found {len(keys)} keys in the dataset: {keys}")
    
#     # Initialize the combined dataset structure
#     combined_dataset = {key: [] for key in keys}
#     del sample_dataset  # Free memory
    
#     total_sequences = 0
    
#     # Process each view
#     for view_idx in view_indices:
#         view_file = osp.join(output_dir, f'human36m_train_vit_view{view_idx}_temp.pth')
        
#         if not os.path.exists(view_file):
#             log(f"Warning: File {view_file} not found, skipping view {view_idx}")
#             continue
            
#         log(f"Loading view {view_idx} dataset...")
#         dataset = joblib.load(view_file)
        
#         # Count sequences in this view
#         if len(keys) > 0 and keys[0] in dataset:
#             num_sequences = len(dataset[keys[0]])
#             log(f"View {view_idx} contains {num_sequences} sequences")
#             total_sequences += num_sequences
#         else:
#             log(f"View {view_idx} appears to be empty or has different structure")
#             continue
        
#         # Append data from this view to the combined dataset
#         for key in keys:
#             if key in dataset:
#                 combined_dataset[key].extend(dataset[key])
#                 log(f"Added {len(dataset[key])} items for key '{key}' from view {view_idx}")
#             else:
#                 log(f"Warning: Key '{key}' not found in view {view_idx}")
        
#         # Free memory
#         del dataset
#         log(f"Finished processing view {view_idx}")
    
#     # Save the combined dataset
#     output_path = osp.join(output_dir, output_file)
#     log(f"Saving combined dataset with {total_sequences} total sequences to {output_path}...")
#     joblib.dump(combined_dataset, output_path)
#     log("Combined dataset saved successfully!")

# if __name__ == "__main__":
#     output_dir = '/home/yjliu/data0/wham_1/dataset/parsed_data/'
    
#     # Stack the first three views (0, 1, 2)
#     stack_view_datasets(
#         output_dir=output_dir,
#         view_indices=[0, 1, 2],
#         output_file='human36m_train_vit_combined_views012.pth'
#     )


# import joblib
# import torch
# import os
# import datetime

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
#     input_path = '/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_view012_train_vit.pth'
#     output_path = '/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_view012_train_vit_stacked.pth'
    
#     # Process the dataset, concatenating along the last dimension
#     stack_or_concatenate_dataset(input_path, output_path, dim=0)