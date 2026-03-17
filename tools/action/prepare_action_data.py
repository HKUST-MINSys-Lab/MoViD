"""
Prepare 3D skeleton data from WHAM outputs for action recognition training
Extracts NTU 25 keypoints from WHAM predictions and saves in pyskl format
"""
import os
import sys
import argparse
import numpy as np
import torch
import pickle
import joblib
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs.config import get_cfg_defaults
from lib.models import build_network, build_body_model


def extract_ntu_joints_from_wham_output(wham_output_path: str, 
                                        network,
                                        device: str = 'cuda:0',
                                        window_size: int = 100) -> List[Dict]:
    """
    Extract NTU 25 keypoints from WHAM output files
    
    Args:
        wham_output_path: Path to WHAM output directory or pickle file
        network: WHAM network model
        device: Device to use
        window_size: Number of frames per sequence
        
    Returns:
        List of dicts with keys: 'skeleton' (T, 25, 3), 'label' (str), 'video_id' (str)
    """
    data = []
    
    # Load WHAM output
    if os.path.isdir(wham_output_path):
        # Directory containing multiple output files
        pkl_files = list(Path(wham_output_path).glob('*.pkl'))
        logger.info(f"Found {len(pkl_files)} pickle files in {wham_output_path}")
        
        for pkl_file in tqdm(pkl_files, desc="Processing files"):
            try:
                wham_data = joblib.load(pkl_file)
                sequences = extract_sequences_from_wham_data(wham_data, network, device, window_size)
                data.extend(sequences)
            except Exception as e:
                logger.warning(f"Failed to process {pkl_file}: {e}")
    else:
        # Single pickle file
        logger.info(f"Loading WHAM output from {wham_output_path}")
        wham_data = joblib.load(wham_output_path)
        sequences = extract_sequences_from_wham_data(wham_data, network, device, window_size)
        data.extend(sequences)
    
    return data


def extract_sequences_from_wham_data(wham_data: Dict,
                                     network,
                                     device: str,
                                     window_size: int) -> List[Dict]:
    """
    Extract skeleton sequences from WHAM data
    
    Args:
        wham_data: WHAM output dictionary
        network: WHAM network model
        device: Device to use
        window_size: Number of frames per sequence
        
    Returns:
        List of sequences
    """
    sequences = []
    
    # WHAM output format may vary, try different keys
    if isinstance(wham_data, dict):
        # Check for different possible formats
        if 'frame_outputs' in wham_data:
            # Format from real_time.py
            frame_outputs = wham_data['frame_outputs']
            sequences = extract_from_frame_outputs(frame_outputs, network, device, window_size)
        elif 'results' in wham_data:
            # Format from demo.py
            results = wham_data['results']
            sequences = extract_from_results(results, network, device, window_size)
        else:
            # Try to extract directly
            sequences = extract_direct(wham_data, network, device, window_size)
    
    return sequences


def extract_from_frame_outputs(frame_outputs: List[Dict],
                               network,
                               device: str,
                               window_size: int) -> List[Dict]:
    """
    Extract sequences from frame_outputs format
    """
    sequences = []
    
    # Group frames by subject_id
    subjects = {}
    for frame_output in frame_outputs:
        subject_id = frame_output.get('subject_id', 0)
        if subject_id not in subjects:
            subjects[subject_id] = []
        subjects[subject_id].append(frame_output)
    
    # Extract sequences for each subject
    for subject_id, frames in subjects.items():
        # Sort by frame_idx
        frames = sorted(frames, key=lambda x: x.get('frame_idx', 0))
        
        # Extract NTU joints
        ntu_joints_list = []
        for frame in frames:
            if 'ntu_joints_3d' in frame:
                ntu_joints = frame['ntu_joints_3d']  # (25, 3)
                ntu_joints_list.append(ntu_joints)
            elif 'output' in frame:
                # Try to extract from output
                output = frame['output']
                if 'verts' in output or 'vertices' in output:
                    vertices = output.get('verts', output.get('vertices'))
                    if isinstance(vertices, np.ndarray):
                        vertices = torch.from_numpy(vertices).float().to(device)
                    if len(vertices.shape) == 2:
                        vertices = vertices.unsqueeze(0)  # (1, 6890, 3)
                    
                    # Extract NTU joints
                    with torch.no_grad():
                        ntu_joints = network.smpl.get_ntu_joints(vertices)  # (1, 25, 3)
                        ntu_joints = ntu_joints[0].cpu().numpy()  # (25, 3)
                    ntu_joints_list.append(ntu_joints)
        
        if len(ntu_joints_list) == 0:
            continue
        
        # Create sequences with sliding window
        ntu_joints_array = np.array(ntu_joints_list)  # (T, 25, 3)
        
        # Split into sequences of window_size
        for start_idx in range(0, len(ntu_joints_array) - window_size + 1, window_size // 2):
            sequence = ntu_joints_array[start_idx:start_idx + window_size]
            
            # Get label from frame (if available)
            label = frames[start_idx].get('action_label', 'unknown')
            
            sequences.append({
                'skeleton': sequence,
                'label': label,
                'video_id': f'subject_{subject_id}_seq_{start_idx}'
            })
    
    return sequences


def extract_from_results(results: Dict,
                        network,
                        device: str,
                        window_size: int) -> List[Dict]:
    """
    Extract sequences from results format (from demo.py)
    """
    sequences = []
    
    for subject_id, subject_data in results.items():
        # Extract vertices or joints
        if 'verts' in subject_data:
            vertices = subject_data['verts']  # (T, 6890, 3)
            if isinstance(vertices, np.ndarray):
                vertices = torch.from_numpy(vertices).float().to(device)
            
            # Extract NTU joints for all frames
            with torch.no_grad():
                ntu_joints = network.smpl.get_ntu_joints(vertices)  # (T, 25, 3)
                ntu_joints = ntu_joints.cpu().numpy()
            
            # Split into sequences
            T = ntu_joints.shape[0]
            for start_idx in range(0, T - window_size + 1, window_size // 2):
                sequence = ntu_joints[start_idx:start_idx + window_size]
                sequences.append({
                    'skeleton': sequence,
                    'label': 'unknown',  # Need to provide labels separately
                    'video_id': f'subject_{subject_id}_seq_{start_idx}'
                })
    
    return sequences


def extract_direct(wham_data: Dict,
                  network,
                  device: str,
                  window_size: int) -> List[Dict]:
    """
    Try to extract directly from wham_data
    """
    sequences = []
    logger.warning("Direct extraction not fully implemented, please check data format")
    return sequences


def main():
    parser = argparse.ArgumentParser(description='Prepare action recognition data from WHAM outputs')
    parser.add_argument('--wham_output', type=str, required=True,
                        help='Path to WHAM output directory or pickle file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output pickle file path')
    parser.add_argument('--config', type=str, default='configs/yamls/stage2.yaml',
                        help='Path to WHAM config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to WHAM checkpoint')
    parser.add_argument('--window_size', type=int, default=100,
                        help='Number of frames per sequence')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load WHAM model
    logger.info("Loading WHAM model...")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(args.device, smpl_batch_size)
    network = build_network(cfg, smpl)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        network.load_state_dict(checkpoint['model_state_dict'])
    else:
        network.load_state_dict(checkpoint)
    network.eval()
    network = network.to(args.device)
    
    # Extract NTU joints
    logger.info("Extracting NTU 25 keypoints from WHAM outputs...")
    data = extract_ntu_joints_from_wham_output(
        args.wham_output,
        network,
        args.device,
        args.window_size
    )
    
    logger.info(f"Extracted {len(data)} sequences")
    
    # Save data
    logger.info(f"Saving data to {args.output}")
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
    
    # Print statistics
    labels = [item['label'] for item in data]
    unique_labels = set(labels)
    logger.info(f"Number of unique labels: {len(unique_labels)}")
    logger.info(f"Labels: {sorted(unique_labels)}")
    
    # Count sequences per label
    from collections import Counter
    label_counts = Counter(labels)
    logger.info("Sequences per label:")
    for label, count in label_counts.most_common():
        logger.info(f"  {label}: {count}")


if __name__ == '__main__':
    main()
