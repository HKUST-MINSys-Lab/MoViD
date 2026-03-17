"""
Finetune STGCN++ model for action recognition using 3D skeleton data from MoViD
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Try to add pyskl to path
_pyskl_paths = [
    os.environ.get('PYSKL_PATH', ''),
    str(REPO_ROOT / 'third-party' / 'pyskl'),
    str(REPO_ROOT / 'pyskl'),
    os.path.expanduser('~/pyskl'),
]
for _pyskl_path in _pyskl_paths:
    if _pyskl_path and os.path.exists(_pyskl_path) and _pyskl_path not in sys.path:
        sys.path.insert(0, _pyskl_path)
        break

try:
    import mmcv
    from mmcv.runner import load_checkpoint
    from pyskl.apis import init_recognizer
    from pyskl.datasets import build_dataset, build_dataloader
    from pyskl.models import build_recognizer
    PYSKL_AVAILABLE = True
except ImportError as e:
    logger.error(f"pyskl not available: {e}. Please install pyskl first.")
    sys.exit(1)


class SkeletonDataset(Dataset):
    """
    Dataset for 3D skeleton sequences from MoViD
    """
    def __init__(self, 
                 data_path: str,
                 label_map: Dict[str, int],
                 window_size: int = 100,
                 num_keypoints: int = 25,
                 transform=None):
        """
        Args:
            data_path: Path to pickle file containing skeleton data
                      Format: List of dicts with keys: 'skeleton' (T, 25, 3), 'label' (str)
            label_map: Dictionary mapping label names to class indices
            window_size: Number of frames per sequence
            num_keypoints: Number of keypoints (25 for NTU format)
            transform: Optional transform to apply
        """
        self.window_size = window_size
        self.num_keypoints = num_keypoints
        self.transform = transform
        self.label_map = label_map
        
        # Load data
        logger.info(f"Loading skeleton data from {data_path}")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        logger.info(f"Loaded {len(self.data)} sequences")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        skeleton = item['skeleton']  # (T, 25, 3)
        label_name = item['label']
        
        # Convert label to index
        label_idx = self.label_map.get(label_name, -1)
        if label_idx == -1:
            logger.warning(f"Unknown label: {label_name}, skipping")
            # Return a dummy sample
            skeleton = np.zeros((self.window_size, self.num_keypoints, 3), dtype=np.float32)
            label_idx = 0
        
        # Ensure skeleton has correct shape
        T = skeleton.shape[0]
        if T < self.window_size:
            # Pad with last frame
            padding = np.tile(skeleton[-1:], (self.window_size - T, 1, 1))
            skeleton = np.concatenate([skeleton, padding], axis=0)
        elif T > self.window_size:
            # Uniformly sample
            indices = np.linspace(0, T - 1, self.window_size, dtype=int)
            skeleton = skeleton[indices]
        
        # Reshape to pyskl format: (M=1, T, V, C)
        skeleton = skeleton.reshape(1, self.window_size, self.num_keypoints, 3)
        
        # Apply transform if provided
        if self.transform:
            skeleton = self.transform(skeleton)
        
        return {
            'keypoint': torch.from_numpy(skeleton).float(),
            'label': torch.tensor(label_idx, dtype=torch.long)
        }


def build_model(config_path: str, checkpoint_path: Optional[str] = None, device: str = 'cuda:0'):
    """
    Build STGCN++ model from config
    """
    logger.info(f"Loading model config from {config_path}")
    config = mmcv.Config.fromfile(config_path)
    
    # Build model
    model = build_recognizer(config.model)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location=device)
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, training from scratch")
    
    model = model.to(device)
    model.train()
    
    return model, config


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        keypoints = batch['keypoint'].to(model.device)
        labels = batch['label'].to(model.device)
        
        # Forward pass
        # pyskl models expect annotation dict
        batch_size = keypoints.shape[0]
        outputs = []
        for i in range(batch_size):
            anno = {
                'keypoint': keypoints[i].cpu().numpy(),
                'label': labels[i].item()
            }
            # Model forward expects specific format
            # We need to use the model's forward method directly
            output = model(keypoints[i:i+1], return_loss=False, return_numpy=True)
            outputs.append(output)
        
        # Convert outputs to tensor
        outputs = torch.tensor(np.array(outputs), device=device)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    Validate model
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            keypoints = batch['keypoint'].to(device)
            labels = batch['label'].to(device)
            
            batch_size = keypoints.shape[0]
            outputs = []
            for i in range(batch_size):
                output = model(keypoints[i:i+1], return_loss=False, return_numpy=True)
                outputs.append(output)
            
            outputs = torch.tensor(np.array(outputs), device=device)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Finetune STGCN++ for action recognition')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to pyskl config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (optional)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data pickle file')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data pickle file')
    parser.add_argument('--label_map', type=str, required=True,
                        help='Path to label map file (one label per line)')
    parser.add_argument('--work_dir', type=str, default='./work_dirs/finetune_stgcn',
                        help='Working directory for outputs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Load label map
    logger.info(f"Loading label map from {args.label_map}")
    with open(args.label_map, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    label_map = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)
    logger.info(f"Loaded {num_classes} classes")
    
    # Build model
    model, config = build_model(args.config, args.checkpoint, args.device)
    
    # Create datasets
    train_dataset = SkeletonDataset(
        args.data,
        label_map,
        window_size=100,  # STGCN++ uses 100 frames
        num_keypoints=25
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_data:
        val_dataset = SkeletonDataset(
            args.val_data,
            label_map,
            window_size=100,
            num_keypoints=25
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # Setup optimizer and loss
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        if val_loader:
            val_loss, val_acc = validate(model, val_loader, criterion, args.device)
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(args.work_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, checkpoint_path)
                logger.info(f"Saved best model with val acc: {val_acc:.2f}%")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.work_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        scheduler.step()
    
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
