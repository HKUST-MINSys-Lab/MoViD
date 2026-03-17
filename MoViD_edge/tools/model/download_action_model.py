"""
Download pyskl action recognition model and config files
"""
import sys
import urllib.request
from pathlib import Path

# Model URLs from pyskl
MODELS = {
    'posec3d_ntu60_xsub': {
        'config': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/configs/posec3d/slowonly_r50_ntu60_xsub/joint.py',
        'checkpoint': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu60_xsub/joint.pth',
        'label_map': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/tools/data/label_map/nturgbd_60.txt'
    },
    'posec3d_ntu120_xsub': {
        'config': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        'checkpoint': 'https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        'label_map': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/tools/data/label_map/nturgbd_120.txt'
    },
    'stgcn_ntu60_xsub_3d': {
        'config': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.py',
        'checkpoint': 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.pth',
        'label_map': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/tools/data/label_map/nturgbd_60.txt'
    },
    'stgcn++_ntu60_xsub_3d': {
        'config': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py',
        'checkpoint': 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/j.pth',
        'label_map': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/tools/data/label_map/nturgbd_60.txt'
    }
}

def download_file(url, dest_path):
    """Download a file from URL to destination path"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"File already exists: {dest_path}")
        return dest_path
    
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Downloaded successfully: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def main():
    # Default model
    model_name = 'posec3d_ntu60_xsub'
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return
    
    model_info = MODELS[model_name]
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / 'models' / 'action_recognition'
    
    print(f"Downloading model: {model_name}")
    print(f"Destination: {base_dir}")
    
    # Download config
    config_path = base_dir / f'{model_name}_config.py'
    download_file(model_info['config'], config_path)
    
    # Download checkpoint
    checkpoint_path = base_dir / f'{model_name}.pth'
    download_file(model_info['checkpoint'], checkpoint_path)
    
    # Download label map
    label_map_path = base_dir / f'{model_name}_labels.txt'
    download_file(model_info['label_map'], label_map_path)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)
    print(f"\nTo use this model, run:")
    print(f"  python3 real_time.py --video realsense --visualize --max_frames 1000 \\")
    print(f"    --action_config {config_path} \\")
    print(f"    --action_checkpoint {checkpoint_path} \\")
    print(f"    --action_label_map {label_map_path}")

if __name__ == '__main__':
    main()
