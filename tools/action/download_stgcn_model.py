"""
Download pretrained STGCN++ model for action recognition
"""
import urllib.request
from pathlib import Path
from loguru import logger

# STGCN++ model URLs from pyskl
STGCN_MODEL = {
    'config': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py',
    'checkpoint': 'http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/j.pth',
    'label_map': 'https://raw.githubusercontent.com/kennymckormick/pyskl/main/tools/data/label_map/nturgbd_60.txt'
}

def download_file(url, dest_path):
    """Download a file from URL to destination path"""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        logger.info(f"File already exists: {dest_path}")
        return dest_path
    
    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Downloaded successfully: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return None

def main():
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / 'models' / 'action_recognition'
    
    logger.info("Downloading STGCN++ pretrained model...")
    logger.info(f"Destination: {base_dir}")
    
    # Download config
    config_path = base_dir / 'stgcn++_ntu60_xsub_3d_config.py'
    download_file(STGCN_MODEL['config'], config_path)
    
    # Download checkpoint
    checkpoint_path = base_dir / 'stgcn++_ntu60_xsub_3d.pth'
    download_file(STGCN_MODEL['checkpoint'], checkpoint_path)
    
    # Download label map
    label_map_path = base_dir / 'stgcn++_ntu60_xsub_3d_labels.txt'
    download_file(STGCN_MODEL['label_map'], label_map_path)
    
    logger.info("\n" + "="*60)
    logger.info("Download complete!")
    logger.info("="*60)
    logger.info(f"\nTo use this model, run:")
    logger.info(f"  python demo.py --video examples/demo_video.mp4 --save_pkl \\")
    logger.info(f"    --action_config {config_path} \\")
    logger.info(f"    --action_checkpoint {checkpoint_path} \\")
    logger.info(f"    --action_label_map {label_map_path}")
    logger.info(f"\nOr use the convenience script:")
    logger.info(f"  bash scripts/demo/run_demo_with_har.sh examples/demo_video.mp4 output/demo_har")

if __name__ == '__main__':
    main()
