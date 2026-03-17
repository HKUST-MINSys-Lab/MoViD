import os, sys
import yaml
import torch
from loguru import logger

from configs import constants as _C
from .smpl import SMPL


def build_body_model(device, batch_size=1, gender='neutral', **kwargs):
    sys.stdout = open(os.devnull, 'w')
    body_model = SMPL(
        model_path=_C.BMODEL.FLDR,
        gender=gender,
        batch_size=batch_size,
        create_transl=False).to(device)
    sys.stdout = sys.__stdout__
    return body_model


def build_network(cfg, smpl):
    from .movid import Network
    
    with open(cfg.MODEL_CONFIG, 'r') as f:
        model_config = yaml.safe_load(f)
    model_config.update({'d_feat': _C.IMG_FEAT_DIM[cfg.MODEL.BACKBONE]})
    
    network = Network(smpl, **model_config).to(cfg.DEVICE)
    
    # Load Checkpoint
    if os.path.isfile(cfg.TRAIN.CHECKPOINT):
        logger.info(f"=> Loading checkpoint '{cfg.TRAIN.CHECKPOINT}'")
        checkpoint = torch.load(cfg.TRAIN.CHECKPOINT, map_location='cpu', weights_only=False)

        # --- SAFE WEIGHT LOADING LOGIC ---
        
        # 1. Get the state dictionary from the checkpoint file.
        #    Handle cases where weights are nested under a 'model' key.
        checkpoint_state_dict = checkpoint.get('model', checkpoint)
        
        # 2. Get the state dictionary of your newly created network.
        model_state_dict = network.state_dict()
        
        # 3. Define keys to explicitly ignore.
        ignore_keys = [
            'smpl.body_pose', 'smpl.betas', 'smpl.global_orient', 
            'smpl.J_regressor_extra', 'smpl.J_regressor_eval'
        ]
        
        # 4. Create a new state_dict to hold only the weights that match perfectly.
        state_dict_to_load = {}
        
        # 5. Iterate over the checkpoint's weights and filter them.
        for k, v_checkpoint in checkpoint_state_dict.items():
            # Skip if the key is in our explicit ignore list.
            if k in ignore_keys:
                continue
                
            # Check if the key exists in the current model and if the shapes match.
            if k in model_state_dict:
                v_model = model_state_dict[k]
                if v_checkpoint.shape == v_model.shape:
                    # If name and shape match, add it to our dictionary for loading.
                    state_dict_to_load[k] = v_checkpoint
                else:
                    # If shapes mismatch, log a warning and skip this layer.
                    logger.warning(f"Shape mismatch for '{k}': skipping. "
                                   f"Checkpoint shape: {v_checkpoint.shape}, Model shape: {v_model.shape}")
            # We don't need an else here, as keys not in the current model will be ignored anyway.

        # 6. Load the carefully filtered state_dict.
        #    strict=False is still useful for reporting layers in the model that
        #    were not present in the checkpoint at all.
        incompatible_keys = network.load_state_dict(state_dict_to_load, strict=False)
        if incompatible_keys.missing_keys:
            logger.info(f"The following layers were not found in the checkpoint and were not loaded: {incompatible_keys.missing_keys}")
        
        logger.info(f"=> Successfully loaded compatible weights from checkpoint '{cfg.TRAIN.CHECKPOINT}'")
        
    else:
        logger.warning(f"=> No checkpoint found at '{cfg.TRAIN.CHECKPOINT}'. Training from scratch.")
        
    return network