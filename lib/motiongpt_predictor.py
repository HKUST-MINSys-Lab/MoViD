"""
MotionGPT Action Predictor for MoViD

Converts MoViD output (pose, trans, betas) to text description using MotionGPT3's m2t task.
"""

import os
import sys
import tempfile
import numpy as np
import torch
from loguru import logger

# Set pyrender to use offscreen rendering without display
# This must be done BEFORE importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'egl'

DEFAULT_MOTIONGPT_PATH = os.environ.get(
    "MOTIONGPT_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "MotionGPT3"),
)


class MotionGPTPredictor:
    """Wrapper for MotionGPT3 motion-to-text prediction."""

    def __init__(
        self,
        motiongpt_path: str = DEFAULT_MOTIONGPT_PATH,
        config_path: str = None,
        checkpoint_path: str = None,
        device: str = "cuda",
        smpl_model_path: str = None,
    ):
        """
        Initialize MotionGPT predictor.

        Args:
            motiongpt_path: Path to MotionGPT3 repository
            config_path: Path to config file (default: configs/test_m2t.yaml)
            checkpoint_path: Path to checkpoint (default: from config)
            device: Device to run on
            smpl_model_path: Path to SMPL model for joint conversion
        """
        self.motiongpt_path = motiongpt_path
        self.device = device

        # Set default SMPL path - try to find it in the current project
        if smpl_model_path is None:
            # Try MoViD's SMPL path first
            movid_smpl = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "body_models", "smpl")
            if os.path.exists(movid_smpl):
                smpl_model_path = movid_smpl
            else:
                # Fallback to MotionGPT3's deps/smpl
                smpl_model_path = os.path.join(motiongpt_path, "deps", "smpl")
        self.smpl_model_path = smpl_model_path

        # Add MotionGPT3 to path
        if motiongpt_path not in sys.path:
            sys.path.insert(0, motiongpt_path)

        # Default paths
        if config_path is None:
            config_path = os.path.join(motiongpt_path, "configs/test_m2t.yaml")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(motiongpt_path, "checkpoints/motiongpt3.ckpt")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Load model
        self._load_model()

        # Load normalization stats
        self._load_stats()

    def _load_model(self):
        """Load MotionGPT3 model."""
        import pytorch_lightning as pl
        from omegaconf import OmegaConf
        from motGPT.config import parse_args
        from motGPT.data.build_data import build_data
        from motGPT.models.build_model import build_model

        # Save current directory and change to MotionGPT3 directory
        # (MotionGPT3 uses relative paths like ./configs/assets.yaml and deps/mot-gpt2)
        original_cwd = os.getcwd()
        os.chdir(self.motiongpt_path)

        try:
            # Parse config
            sys.argv = ['demo.py', '--cfg', self.config_path]
            self.cfg = parse_args(phase="demo")

            # Set FOLDER_EXP (normally set by create_logger, but we skip it for demo)
            from omegaconf import OmegaConf
            OmegaConf.set_struct(self.cfg, False)
            self.cfg.FOLDER_EXP = os.path.join(self.motiongpt_path, "demo_output")
            os.makedirs(self.cfg.FOLDER_EXP, exist_ok=True)
            OmegaConf.set_struct(self.cfg, True)

            # Set device
            if self.device == "cuda":
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in self.cfg.DEVICE)
                self.torch_device = torch.device("cuda")
            else:
                self.torch_device = torch.device("cpu")

            # Set random seed for reproducibility (same as demo.py)
            import pytorch_lightning as pl
            seed_value = getattr(self.cfg, 'SEED_VALUE', 1234)
            pl.seed_everything(seed_value)

            # Build datamodule (for normalization)
            # For demo mode, we use DummyDataModule to avoid dataset loading issues
            logger.info("Using demo-only datamodule for MotionGPT.")
            self.datamodule = self._create_dummy_datamodule()

            # Build model (this requires relative paths like deps/mot-gpt2)
            self.model = build_model(self.cfg, self.datamodule).eval()

            # Load checkpoint
            if os.path.exists(self.checkpoint_path):
                logger.info(f"Loading MotionGPT checkpoint from {self.checkpoint_path}")
                state_dict = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
                self.model.load_state_dict(state_dict, strict=False)
            else:
                logger.warning(f"Checkpoint not found: {self.checkpoint_path}")

            self.model.to(self.torch_device)
            logger.info("MotionGPT model loaded successfully")
        finally:
            # Restore original working directory
            os.chdir(original_cwd)

    def _create_dummy_datamodule(self):
        """Create minimal datamodule for demo."""
        import numpy as np
        import torch
        from os.path import join as pjoin
        from motGPT.data.humanml.scripts.motion_process import recover_from_ric
        from motGPT.data.humanml.utils.word_vectorizer import WordVectorizer
        from argparse import Namespace

        class DummyDataModule:
            def __init__(self, cfg, motiongpt_path):
                self.cfg = cfg
                self.name = "humanml3d"
                self.njoints = 22
                self.fps = 20

                # Load stats
                mean_path = pjoin(motiongpt_path, "assets", "meta", "mean.npy")
                std_path = pjoin(motiongpt_path, "assets", "meta", "std.npy")

                self._mean = np.load(mean_path)
                self._std = np.load(std_path)
                self._mean_t = torch.tensor(self._mean)
                self._std_t = torch.tensor(self._std)
                self._recover_from_ric = recover_from_ric

                # Create hparams with w_vectorizer for M2TMetrics
                self.hparams = Namespace()
                self.hparams.mean = self._mean
                self.hparams.std = self._std

                # Load word vectorizer
                glove_path = pjoin(motiongpt_path, "deps", "glove")
                try:
                    self.hparams.w_vectorizer = WordVectorizer(glove_path, "our_vab")
                except Exception as e:
                    logger.warning(f"Failed to load word vectorizer: {e}")
                    self.hparams.w_vectorizer = None

            def normalize(self, features):
                mean = self._mean_t.to(features)
                std = self._std_t.to(features)
                return (features - mean) / std

            def denormalize(self, features):
                mean = self._mean_t.to(features)
                std = self._std_t.to(features)
                return features * std + mean

            def feats2joints(self, features):
                feats_dn = self.denormalize(features)
                return self._recover_from_ric(feats_dn, self.njoints)

        return DummyDataModule(self.cfg, self.motiongpt_path)

    def _load_stats(self):
        """Load HumanML3D normalization stats."""
        mean_path = os.path.join(self.motiongpt_path, "assets", "meta", "mean.npy")
        std_path = os.path.join(self.motiongpt_path, "assets", "meta", "std.npy")

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)

    def pose_to_hml263(self, pose, trans, betas, feet_thre=0.002):
        """
        Convert MoViD pose parameters to HumanML3D 263-dim features.

        Args:
            pose: (T, 72) SMPL pose parameters
            trans: (T, 3) translation
            betas: (T, 10) or (10,) shape parameters
            feet_thre: foot contact threshold

        Returns:
            features: (T-1, 263) HumanML3D features
        """
        from smplx import SMPL
        from motGPT.data.humanml.scripts import motion_process as mp
        from motGPT.data.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain
        from motGPT.data.humanml.common.skeleton import Skeleton

        T = pose.shape[0]

        # Handle betas shape
        if betas.ndim == 1:
            betas = np.tile(betas[None, :], (T, 1))

        # Get joints from SMPL
        smpl = SMPL(
            model_path=self.smpl_model_path,
            gender='neutral',
            batch_size=T,
        ).to('cpu')

        with torch.no_grad():
            output = smpl(
                global_orient=torch.from_numpy(pose[:, :3]).float(),
                body_pose=torch.from_numpy(pose[:, 3:]).float(),
                betas=torch.from_numpy(betas).float(),
                transl=torch.from_numpy(trans).float(),
            )

        # Get first 22 joints
        joints3d = output.joints[:, :22, :].numpy()

        # SMPL uses Y-down, HumanML3D expects Y-up
        joints3d[:, :, 1] = -joints3d[:, :, 1]

        # MoViD outputs camera-relative coordinates where person faces camera (-Z)
        # HumanML3D expects person to face +Z, so flip Z axis
        joints3d[:, :, 2] = -joints3d[:, :, 2]

        # Convert to HumanML3D features
        mp.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
        mp.kinematic_chain = t2m_kinematic_chain

        positions_t = torch.from_numpy(joints3d).float()
        tgt_skel = Skeleton(mp.n_raw_offsets, mp.kinematic_chain, "cpu")
        tgt_offsets = tgt_skel.get_offsets_joints(positions_t[0])

        feats, *_ = mp.process_file(
            positions_t,
            feet_thre=feet_thre,
            tgt_offsets=tgt_offsets,
            kinematic_chain=mp.kinematic_chain,
            src_skel=None,
        )

        return np.asarray(feats, dtype=np.float32)

    def predict_from_features(self, features, chunk_size=100):
        """
        Predict action description from HumanML3D features.

        Args:
            features: (T, 263) HumanML3D features
            chunk_size: Number of frames per chunk (default 100 = 5 seconds at 20fps)

        Returns:
            list of dict with 'start_time', 'end_time', 'description'
        """
        from motGPT.data.utils import collate_tensors

        T = features.shape[0]
        fps = 20
        results = []

        # Split into chunks and collect all valid chunks
        num_chunks = (T + chunk_size - 1) // chunk_size
        chunk_infos = []  # (start, end, chunk_len)
        chunk_feats_list = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, T)
            chunk_len = end - start

            # Skip very short chunks
            if chunk_len < 20:
                continue

            chunk_feats = features[start:end]
            feats_tensor = torch.tensor(chunk_feats, device=self.torch_device)
            feats_normalized = self.datamodule.normalize(feats_tensor)

            chunk_infos.append((start, end, chunk_len))
            chunk_feats_list.append(feats_normalized)

        if not chunk_feats_list:
            return results

        # Process all chunks as a single batch (same as demo.py / run_m2t_e2e.sh)
        # This ensures consistent results with batch normalization
        motion_feats = collate_tensors(chunk_feats_list)
        motion_lengths = [info[2] for info in chunk_infos]

        # Prepare input texts
        input_motion_holder_seq = self.model.lm.input_motion_holder_seq
        texts = []
        for _ in chunk_infos:
            text = "Describe the motion represented by <Motion_Placeholder> using plain English."
            text = text.replace('<Motion_Placeholder>', input_motion_holder_seq)
            texts.append(text)

        batch = {
            "length": motion_lengths,
            "text": texts,
            "motion": motion_feats,
        }

        # Get motion tokens for all chunks
        result = self.model.lm.motion_feats_to_tokens(
            self.model.vae,
            motion_feats,
            motion_lengths,
            modes='m2t'
        )
        if isinstance(result, tuple):
            motion_tokens_input, _ = result
        else:
            motion_tokens_input = result
        batch['motion_tokens_input'] = motion_tokens_input

        # Run inference on all chunks at once
        with torch.no_grad():
            outputs = self.model(batch, task='m2t')

        gen_texts = outputs['texts']

        # Collect results
        for i, (start, end, chunk_len) in enumerate(chunk_infos):
            description = gen_texts[i] if i < len(gen_texts) else ""
            results.append({
                'start_frame': start,
                'end_frame': end,
                'start_time': start / fps,
                'end_time': end / fps,
                'description': description.strip(),
            })

        return results

    def predict_from_movid(self, movid_result, subject_id=0, chunk_size=100):
        """
        Predict action description from MoViD output.

        Args:
            movid_result: dict with 'pose', 'trans', 'betas' keys
            subject_id: subject ID in MoViD output
            chunk_size: frames per chunk

        Returns:
            list of dict with predictions
        """
        if isinstance(movid_result, dict) and subject_id in movid_result:
            data = movid_result[subject_id]
        else:
            data = movid_result

        pose = data['pose']
        trans = data['trans']
        betas = data['betas']

        # Convert to HumanML3D features
        features = self.pose_to_hml263(pose, trans, betas)

        # Predict
        return self.predict_from_features(features, chunk_size=chunk_size)

    def summarize_to_action(self, text):
        """
        Summarize description text to a short action label.

        Args:
            text: Description text

        Returns:
            Short action label
        """
        import re

        text_lower = text.lower().strip().strip('"').strip("'")

        # Pattern matching for common actions
        patterns = [
            (r'lift.*(arm|hand)|raise.*(arm|hand)|arm.*up', 'lift arms'),
            (r'lower.*(arm|hand)|put.*(arm|hand).*down', 'lower arms'),
            (r'wave|waving', 'wave'),
            (r'stretch', 'stretch'),
            (r'reach', 'reach'),
            (r'kick', 'kick'),
            (r'jump', 'jump'),
            (r'hop', 'hop'),
            (r'squat', 'squat'),
            (r'walk.*backward|walk.*back', 'walk backward'),
            (r'walk.*forward', 'walk forward'),
            (r'walk.*circle', 'walk in circle'),
            (r'walk', 'walk'),
            (r'run', 'run'),
            (r'jog', 'jog'),
            (r'turn.*around', 'turn around'),
            (r'turn.*left', 'turn left'),
            (r'turn.*right', 'turn right'),
            (r'turn', 'turn'),
            (r'spin', 'spin'),
            (r'bend.*over|bend.*down', 'bend over'),
            (r'bend', 'bend'),
            (r'lean', 'lean'),
            (r'crouch', 'crouch'),
            (r'kneel', 'kneel'),
            (r'sit.*down', 'sit down'),
            (r'sit', 'sit'),
            (r'stand.*up', 'stand up'),
            (r'stand', 'stand'),
            (r'throw', 'throw'),
            (r'catch', 'catch'),
            (r'pick.*up', 'pick up'),
            (r'put.*down|place', 'put down'),
            (r'grab|hold', 'grab'),
            (r'push', 'push'),
            (r'pull', 'pull'),
            (r'danc', 'dance'),
            (r'exercise', 'exercise'),
            (r'punch', 'punch'),
            (r'clap', 'clap'),
            (r'balance', 'balance'),
            (r'stumbl|stagger', 'stumble'),
            (r'fall', 'fall'),
            (r'nod', 'nod'),
            (r'shake.*head', 'shake head'),
            (r'bow', 'bow'),
            (r'lay|lying|lie', 'lay down'),
            (r'move', 'move'),
        ]

        for pattern, label in patterns:
            if re.search(pattern, text_lower):
                return label

        return 'motion'
