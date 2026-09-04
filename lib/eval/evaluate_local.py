import torch
import numpy as np
from collections import defaultdict

from smplx import SMPL
from loguru import logger
from progress.bar import Bar

from configs import constants as _C
from configs.config import parse_args
from lib.data.dataloader import setup_eval_dataloader
from lib.models import build_network, build_body_model
from lib.eval.eval_utils import (
    compute_error_accel,
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
)
from lib.utils import transforms
from lib.utils.utils import prepare_output_dir, prepare_batch
from lib.utils.imutils import avg_preds


m2mm = 1e3


@torch.no_grad()
def main(cfg, args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    logger.info(f"GPU name -> {torch.cuda.get_device_name()}")
    logger.info(f"GPU feat -> {torch.cuda.get_device_properties('cuda')}")

    eval_set = args.eval_set
    eval_split = args.eval_split
    eval_loader = setup_eval_dataloader(cfg, eval_set, eval_split, cfg.MODEL.BACKBONE)
    logger.info(f"Dataset loaded: {eval_set}_{eval_split}_{cfg.MODEL.BACKBONE}")

    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl_model = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl_model)
    network.eval()

    smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(cfg.DEVICE) for k in ["male", "female", "neutral"]}
    J_regressor_eval = torch.from_numpy(
        np.load(_C.BMODEL.JOINTS_REGRESSOR_H36M)
    )[_C.KEYPOINTS.H36M_TO_J14, :].unsqueeze(0).float().to(cfg.DEVICE)
    pelvis_idxs = [2, 3]

    accumulator = defaultdict(list)
    bar = Bar("Inference", fill="#", max=len(eval_loader))
    for i in range(len(eval_loader)):
        batch = eval_loader.dataset.load_data(i, False)
        if batch is None:
            bar.next()
            continue

        x, inits, features, kwargs, gt = prepare_batch(
            batch,
            cfg.DEVICE,
            cfg.TRAIN.STAGE == "stage2" or cfg.TRAIN.STAGE == "stage3",
        )

        if cfg.FLIP_EVAL:
            flipped_batch = eval_loader.dataset.load_data(i, True)
            f_x, f_inits, f_features, f_kwargs, _ = prepare_batch(
                flipped_batch,
                cfg.DEVICE,
                cfg.TRAIN.STAGE == "stage2" or cfg.TRAIN.STAGE == "stage3",
            )
            flipped_pred = network(f_x, None, f_inits, f_features, **f_kwargs)

        pred = network(x, None, inits, features, **kwargs)

        if cfg.FLIP_EVAL:
            flipped_pose = flipped_pred["pose"].squeeze(0)
            flipped_shape = flipped_pred["betas"].squeeze(0)
            pose = pred["pose"].squeeze(0)
            shape = pred["betas"].squeeze(0)
            flipped_pose = flipped_pose.reshape(-1, 24, 6)
            pose = pose.reshape(-1, 24, 6)
            avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
            avg_pose = avg_pose.reshape(-1, 144)

            network.pred_pose = avg_pose.view_as(network.pred_pose)
            network.pred_shape = avg_shape.view_as(network.pred_shape)
            pred = network.forward_smpl(**kwargs)

        pred_output = smpl["neutral"](
            body_pose=pred["poses_body"],
            global_orient=pred["poses_root_cam"],
            betas=pred["betas"].squeeze(0),
            pose2rot=False,
        )
        pred_verts = pred_output.vertices.cpu()
        pred_j3d = torch.matmul(J_regressor_eval, pred_output.vertices).cpu()

        gender = batch["gender"] if isinstance(batch["gender"], str) else batch["gender"][0]
        target_output = smpl[gender](
            body_pose=transforms.rotation_6d_to_matrix(gt["pose"][0, :, 1:]),
            global_orient=transforms.rotation_6d_to_matrix(gt["pose"][0, :, :1]),
            betas=gt["betas"][0],
            pose2rot=False,
        )
        target_verts = target_output.vertices.cpu()
        target_j3d = torch.matmul(J_regressor_eval, target_output.vertices).cpu()

        pred_j3d, target_j3d, pred_verts, target_verts = batch_align_by_pelvis(
            [pred_j3d, target_j3d, pred_verts, target_verts], pelvis_idxs
        )
        S1_hat = batch_compute_similarity_transform_torch(pred_j3d, target_j3d)
        pa_mpjpe = torch.sqrt(((S1_hat - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
        mpjpe = torch.sqrt(((pred_j3d - target_j3d) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
        pve = torch.sqrt(((pred_verts - target_verts) ** 2).sum(dim=-1)).mean(dim=-1).numpy() * m2mm
        accel = compute_error_accel(joints_pred=pred_j3d, joints_gt=target_j3d)[1:-1] * (30 ** 2)

        summary_string = (
            f"{batch['vid']} | PA-MPJPE: {pa_mpjpe.mean():.1f} "
            f"MPJPE: {mpjpe.mean():.1f} PVE: {pve.mean():.1f}"
        )
        bar.suffix = summary_string
        bar.next()

        accumulator["pa_mpjpe"].append(pa_mpjpe)
        accumulator["mpjpe"].append(mpjpe)
        accumulator["pve"].append(pve)
        accumulator["accel"].append(accel)

    mean_stats = {}
    std_stats = {}
    for k, v in accumulator.items():
        mean_stats[k] = np.concatenate(v).mean()
        std_stats[k] = np.concatenate(v).std()

    print("")
    log_str = f"Evaluation on {eval_set} {eval_split}, "
    log_str += " ".join([f"{k.upper()}: {v:.4f}," for k, v in mean_stats.items()])
    log_str += " ".join([f"{k.upper()}_STD: {v:.4f}," for k, v in std_stats.items()])
    logger.info(log_str)


if __name__ == "__main__":
    cfg, cfg_file, args = parse_args(test=True)
    cfg = prepare_output_dir(cfg, cfg_file)
    main(cfg, args)
