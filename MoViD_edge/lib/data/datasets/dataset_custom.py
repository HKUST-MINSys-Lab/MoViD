from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from ..utils.normalizer import Normalizer
from ...models import build_body_model
from ...utils import transforms
from ...utils.imutils import compute_cam_intrinsics
from ...utils.kp_utils import root_centering

KEYPOINTS_THR = 0.3


def convert_dpvo_to_cam_angvel(traj, fps):
    """Convert DPVO trajectory output to camera angular velocity."""

    quat = traj[:, 3:]
    quat = quat[:, [3, 0, 1, 2]]

    # Quaternion stores camera-to-world rotation. Convert it to world-to-camera.
    world2cam = transforms.quaternion_to_matrix(torch.from_numpy(quat)).float()
    rotation = world2cam.mT

    cam_angvel = transforms.matrix_to_axis_angle(
        rotation[:-1] @ rotation[1:].transpose(-1, -2)
    )
    cam_angvel = transforms.matrix_to_rotation_6d(
        transforms.axis_angle_to_matrix(cam_angvel)
    )

    cam_angvel = cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]]).to(cam_angvel)
    cam_angvel = cam_angvel * fps
    cam_angvel = torch.cat((cam_angvel, cam_angvel[:1]), dim=0)
    return cam_angvel


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tracking_results, slam_results, width, height, fps):
        self.tracking_results = tracking_results
        self.slam_results = slam_results
        self.width = width
        self.height = height
        self.fps = fps
        self.res = torch.tensor([width, height]).float()
        self.intrinsics = compute_cam_intrinsics(self.res)

        self.device = cfg.DEVICE.lower()

        self.smpl = build_body_model("cpu")
        self.keypoints_normalizer = Normalizer(cfg)

        self._to = lambda x: x.unsqueeze(0).to(self.device)

    def __len__(self):
        return len(self.tracking_results.keys())

    def load_data(self, index, flip=False):
        self.prefix = "flipped_" if flip else ""
        return self.__getitem__(index)

    def __getitem__(self, _index):
        if _index >= len(self):
            return None

        index = sorted(list(self.tracking_results.keys()))[_index]

        kp2d = torch.from_numpy(self.tracking_results[index][self.prefix + "keypoints"]).float()
        mask = kp2d[..., -1] < KEYPOINTS_THR
        bbox = torch.from_numpy(self.tracking_results[index][self.prefix + "bbox"]).float()

        norm_kp2d, _ = self.keypoints_normalizer(
            kp2d[..., :-1].clone(),
            self.res,
            self.intrinsics,
            224,
            224,
            bbox,
        )

        features = self.tracking_results[index][self.prefix + "features"]

        init_output = self.smpl.get_output(
            global_orient=self.tracking_results[index][self.prefix + "init_global_orient"],
            body_pose=self.tracking_results[index][self.prefix + "init_body_pose"],
            betas=self.tracking_results[index][self.prefix + "init_betas"],
            pose2rot=False,
            return_full_pose=True,
        )
        init_kp3d = root_centering(init_output.joints[:, :17], "coco")
        init_kp = torch.cat(
            (init_kp3d.reshape(1, -1), norm_kp2d[0].clone().reshape(1, -1)),
            dim=-1,
        )
        init_smpl = transforms.matrix_to_rotation_6d(init_output.full_pose)
        init_root = transforms.matrix_to_rotation_6d(init_output.global_orient)

        cam_angvel = convert_dpvo_to_cam_angvel(self.slam_results, self.fps)

        return (
            index,
            self._to(norm_kp2d),
            (self._to(init_kp), self._to(init_smpl)),
            self._to(features),
            self._to(mask),
            init_root.to(self.device),
            self._to(cam_angvel),
            self.tracking_results[index]["frame_id"],
            {
                "cam_intrinsics": self._to(self.intrinsics),
                "bbox": self._to(bbox),
                "res": self._to(self.res),
            },
        )
