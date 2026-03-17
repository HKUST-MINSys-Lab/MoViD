import joblib
import torch
import os.path as osp
import os
import sys

import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from configs import constants as _C
from lib.models.smpl import SMPL
from lib.utils import transforms
smpl = {
    'neutral': SMPL(model_path=_C.BMODEL.FLDR),
    'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'),
    'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
}
kp3d = {}
for view_id in range(4):
    dataset = joblib.load(f'/home/yjliu/data0/wham_1/dataset/parsed_data/human36m_train_vit_view{view_id}_temp.pth')
    dataset['init_kp3d'] = []
    dataset['init_pose'] = []

    i=0
    init_pose = transforms.axis_angle_to_matrix(dataset['pose'][i][0])
    init_global_orient = init_pose[0].reshape(1,-1, 3, 3)
    init_body_pose = init_pose[1:].reshape(1,-1, 3, 3)
    init_shape = dataset['betas'][i][0]
    init_shape = init_shape.reshape(1, 10)
    pred_output = smpl['neutral'].get_output(global_orient=init_global_orient.cpu(),
                                            body_pose=init_body_pose.cpu(),
                                            betas=init_shape.cpu(),
                                            pose2rot=False)
    kp3d[view_id] = pred_output.joints


torch.save(kp3d, '/home/yjliu/data0/wham_1/kp3d_h36m_view.pth')
