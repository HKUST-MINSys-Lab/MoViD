import torch
import cv2
import numpy as np
import json
import trt_pose.coco
from torch2trt import TRTModule
from trt_pose.parse_objects import ParseObjects
from collections import defaultdict
import scipy.signal as signal

VIS_THRESH = 0.2
BBOX_CONF = 0.1
TRACKING_THR = 0.1
MINIMUM_FRMAES = 0
MINIMUM_JOINTS = 6

class DetectionModel(object):
    def __init__(self, device='cuda'):
        with open('human_pose.json', 'r') as f:
            human_pose = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.num_parts = len(human_pose['keypoints'])
        self.model = TRTModule()
        self.model.load_state_dict(torch.load('densenet121_baseline_att_256x256_B_epoch_160.pth'))
        self.model.eval()
        self.device = device
        self.WIDTH = 256
        self.HEIGHT = 256
        self.parse_objects = ParseObjects(self.topology)
        self.next_id = 0
        self.frame_id = 0
        self.tracking_results = {
            'id': [],
            'frame_id': [],
            'bbox': [],
            'keypoints': []
        }
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(self.device)

    def compute_bboxes_from_keypoints(self, s_factor=1.2):
        X = self.tracking_results['keypoints'].copy()
        mask = X[..., -1] > VIS_THRESH

        bbox = np.zeros((len(X), 3))
        for i, (kp, m) in enumerate(zip(X, mask)):
            bb = [kp[m, 0].min(), kp[m, 1].min(),
                  kp[m, 0].max(), kp[m, 1].max()]
            cx, cy = [(bb[2]+bb[0])/2, (bb[3]+bb[1])/2]
            bb_w = bb[2] - bb[0]
            bb_h = bb[3] - bb[1]
            s = np.stack((bb_w, bb_h)).max()
            bb = np.array((cx, cy, s))
            bbox[i] = bb
        
        bbox[:, 2] = bbox[:, 2] * s_factor / 200.0
        self.tracking_results['bbox'] = bbox
    

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.WIDTH, self.HEIGHT))
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def track(self, img, fps, length):
        for key in ['id', 'frame_id', 'keypoints','bbox']:
            self.tracking_results[key] = list(self.tracking_results[key][-5:]) if len(self.tracking_results[key]) > 0 else []

        data = self.preprocess(img)
        with torch.no_grad():
            cmap, paf = self.model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        # 只保留第一个人
        if counts[0] > 0:
            obj = objects[0][0]
            kpts = []
            for k in range(self.num_parts):
                idx = obj[k]
                if idx >= 0:
                    peak = peaks[0][k][idx]
                    # peak[0], peak[1] 是归一化坐标
                    y, x = float(peak[0]), float(peak[1])
                    # 还原到原图坐标
                    x_img = x * img.shape[1]
                    y_img = y * img.shape[0]
                    # 取heatmap响应作为置信度
                    heatmap_h, heatmap_w = cmap.shape[2], cmap.shape[3]
                    y_hm = int(round(y * (heatmap_h - 1)))
                    x_hm = int(round(x * (heatmap_w - 1)))
                    # 防止越界
                    y_hm = np.clip(y_hm, 0, heatmap_h - 1)
                    x_hm = np.clip(x_hm, 0, heatmap_w - 1)
                    conf = float(cmap[0, k, y_hm, x_hm])
                    kpts.append([x_img, y_img, conf])
                else:
                    kpts.append([0, 0, 0])
            kpts = np.array(kpts)
            if kpts.shape[0] > 17:
                kpts = kpts[:17]  # 只保留前17个关键点
            # bbox: xyxy
            valid = kpts[:, 2] > 0.1
            # 新增：如果有效关键点数太少，直接跳过该帧
            if valid.sum() < 6:
                self.frame_id += 1
                return  # 跳过该帧
            if valid.sum() > 0:
                x1, y1 = kpts[valid, 0].min(), kpts[valid, 1].min()
                x2, y2 = kpts[valid, 0].max(), kpts[valid, 1].max()
                bbox = np.array([x1, y1, x2, y2])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                scale = max(x2 - x1, y2 - y1) / 200.0  # 与copy保持一致
                bbox = np.array([[cx, cy, scale]])
            else:
                bbox = np.array([[0, 0, 1]])
            
            self.tracking_results['id'].append(0)
            self.tracking_results['frame_id'].append(self.frame_id)
            self.tracking_results['bbox'].append(bbox)
            self.tracking_results['keypoints'].append(kpts)
        self.frame_id += 1

    def process(self, fps):
        for key in ['id', 'frame_id', 'keypoints']:
            self.tracking_results[key] = np.array(self.tracking_results[key])
        #self.compute_bboxes_from_keypoints()     
        bbox_list = self.tracking_results['bbox']
        # 将所有元素flatten成1维，确保形状一致
        bbox_cleaned = [np.array(b).reshape(-1) for b in bbox_list]
        self.tracking_results['bbox'] = np.vstack(bbox_cleaned)   

        output = defaultdict(lambda: defaultdict(list))

        ids = np.unique(self.tracking_results['id'])
        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())

        for _id in ids:
            idxs = np.where(self.tracking_results['id'] == _id)[0]
            for key, val in self.tracking_results.items():
                if key == 'id': continue
                output[_id][key] = val[idxs]
        
        # Smooth bounding box detection
        ids = list(output.keys())
        for _id in ids:
            if len(output[_id]['bbox']) < MINIMUM_FRMAES:
                del output[_id]
                continue
            
            kernel = int(int(fps/2) / 2) * 2 + 1
            if kernel < len(output[_id]['bbox']):
                smoothed_bbox = np.array([signal.medfilt(param, kernel) for param in output[_id]['bbox'].T]).T
                output[_id]['bbox'] = smoothed_bbox

        return output