import torch
import numpy as np
import sys
from collections import defaultdict
import joblib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.utils import transforms
# 加载数据集
# 初始化 new_dataset
new_dataset = defaultdict(list)  # 将默认值改为 list
tt = lambda x: torch.from_numpy(x).float()


dataset = torch.load('/home/yjliu/data0/wham_1/dataset/parsed_data/totalcapture_processed.pth')

for i in range(len(dataset['res'])):
    res_tensor = dataset['res'][i].clone().detach().unsqueeze(0).repeat(len(dataset['kp2d'][i]), 1)
    dataset['res'][i] = res_tensor

for key, value in dataset.items():
    if key == 'vid':
        temp = defaultdict(list)
        for i in range(len(value)):
            # 将结果添加到 new_dataset[key] 中
            temp[key].append(torch.tensor(int(i)).unsqueeze(0).repeat(len(dataset['kp2d'][i]),1))
        new_dataset[key] = torch.cat(temp[key], dim=0)

    elif key == 'gender':
        temp = defaultdict(list)
        for i in range(len(value)):
            temp[key].append(torch.tensor(int(0)).unsqueeze(0).repeat(len(dataset['kp2d'][i]),1))
        new_dataset[key] = torch.cat(temp[key], dim=0)


    # elif key == 'res':
    #     for i in range(len(value)):
    #         new_dataset2[key].append(value[i].unsqueeze(0).repeat(len(new_dataset['kp2d'][i]),1))
    #     new_dataset2[key] = torch.cat(new_dataset2[key], dim=0)
    elif key == 'frame_id':
        new_dataset[key] = value

    elif key == 'intrinsics' or key == 'subject' or key == 'action':
        continue
    elif key == 'bone_vectors':
        for i in range(len(value)):
            new_dataset[key].append(np.array(value[i]))
        new_dataset[key] = np.concatenate(new_dataset[key], axis=0)
    else:
        print(key)
        new_dataset[key] = torch.from_numpy(np.concatenate(value, axis=0))


# 保存新数据集
joblib.dump(new_dataset, 'dataset/parsed_data/totalcapture_train_vit.pth')

# import torch
# import numpy as np
# from collections import defaultdict
# import joblib

# # 初始化最终数据集
# final_dataset = defaultdict(list)
# dataset = torch.load(f'/home/yjliu/data0/WHAM/dataset/parsed_data/freeman_train_smpl_06_vit.pth')
# # 遍历 j = 0 到 j = 10
# for j in range(8):  # j 从 0 到 10
#     # 加载数据集
#     new_dataset = defaultdict(list)
    
#     # 提取第 j 个数据
#     for key, value in dataset[j].items():
#         new_dataset[key] = value

#     # for i in range(len(new_dataset['res'])):
#     #     res_tensor = new_dataset['res'][i].clone().detach().unsqueeze(0).repeat(len(new_dataset['kp2d'][i]), 1)
#     #     new_dataset['res'][i] = res_tensor

#     joblib.dump(new_dataset, f'dataset/parsed_data/freeman_train_smpl_06{j}_vit.pth')
#     # 初始化 new_dataset2
#     new_dataset2 = defaultdict(list)
#     unvalid = []  # 记录无效索引

#     # 遍历 new_dataset，记录无效索引
#     for i in range(len(new_dataset['frame_id'])):
#         if new_dataset['frame_id'][i] is None:
#             unvalid.append(i)

#     # 剔除无效索引，并处理每个键
#     for key, value in new_dataset.items():
#         if key == 'vid':
#             valid_vid = []
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue  # 跳过无效索引
#                 # 生成 vid 张量并重复 frame_id[i] 的长度
#                 vid_tensor = torch.tensor(int(i)).unsqueeze(0).repeat(len(new_dataset['frame_id'][i]), 1)
#                 valid_vid.append(vid_tensor)
#             new_dataset2[key] = torch.cat(valid_vid, dim=0)  # 合并有效 vid 张量
#         elif key == 'gender':
#             valid_gender = []
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue  # 跳过无效索引
#                 # 生成 gender 张量并重复 frame_id[i] 的长度
#                 gender_tensor = torch.tensor(int(0)).unsqueeze(0).repeat(len(new_dataset['frame_id'][i]), 1)
#                 valid_gender.append(gender_tensor)
#             new_dataset2[key] = torch.cat(valid_gender, dim=0)  # 合并有效 gender 张量
#         elif key == 'frame_id':
#             # 剔除无效索引
#             new_dataset2[key] = [value[i] for i in range(len(value)) if i not in unvalid]
#         elif key == 'intrinsics':
#             continue  # 跳过 intrinsics
#         else:
#             # 对其他键的值剔除无效索引并合并
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue
#                 new_dataset2[key].append(value[i])

#     # 添加 'view' 字段，用 j 表示数据来源
#     num_samples = len(new_dataset2['frame_id'])  # 获取当前数据集的有效样本数
#     new_dataset2['view'] = torch.tensor([j] * num_samples)  # 添加 view 字段

#     # 将当前数据集合并到最终数据集
#     for key, value in new_dataset.items():
#         final_dataset[key].append(value)

# # 合并最终数据集
# for key in final_dataset:
#     # if isinstance(final_dataset[key][0], torch.Tensor):
#     #     final_dataset[key] = torch.cat(final_dataset[key], dim=0)  # 合并张量
#     if isinstance(final_dataset[key][0], list):
#         final_dataset[key] = [item for sublist in final_dataset[key] for item in sublist]  # 展平列表

# # 保存最终数据集
# joblib.dump(final_dataset, 'dataset/parsed_data/freeman_train_smpl_06_all_vit.pth')
