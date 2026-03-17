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
# Load dataset
# Initialize new_dataset
new_dataset = defaultdict(list)  # Use list as the default container
tt = lambda x: torch.from_numpy(x).float()


dataset = torch.load('/home/yjliu/data0/movid_1/dataset/parsed_data/totalcapture_processed.pth')

for i in range(len(dataset['res'])):
    res_tensor = dataset['res'][i].clone().detach().unsqueeze(0).repeat(len(dataset['kp2d'][i]), 1)
    dataset['res'][i] = res_tensor

for key, value in dataset.items():
    if key == 'vid':
        temp = defaultdict(list)
        for i in range(len(value)):
            # Append the result to new_dataset[key]
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


# Save the new dataset
joblib.dump(new_dataset, 'dataset/parsed_data/totalcapture_train_vit.pth')

# import torch
# import numpy as np
# from collections import defaultdict
# import joblib

# # Initialize the final dataset
# final_dataset = defaultdict(list)
# dataset = torch.load(f'/home/yjliu/data0/MoViD/dataset/parsed_data/freeman_train_smpl_06_vit.pth')
# # Iterate over j = 0 to j = 10
# for j in range(8):  # j from 0 to 10
#     # Load dataset
#     new_dataset = defaultdict(list)
    
#     # Extract the j-th data entry
#     for key, value in dataset[j].items():
#         new_dataset[key] = value

#     # for i in range(len(new_dataset['res'])):
#     #     res_tensor = new_dataset['res'][i].clone().detach().unsqueeze(0).repeat(len(new_dataset['kp2d'][i]), 1)
#     #     new_dataset['res'][i] = res_tensor

#     joblib.dump(new_dataset, f'dataset/parsed_data/freeman_train_smpl_06{j}_vit.pth')
#     # Initialize new_dataset2
#     new_dataset2 = defaultdict(list)
#     unvalid = []  # record invalid indices

#     # Traverse new_dataset and record invalid indices
#     for i in range(len(new_dataset['frame_id'])):
#         if new_dataset['frame_id'][i] is None:
#             unvalid.append(i)

#     # Remove invalid indices and process each key
#     for key, value in new_dataset.items():
#         if key == 'vid':
#             valid_vid = []
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue  # skip invalid indices
#                 # Create a vid tensor and repeat it to match the length of frame_id[i]
#                 vid_tensor = torch.tensor(int(i)).unsqueeze(0).repeat(len(new_dataset['frame_id'][i]), 1)
#                 valid_vid.append(vid_tensor)
#             new_dataset2[key] = torch.cat(valid_vid, dim=0)  # Concatenate valid vid tensors
#         elif key == 'gender':
#             valid_gender = []
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue  # skip invalid indices
#                 # Create a gender tensor and repeat it to match the length of frame_id[i]
#                 gender_tensor = torch.tensor(int(0)).unsqueeze(0).repeat(len(new_dataset['frame_id'][i]), 1)
#                 valid_gender.append(gender_tensor)
#             new_dataset2[key] = torch.cat(valid_gender, dim=0)  # Concatenate valid gender tensors
#         elif key == 'frame_id':
#             # Remove invalid indices
#             new_dataset2[key] = [value[i] for i in range(len(value)) if i not in unvalid]
#         elif key == 'intrinsics':
#             continue  # skip intrinsics
#         else:
#             # Remove invalid indices from the values of other keys and concatenate them
#             for i in range(len(value)):
#                 if i in unvalid:
#                     continue
#                 new_dataset2[key].append(value[i])

#     # Add the 'view' field, using j to indicate the data source
#     num_samples = len(new_dataset2['frame_id'])  # Get the number of valid samples in the current dataset
#     new_dataset2['view'] = torch.tensor([j] * num_samples)  # Add the view field

#     # Merge the current dataset into the final dataset
#     for key, value in new_dataset.items():
#         final_dataset[key].append(value)

# # Merge the final dataset
# for key in final_dataset:
#     # if isinstance(final_dataset[key][0], torch.Tensor):
#     #     final_dataset[key] = torch.cat(final_dataset[key], dim=0)  # Concatenate tensors
#     if isinstance(final_dataset[key][0], list):
#         final_dataset[key] = [item for sublist in final_dataset[key] for item in sublist]  # Flatten the list

# # Save the final dataset
# joblib.dump(final_dataset, 'dataset/parsed_data/freeman_train_smpl_06_all_vit.pth')
