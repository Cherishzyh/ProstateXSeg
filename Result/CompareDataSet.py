import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from random import shuffle
from MeDIT.Normalize import Normalize01
from scipy.stats import ttest_ind, normaltest

ECE_folder = r'/home/zhangyihong/Documents/ProstateECE/OriginalData/ResampleData'
ProstateX_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# plt.figure(figsize=(6, 6))
pixel_list = []
for case in sorted(os.listdir(ProstateX_folder)):
    print(case)
    case_folder = os.path.join(ProstateX_folder, case)
    if os.path.isfile(case_folder): continue
    try:
        t2_image = sitk.ReadImage(os.path.join(case_folder, 't2.nii'))
        t2_data = Normalize01(sitk.GetArrayFromImage(t2_image))
        t2_flatten = torch.from_numpy(t2_data.flatten()).to(device)
        pixel_list.append(t2_flatten)
    except Exception as e:
        print(e)
# plt.hist(torch.cat(pixel_list).tolist(), alpha=0.4, bins=50)

pixel_list_2 = []
ece_list = os.listdir(ECE_folder)
shuffle(ece_list)
for index, case in enumerate(ece_list):
    print(case)
    case_folder = os.path.join(ECE_folder, case)
    if os.path.isfile(case_folder): continue
    t2_image = sitk.ReadImage(os.path.join(case_folder, 't2.nii'))
    t2_data = Normalize01(sitk.GetArrayFromImage(t2_image))
    t2_flatten = torch.from_numpy(t2_data.flatten()).to(device)
    pixel_list_2.append(t2_flatten)
    if index > 80:
        break
#
# plt.hist(torch.cat(pixel_list).tolist(), alpha=0.4, bins=50)
# plt.show()
# plt.close()
# print(normaltest(torch.cat(pixel_list).tolist()))
# print(normaltest(torch.cat(pixel_list_2).tolist()))
print(ttest_ind(torch.cat(pixel_list).tolist(), torch.cat(pixel_list_2).tolist()))