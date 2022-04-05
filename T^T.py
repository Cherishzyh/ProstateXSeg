import os
import torch
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn.functional as F

from MeDIT.ArrayProcess import ExtractBlock
from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Normalize import Normalize01
from MeDIT.Visualization import Imshow3DArray

def CheckDataShape():
    data_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/RoiSlice'
    shape_list = []
    for case in os.listdir(data_folder):
        case_path = os.path.join(data_folder, case)
        data = np.load(case_path)
        shape_list.append(data.shape)
    print(list(set(shape_list)))
# CheckDataShape()

########################################################################################################################


def ImshowData(data_path):

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    _, t2, _ = LoadImage(os.path.join(data_path, 't2.nii'))
    _, roi, _ = LoadImage(os.path.join(data_path, 'roi.nii.gz'))
    print(np.unique(roi))
    print(np.sum((roi == 1).astype(int)), np.sum((roi == 2).astype(int)),
          np.sum((roi == 3).astype(int)), np.sum((roi == 4).astype(int)))

    if len(np.unique(roi)) == 5:
        # Imshow3DArray(Normalize01(t2), roi=[Normalize01((roi == 1).astype(int)), Normalize01((roi == 2).astype(int)),
        #                                     Normalize01((roi == 3).astype(int)), Normalize01((roi == 4).astype(int))])
        Imshow3DArray(Normalize01(t2), roi=Normalize01((roi == 4).astype(int)))


def TestImshowData():
    for case in os.listdir(r'W:\Public Datasets\PROSTATEx_Seg\Seg'):
        ImshowData(os.path.join(r'W:\Public Datasets\PROSTATEx_Seg\Seg', case))
# TestImshowData()


########################################################################################################################
def RoiNumStatistics(case_list, case_folder):
    roi1_sum, roi2_sum, roi3_sum, roi4_sum = 0, 0, 0, 0
    for case in case_list:
        roi_path = os.path.join(case_folder, case+'.npy')
        roi = np.load(roi_path)
        if np.sum(roi[0, ...]) != 0:
            roi1_sum += 1
        if np.sum(roi[1, ...]) != 0:
            roi2_sum += 1
        if np.sum(roi[2, ...]) != 0:
            roi3_sum += 1
        if np.sum(roi[3, ...]) != 0:
            roi4_sum += 1
    print(roi1_sum, roi2_sum, roi3_sum, roi4_sum)
    return roi1_sum, roi2_sum, roi3_sum, roi4_sum


def TestRoiNum():
    roi_folder = r'X:\CNNFormatData\ProstateX_Seg_ZYH\OneSlice\RoiSlice'
    case_name = pd.read_csv(r'X:\CNNFormatData\ProstateX_Seg_ZYH\test_name.csv')
    case_list = case_name.loc[0].tolist()
    RoiNumStatistics(case_list, roi_folder)
# TestRoiNum()


########################################################################################################################
def DownSample():
    data_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/RoiSlice'
    for case in os.listdir(data_folder):
        case_path = os.path.join(data_folder, case)
        roi = np.load(case_path)
        roi_crop, _ = ExtractBlock(roi, (4, 200, 200), center_point=[-1, -1, -1])
        roi1 = F.interpolate(torch.from_numpy(roi_crop[np.newaxis, ...].astype(float)), size=(100, 100), mode='bilinear', align_corners=True)
        roi2 = F.interpolate(torch.from_numpy(roi_crop[np.newaxis, ...].astype(float)), size=(50, 50), mode='bilinear', align_corners=True)
        roi3 = F.interpolate(torch.from_numpy(roi_crop[np.newaxis, ...].astype(float)), size=(25, 25), mode='bilinear', align_corners=True)

        plt.subplot(221)
        plt.imshow(roi_crop[3, ...], cmap='gray')
        plt.subplot(222)
        plt.imshow(roi1.numpy()[0, 3, ...], cmap='gray')
        plt.subplot(223)
        plt.imshow(roi2.numpy()[0, 3, ...], cmap='gray')
        plt.subplot(224)
        plt.imshow(roi3.numpy()[0, 3, ...], cmap='gray')
        plt.show()
# DownSample()


########################################################################################################################
def TestN4ITK():
    from MeDIT.Visualization import Imshow3DArray
    from MeDIT.Normalize import Normalize01
    data_folder = r'W:\Public Datasets\PROMISE12\TrainingData'
    data_list = os.listdir(data_folder)

    mhd_list = [data for data in data_list if '.mhd' in data]
    mhd_list = set([data[:6] for data in mhd_list])
    for case in mhd_list:
        print(case)
        if case == 'Case06':
            continue
        mhd_case = '{}.mhd'.format(case)
        image = sitk.ReadImage(os.path.join(data_folder, mhd_case))
        data = sitk.GetArrayFromImage(image).transpose(1, 2, 0)
        # Imshow3DArray((Normalize01(data)))
        mask_image = sitk.OtsuThreshold(sitk.Cast(image, sitk.sitkFloat32), 0, 1, 200)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output_image = corrector.Execute(sitk.Cast(image, sitk.sitkFloat32), mask_image)

        corrected_data = sitk.GetArrayFromImage(output_image).transpose(1, 2, 0)

        diff_image = data / corrected_data

        Imshow3DArray(np.concatenate((Normalize01(data), Normalize01(corrected_data), Normalize01(np.nan_to_num(diff_image))), axis=1))
        # Imshow3DArray(Normalize01(diff_image))
        print()

        # mhd_seg = '{}_segmentation.mhd'.format(case)
        # seg_image = sitk.ReadImage(os.path.join(data_folder, mhd_seg))
        # seg_data = sitk.GetArrayFromImage(seg_image).transpose(1, 2, 0)


    # raw_list = set([data[:6] for data in data_list if '.raw' in data])
    # for case in raw_list:
    #     raw_case = '{}.raw'.format(case)
    #     raw_seg = '{}_segmentation.raw'.format(case)
    #     imgData = np.fromfile(os.path.join(data_folder, raw_case))
    #     imgDataSeg = np.fromfile(os.path.join(data_folder, raw_seg))
    #     print(imgData.shape)
# TestN4ITK()


########################################################################################################################

def Squeeze():
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResult'
    for case in os.listdir(data_path):
        pred = np.load(os.path.join(data_path, case))
        if np.ndim(pred) == 4:
            pred = np.squeeze(pred)
            np.save(os.path.join(data_path, case), pred)
# Squeeze()

########################################################################################################################


def UnSqueeze():
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/RoiSlice_NoOneHot'
    save_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/RoiSliceNoOneHot'
    for index, case in enumerate(os.listdir(data_path)):
        pred = np.load(os.path.join(data_path, case))
        pred = pred[np.newaxis, ...]
        print('#############   {} / 1727  #############'.format(index+1))
        np.save(os.path.join(save_path, case), pred)

# UnSqueeze()


########################################################################################################################


def KeepLargestROI():
    from PreProcess.DistanceMapNumpy import KeepLargest
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResult'
    for index, case in enumerate(os.listdir(data_path)):
        pred = np.load(os.path.join(data_path, case))
        pred = np.argmax(pred, axis=0)
        _, n_labels, new_pred = KeepLargest(pred)
        # print(n_labels)
        # plt.imshow(pred[0], cmap='gray')
        # plt.show()

        print('#############   {} / 1727  #############'.format(index+1))
        np.save(os.path.join(data_path, case), new_pred[np.newaxis, ...])

# KeepLargestROI()

########################################################################################################################


def ChangeZero2One():
    # data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/DistanceMap/DisMap'
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResult'
    for index, case in enumerate(os.listdir(data_path)):
        pred = np.load(os.path.join(data_path, case))
        if (pred == 0).all():
            # pred = np.ones_like(pred)
            # np.save(os.path.join(data_path, case), pred)
            print('#############{}:   {} / 1727  #############'.format(case, index+1))
# ChangeZero2One()


def Dilate():
    from scipy.ndimage import binary_dilation
    from MeDIT.Normalize import NormalizeZ
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResult'
    save_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResultDilate'
    for index, case in enumerate(os.listdir(data_path)):
        pred = np.squeeze(np.load(os.path.join(data_path, case)))
        dilate_pred = binary_dilation(pred, structure=np.ones((11, 11)))
        # diff = dilate_pred - pred
        # plt.subplot(131)
        # plt.imshow(pred, cmap='gray')
        # plt.subplot(132)
        # plt.imshow(dilate_pred, cmap='gray')
        # plt.subplot(133)
        # plt.imshow(diff, cmap='gray')
        # plt.show()
        np.save(os.path.join(save_path, case), dilate_pred[np.newaxis, ...])
        print('#############{}:   {} / 1727  #############'.format(case, index + 1))
# Dilate()


def Show():
    folder = r'Z:\PM\Prostate301_copy'

    for index, case in enumerate(os.listdir(folder)):
        case_folder = os.path.join(folder, case)
        t2_path = os.path.join(case_folder, 't2.nii')
        roi_path = os.path.join(case_folder, 'ProstateX_UNet.nii.gz')
        _, t2, _ = LoadImage(t2_path, is_show_info=True)
        _, roi, _ = LoadImage(roi_path)

        roi_1 = (roi == 1).astype(int)
        roi_2 = (roi == 2).astype(int)
        roi_3 = (roi == 3).astype(int)
        roi_4 = (roi == 4).astype(int)

        Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi_1), Normalize01(roi_2), Normalize01(roi_3), Normalize01(roi_4)])
# Show()


def MyShow():
    _, t2, _ = LoadImage(r'Z:\PM\t2.nii', is_show_info=True)
    _, roi, _ = LoadImage(r'Z:\PM\ProstateX_UNet.nii.gz', is_show_info=True)
    Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(roi)], axis=1))
# MyShow()


def Copy():
    src_root = r'/home/zhangyihong/Documents/Prostate301'
    des_root = r'/home/zhangyihong/Documents/Prostate301_copy'
    for case in os.listdir(src_root):
        des_folder = os.path.join(des_root, case)

        des_roi_path = os.path.join(des_folder, 'ProstateX_UNet.nii.gz')
        des_t2_path = os.path.join(des_folder, 't2.nii')
        src_roi_path = os.path.join(src_root, '{}/ProstateX_UNet.nii.gz'.format(case))
        src_t2_path = os.path.join(src_root, '{}/t2.nii'.format(case))

        if os.path.exists(src_roi_path) and os.path.exists(src_t2_path):
            if not os.path.exists(des_folder):
                os.mkdir(des_folder)
            shutil.copy(src_roi_path, des_roi_path)
            shutil.copy(src_t2_path, des_t2_path)
            print('********** o(´^｀)o | copying {} **********'.format(case))
        else:
            print('{} have no t2'.format(case))
# Copy()


def Demo():
    case_folder = r'Z:\PM\Prostate301_copy\PM01'

    t2_path = os.path.join(case_folder, 't2.nii')
    # roi_path = os.path.join(case_folder, 'ProstateX_UNet_normal.nii.gz')
    roi_path_flip = os.path.join(case_folder, 'ProstateX_UNet.nii.gz')
    _, t2, _ = LoadImage(t2_path, is_show_info=True)
    _, roi_flip, _ = LoadImage(roi_path_flip)
    # _, roi, _ = LoadImage(roi_path)
    t2_flip = np.flip(t2, axis=1)
    roi_flip = np.flip(roi_flip, axis=1)

    roi_1_flip = (roi_flip == 1).astype(int)
    roi_2_flip = (roi_flip == 2).astype(int)
    roi_3_flip = (roi_flip == 3).astype(int)
    roi_4_flip = (roi_flip == 4).astype(int)

    Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi_1_flip), Normalize01(roi_2_flip), Normalize01(roi_3_flip), Normalize01(roi_4_flip)])
# Demo()


# def ShowNPY():
#     folder = r'Z:\PM\Prostate301_test_npy'
#     for index, case in enumerate(os.listdir(folder)):
#         roi_list = []
#         case_folder = os.path.join(folder, case)
#         t2_path = os.path.join(case_folder, 't2.npy')
#         t2 = np.load(t2_path).transpose([1, 2, 0])
#         for num in range(18):
#             roi_path = os.path.join(case_folder, 'prediction_{}.npy'.format(num))
#             roi_list.append(np.load(roi_path))
#         roi = np.array(roi_list)  #(18, 20, 5, 200, 200)
#         roi = np.squeeze(np.sum(roi, axis=0) / 18)  #(20, 5, 200, 200)
#         roi = np.argmax(roi, axis=1)
#         roi = roi.transpose([1, 2, 0])
#
#         roi_1 = (roi == 1).astype(int)
#         roi_2 = (roi == 2).astype(int)
#         roi_3 = (roi == 3).astype(int)
#         roi_4 = (roi == 4).astype(int)
#
#         Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi_1), Normalize01(roi_2), Normalize01(roi_3), Normalize01(roi_4)])
#     # Imshow3DArray(np.concatenate([Normalize01(t2.transpose([1, 2, 0])), Normalize01(roi.transpose([1, 2, 0]))], axis=1))
# ShowNPY()

# def ShowNPY():
#     folder = r'Z:\PM\Prostate301_test_npy'
#     for index, case in enumerate(os.listdir(folder)):
#         case = 'PM03'
#         case_folder = os.path.join(folder, case)
#         t2_path = os.path.join(case_folder, 't2.npy')
#         t2 = np.load(t2_path).transpose([1, 2, 0])
#         # for num in range(18):
#         roi_path = os.path.join(case_folder, 'prediction_{}.npy'.format(str(0)))
#         roi = np.load(roi_path)
#
#         roi = np.argmax(roi, axis=1)
#         roi = roi.transpose([1, 2, 0])
#
#         roi_1 = (roi == 1).astype(int)
#         roi_2 = (roi == 2).astype(int)
#         roi_3 = (roi == 3).astype(int)
#         roi_4 = (roi == 4).astype(int)
#
#         Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi_1), Normalize01(roi_2), Normalize01(roi_3), Normalize01(roi_4)])


#
# import math
#
# import pylab
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
#
# def gen_data(N):
#     X = np.random.randn(N, 1)
#     w1 = 2.
#     b1 = 8.
#     sigma1 = 1e1  # ground truth
#     Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, 1)
#     w2 = 3
#     b2 = 3.
#     sigma2 = 1e0  # ground truth
#     Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, 1)
#     return X, Y1, Y2
#
#
# class TrainData(Dataset):
#
#     def __init__(self, feature_num, X, Y1, Y2):
#
#         self.feature_num = feature_num
#
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.Y1 = torch.tensor(Y1, dtype=torch.float32)
#         self.Y2 = torch.tensor(Y2, dtype=torch.float32)
#
#     def __len__(self):
#         return self.feature_num
#
#     def __getitem__(self, idx):
#         return self.X[idx,:], self.Y1[idx,:], self.Y2[idx,:]
#
#
# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num, model):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.model = model
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.zeros((task_num)))
#
#     def forward(self, input, targets):
#
#         outputs = self.model(input)
#
#         precision1 = torch.exp(-self.log_vars[0])
#         loss = torch.sum(precision1 * (targets[0] - outputs[0]) ** 2. + self.log_vars[0], -1)
#
#         precision2 = torch.exp(-self.log_vars[1])
#         loss += torch.sum(precision2 * (targets[1] - outputs[1]) ** 2. + self.log_vars[1], -1)
#
#         loss = torch.mean(loss)
#
#         return loss, self.log_vars.data.tolist()
#
#
# class MTLModel(torch.nn.Module):
#     def __init__(self, n_hidden, n_output):
#         super(MTLModel, self).__init__()
#
#         self.net1 = nn.Sequential(nn.Linear(1, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
#         self.net2 = nn.Sequential(nn.Linear(1, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output))
#
#     def forward(self, x):
#         return [self.net1(x), self.net2(x)]


def ShowNii():
    from MeDIT.Visualization import FlattenImages
    from copy import deepcopy
    data_folder = r'X:\PrcoessedData\ProstateX_Seg_ZYH'
    save_folder = r'X:\RawData\ProstateX_Seg_ZYH\Data'
    for case in os.listdir(data_folder):
        case_path = os.path.join(data_folder, case)
        image, data, _ = LoadImage(os.path.join(case_path, 't2_resize.nii'))
        _, roi, _ = LoadImage(os.path.join(case_path, 'roi_resize.nii.gz'), dtype=np.int32)
        print(case, image.GetSpacing(), image.GetSize())
        roi_1 = deepcopy(roi)
        roi_2 = deepcopy(roi)
        roi_3 = deepcopy(roi)
        roi_4 = deepcopy(roi)
        roi_1[roi_1 != 1] = 0
        roi_2[roi_2 != 2] = 0
        roi_3[roi_3 != 3] = 0
        roi_4[roi_4 != 4] = 0
        data_flatten = FlattenImages(data.transpose((2, 0, 1)))
        roi_flatten_1 = FlattenImages(roi_1.transpose((2, 0, 1)))
        roi_flatten_2 = FlattenImages(roi_2.transpose((2, 0, 1)))
        roi_flatten_3 = FlattenImages(roi_3.transpose((2, 0, 1)))
        roi_flatten_4 = FlattenImages(roi_4.transpose((2, 0, 1)))
        plt.figure(figsize=(8, 8), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(data_flatten, cmap='gray', vmin=0.)
        plt.contour(roi_flatten_1, colors='r')
        plt.contour(roi_flatten_2, colors='g')
        plt.contour(roi_flatten_3, colors='b')
        plt.contour(roi_flatten_4, colors='y')
        plt.savefig(os.path.join(save_folder, '{}.jpg'.format(case)), pad_inches=0, dpi=1000)
        plt.close()
# ShowNii()


def ShowNPY():
    from MeDIT.Visualization import FlattenImages
    from copy import deepcopy
    data_folder = r'X:\PrcoessedData\ProstateX_Seg_ZYH\3D'
    save_folder = r'X:\PrcoessedData\ProstateX_Seg_ZYH\3D\Image'
    for index, case in enumerate(os.listdir(os.path.join(data_folder, 'T2Slice'))):
        t2_path = os.path.join(data_folder, 'T2Slice/{}'.format(case))
        t2 = np.load(t2_path).transpose([1, 2, 0])
        roi_path = os.path.join(data_folder, 'RoiSlice/{}'.format(case))
        roi = np.load(roi_path).transpose([1, 2, 0])

        roi_1 = (roi == 1).astype(int)
        roi_2 = (roi == 2).astype(int)
        roi_3 = (roi == 3).astype(int)
        roi_4 = (roi == 4).astype(int)

        data_flatten = FlattenImages(t2.transpose((2, 0, 1)))
        roi_flatten_1 = FlattenImages(roi_1.transpose((2, 0, 1)))
        roi_flatten_2 = FlattenImages(roi_2.transpose((2, 0, 1)))
        roi_flatten_3 = FlattenImages(roi_3.transpose((2, 0, 1)))
        roi_flatten_4 = FlattenImages(roi_4.transpose((2, 0, 1)))
        plt.figure(figsize=(8, 8), dpi=100)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(data_flatten, cmap='gray', vmin=0.)
        plt.contour(roi_flatten_1, colors='r')
        plt.contour(roi_flatten_2, colors='g')
        plt.contour(roi_flatten_3, colors='b')
        plt.contour(roi_flatten_4, colors='y')
        plt.savefig(os.path.join(save_folder, '{}.jpg'.format(case.split('.npy')[0])), pad_inches=0)
        plt.close()
# ShowNPY()


def Count():
    from copy import deepcopy
    data_folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\train_case_name.csv'
    train_list = pd.read_csv(csv_path).values.tolist()[0]
    csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\test_case_name.csv'
    test_list = pd.read_csv(csv_path).values.tolist()[0]
    csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\val_case_name.csv'
    val_list = pd.read_csv(csv_path).values.tolist()[0]

    total_slice, non_slice, pz_slice, tz_slice, u_slice, asf_slice = 0, 0, 0, 0, 0, 0
    for case in os.listdir(data_folder):
        if case in val_list:
            case_path = os.path.join(data_folder, case)
            # image, data, _ = LoadImage(os.path.join(case_path, 't2.nii'))
            _, roi, _ = LoadImage(os.path.join(case_path, 'roi.nii.gz'), dtype=np.int32)
            roi = roi[:, :, 1: -1]
            # print(case, image.GetSpacing(), image.GetSize())
            roi_1 = deepcopy(roi)
            roi_2 = deepcopy(roi)
            roi_3 = deepcopy(roi)
            roi_4 = deepcopy(roi)
            roi_1[roi_1 != 1] = 0 # PZ
            roi_2[roi_2 != 2] = 0 # TZ
            roi_3[roi_3 != 3] = 0 # U
            roi_4[roi_4 != 4] = 0 # AFS
            total_slice += roi.shape[2]
            non_slice += (roi.shape[2] - np.count_nonzero(np.sum(roi, axis=(0, 1))))
            pz_slice += np.count_nonzero(np.sum(roi_1, axis=(0, 1)))
            tz_slice += np.count_nonzero(np.sum(roi_2, axis=(0, 1)))
            u_slice += np.count_nonzero(np.sum(roi_3, axis=(0, 1)))
            asf_slice += np.count_nonzero(np.sum(roi_4, axis=(0, 1)))
    print(total_slice, non_slice, pz_slice, tz_slice, u_slice, asf_slice)
# Count()


# csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\train_name.csv'
# train_list = pd.read_csv(csv_path).values.tolist()
# csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\test_name.csv'
# test_list = pd.read_csv(csv_path).values.tolist()
# csv_path = r'X:\CNNFormatData\ProstateX_Seg_ZYH\val_name.csv'
# val_list = pd.read_csv(csv_path).values.tolist()

# data_1 = sitk.GetArrayFromImage(sitk.ReadImage(r'W:\Public Datasets\PROSTATEx_Seg\Seg\ProstateX-0180\t2.nii'))
# data_2 = sitk.GetArrayFromImage(sitk.ReadImage(r'X:\RawData\ProstateX_Seg_ZYH\Data\ProstateX-0180\t2_resize.nii'))
# data_3 = np.load(r'X:\PrcoessedData\ProstateX_Seg_ZYH\3D\T2Slice\ProstateX-0180.npy')
# plt.hist(data_1.flatten(), bins=50, color='b', alpha=0.3)
# plt.hist(data_2.flatten(), bins=50, color='r', alpha=0.3)
# plt.hist(data_3.flatten(), bins=50, color='y', alpha=0.3)
# plt.show()

# from copy import deepcopy
# from MeDIT.Visualization import FlattenImages
# case_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm'
# t2_path = os.path.join(case_path, 'T2')
# pz_path = os.path.join(case_path, 'PZ')
# # image, data, _ = LoadImage(os.path.join(case_path, 't2.nii'))
# # _, roi, _ = LoadImage(os.path.join(case_path, 'roi.nii.gz'), dtype=np.int32)
# # print(image.GetSpacing(), image.GetSize())
# for case in os.listdir(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/T2'):
#     try:
#         t2 = np.load(os.path.join(t2_path, case))
#         pz = np.load(os.path.join(pz_path, case))
#         plt.figure(figsize=(8, 8), dpi=100)
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
#         plt.margins(0, 0)
#         plt.imshow(t2[1], cmap='gray')
#         plt.contour(pz[0], colors='r')
#         plt.savefig(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/Image', '{}.jpg'.format(case.split('.npy')[0])), pad_inches=0)
#         plt.close()
#     except Exception as e:
#         print(e)


def GetCenter(mask):
    roi_row = np.sum(mask, axis=1)
    roi_column = np.sum(mask, axis=0)

    row = np.nonzero(roi_row)[0]
    column = np.nonzero(roi_column)[0]

    center = [int(np.mean(row)), int(np.mean(column))]
    return center



def SaveFig():
    from MeDIT.SaveAndLoad import LoadImage
    from MeDIT.Visualization import FlattenImages
    from MeDIT.ArrayProcess import ExtractBlock
    from copy import deepcopy
    folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    for case in os.listdir(folder):
        t2_path = os.path.join(folder, '{}/t2.nii'.format(case))
        roi_path = os.path.join(folder, '{}/roi.nii.gz'.format(case))
        _, t2, _ = LoadImage(t2_path)
        _, roi, _ = LoadImage(roi_path)
        roi_asf = deepcopy(roi)
        roi_asf[roi_asf != 1] = 0
        roi_asf[roi == 1] = 1
        center = GetCenter(roi[:, :, roi.shape[-1]//2])
        t2, _ = ExtractBlock(t2, (140, 140, t2.shape[-1]), center_point=(center[0], center[1], roi.shape[-1]//2))
        roi_asf, _ = ExtractBlock(roi_asf, (140, 140, t2.shape[-1]), center_point=(center[0], center[1], roi.shape[-1]//2))
        slice = np.count_nonzero(roi_asf, axis=(0, 1))
        slice_list = [index for index in range(len(slice)) if slice[index] > 0]
        merge_image = FlattenImages(list(t2[:, :, slice_list].transpose(2, 0, 1)))
        merge_roi = FlattenImages(list(roi_asf[:, :, slice_list].transpose(2, 0, 1)))
        plt.figure(0, figsize=(12, 12))
        plt.axis('off')
        plt.imshow(merge_image, cmap='gray')
        plt.contour(merge_roi, colors='r')
        plt.savefig(os.path.join(r'C:\Users\ZhangYihong\Desktop\image_pz', '{}.jpg'.format(case)), bbox_inches='tight', pad_inches=0.05)
        plt.close()
        print(case)
# SaveFig()


pz, tz, dpu, asf = 0, 0, 0, 0
train_name = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/all_train_name.csv', index_col=False).values[0].tolist()
test_name = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/test_name.csv', index_col=False).values[0].tolist()
train_name = [case for case in train_name if '_-_slice0' not in case]
test_name = [case for case in test_name if '_-_slice0' not in case]
for case in os.listdir(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/PZ'):
    if case.split('.npy')[0] in test_name:
        PZ = np.load(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/PZ', case))
        TZ = np.load(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/CZ', case))
        AS = np.load(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/ASF', case))
        DPU = np.load(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm/DPU', case))

        if np.sum(PZ) > 0:
            pz += 1
        if np.sum(TZ) > 0:
            tz += 1
        if np.sum(DPU) > 0:
            dpu += 1
        if np.sum(AS) > 0:
            asf += 1
print(pz, tz, dpu, asf)