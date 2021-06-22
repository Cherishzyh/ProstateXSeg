import os
import torch
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch.nn.functional as F

from BasicTool.MeDIT.ArrayProcess import ExtractBlock
from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.Visualization import Imshow3DArray


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
    from BasicTool.MeDIT.Visualization import Imshow3DArray
    from BasicTool.MeDIT.Normalize import Normalize01
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
    from SSHProject.BasicTool.MeDIT.Normalize import NormalizeZ
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

def ShowNPY():
    folder = r'Z:\PM\Prostate301_test_npy'
    for index, case in enumerate(os.listdir(folder)):
        case = 'PM03'
        case_folder = os.path.join(folder, case)
        t2_path = os.path.join(case_folder, 't2.npy')
        t2 = np.load(t2_path).transpose([1, 2, 0])
        # for num in range(18):
        roi_path = os.path.join(case_folder, 'prediction_{}.npy'.format(str(0)))
        roi = np.load(roi_path)

        roi = np.argmax(roi, axis=1)
        roi = roi.transpose([1, 2, 0])

        roi_1 = (roi == 1).astype(int)
        roi_2 = (roi == 2).astype(int)
        roi_3 = (roi == 3).astype(int)
        roi_4 = (roi == 4).astype(int)

        Imshow3DArray(Normalize01(t2), roi=[Normalize01(roi_1), Normalize01(roi_2), Normalize01(roi_3), Normalize01(roi_4)])
# ShowNPY()