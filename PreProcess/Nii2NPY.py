import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk

from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Normalize import Normalize01, NormalizeZ
from MeDIT.Visualization import Imshow3DArray
from MIP4AIM.NiiProcess.Resampler import Resampler


def CropData(t2_data, crop_shape, center=[], is_roi=False):
    from MeDIT.ArrayProcess import ExtractPatch

    # Crop
    t2_crop, _ = ExtractPatch(t2_data, crop_shape, center_point=center)

    if not is_roi:
        # Normalization
        t2_crop -= np.mean(t2_crop)
        t2_crop /= np.std(t2_crop)

    return t2_crop


def Crop3DData(t2_data, crop_shape, is_roi=False):
    from MeDIT.ArrayProcess import ExtractBlock
    # CropD
    t2_crop, _ = ExtractBlock(t2_data, crop_shape)

    if not is_roi:
        # Normalization
        t2_crop -= np.mean(t2_crop)
        t2_crop /= np.std(t2_crop)

    return t2_crop
# def GetCenter(roi):
#     roi_row = []
#     roi_column = []
#     for row in range(roi.shape[0]):
#         roi_row.append(np.sum(roi[row, ...]))
#     for column in range(roi.shape[1]):
#         roi_column.append(np.sum(roi[..., column]))
#
#     max_row = max(roi_row)
#     max_column = max(roi_column)
#     row_index = roi_row.index(max_row)
#     column_index = roi_column.index(max_column)
#
#     column = np.argmax(roi[row_index])
#     row = np.argmax(roi[..., column_index])
#     # center = [int(column + max_row // 2), int(row + max_column // 2)]
#     center = [int(row + max_column // 2), int(column + max_row // 2)]
#     return center


# def GetROICenter(roi):
#     '''
#     :param roi: 2D roi include multi-class
#     :return: the center of entire roi
#     '''
#
#     assert len(roi.shape) == 2
#     roi_binary = (roi >= 1).astype(int)
#     center = GetCenter(roi_binary)
#     return center


# def ShapeofROI(data_root):
#     length_list = []
#     width_list = []
#     for case in os.listdir(data_root):
#         print(case)
#         case_path = os.path.join(data_root, case)
#         roi_path = os.path.join(case_path, 'roi.nii.gz')
#         _, roi, _ = LoadImage(roi_path, dtype=np.int)
#         roi = (roi >= 1).astype(int)
#         for slice in range(roi.shape[-1]):
#             _, shape = GetCenter(roi[..., slice])
#             length_list.append(shape[3]-shape[2])
#             width_list.append(shape[1]-shape[0])
#     print(max(length_list), max(width_list))


def ROIOneHot(roi, roi_class=(0, 1, 2, 3, 4)):
    '''
    :param roi:
    :return:
    '''
    roi_list = []
    for index in roi_class:
        roi_list.append((roi == index).astype(int))
    return np.array(roi_list, dtype=np.int32)

    # roi_class = [0, 1, 2, 3, 4]
    # if len(roi.shape) == 2:
    #     for index in roi_class:
    #         roi_list.append((roi == index).astype(int))
    #     return np.array(roi_list, dtype=np.int32)
    # elif len(roi.shape) == 3:
    #     roi_list_3d = []
    #     for slice in range(roi.shape[0]):
    #         roi_list = []
    #         for index in roi_class:
    #             roi_list.append((roi[slice] == index).astype(int))
    #         roi_list_3d.append(roi_list)
    #     return np.array(roi_list_3d, dtype=np.int32)


def ResizeSipmleITKImage(image, is_roi=False, expected_resolution=None, store_path=''):
    if is_roi:
        shape = (image.GetSize()[0], image.GetSize()[1], image.GetSize()[2])
        resolution = (image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[2])
    else:
        shape = (image.GetSize()[0], image.GetSize()[1], image.GetSize()[2])
        resolution = (image.GetSpacing()[0], image.GetSpacing()[1], image.GetSpacing()[2])

    expected_resolution = list(expected_resolution)
    dim_0, dim_1, dim_2 = False, False, False
    if expected_resolution[0] < 1e-6:
        expected_resolution[0] = resolution[0]
        dim_0 = True
    if expected_resolution[1] < 1e-6:
        expected_resolution[1] = resolution[1]
        dim_1 = True
    if expected_resolution[2] < 1e-6:
        expected_resolution[2] = resolution[-1]
    expected_shape = [int(raw_resolution * raw_size / dest_resolution) for
                      dest_resolution, raw_size, raw_resolution in
                      zip(expected_resolution, shape, resolution)]
    if dim_0: expected_shape[0] = shape[0]
    if dim_1: expected_shape[1] = shape[1]
    if dim_2: expected_shape[2] = shape[2]
    expected_shape = tuple(expected_shape)

    transform = sitk.Transform()
    transform.SetIdentity()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(expected_resolution)
    resample_filter.SetSize(expected_shape)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputOrigin(image.GetOrigin())
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputPixelType(sitk.sitkFloat32)

    if is_roi:
        # resample_filter.SetOutputPixelType(sitk.sitkInt32)
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_filter.SetInterpolator(sitk.sitkBSpline)

    image_resize = resample_filter.Execute(image)
    if is_roi:
        data_resize = sitk.GetArrayFromImage(image_resize)
        # data_resize[data_resize < 0.5] = 0
        # data_resize[0.5 <= data_resize < 1.5] = 1
        # data_resize[1.5 <= data_resize < 2.5] = 2
        # data_resize[2.5 <= data_resize < 3.5] = 3
        # data_resize[data_resize > 3.5] = 4
        new_data = np.zeros(data_resize.shape, dtype=np.uint8)
        pixels = np.unique(np.asarray(sitk.GetArrayFromImage(image), dtype=int))
        for i in range(len(pixels)):
            if i == (len(pixels) - 1):
                max = pixels[i]
                min = (pixels[i - 1] + pixels[i]) / 2
            elif i == 0:
                max = (pixels[i] + pixels[i + 1]) / 2
                min = pixels[i]
            else:
                max = (pixels[i] + pixels[i + 1]) / 2
                min = (pixels[i - 1] + pixels[i]) / 2
            new_data[np.bitwise_and(data_resize > min, data_resize <= max)] = pixels[i]
        output = sitk.GetImageFromArray(data_resize)
        output.CopyInformation(image_resize)
    else:
        output = image_resize

    if store_path and store_path.endswith(('.nii', '.nii.gz')):
        sitk.WriteImage(output, store_path)
    return output


def TestResize():
    data_folder = r'W:\Public Datasets\PROSTATEx_Seg\Seg'
    for case in os.listdir(data_folder):
        # case = 'ProstateX-0090'
        print(case)
        case_folder = os.path.join(data_folder, case)
        save_path = os.path.join(r'X:\RawData\ProstateX_Seg_ZYH\Data', case)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        roi_path = os.path.join(case_folder, 'roi.nii.gz')
        t2_path = os.path.join(case_folder, 't2.nii')
        t2 = sitk.ReadImage(t2_path)
        roi = sitk.ReadImage(roi_path)
        if t2.GetSpacing() == (0.5, 0.5, 3.0):
            sitk.WriteImage(t2, os.path.join(save_path, 't2_resize.nii'))
            sitk.WriteImage(roi, os.path.join(save_path, 'roi_resize.nii.gz'))
        else:
            ResizeSipmleITKImage(t2, is_roi=False, expected_resolution=(0.5, 0.5, 3.0), store_path=os.path.join(save_path, 't2_resize.nii'))
            ResizeSipmleITKImage(roi, is_roi=True, expected_resolution=(0.5, 0.5, 3.0),
                                 store_path=os.path.join(save_path, 'roi_resize.nii.gz'))
        # break


def SaveNPY(case, data_path, des_path, crop_shape):
    print(case)
    t2_path = os.path.join(data_path, 't2_resize.nii')
    roi_path = os.path.join(data_path, 'roi_resize.nii')

    image, t2, _ = LoadImage(t2_path)
    _, roi, _ = LoadImage(roi_path, dtype=np.int32)
    t2 = t2.transpose((2, 0, 1))
    roi = roi.transpose((2, 0, 1))

    t2_save_path = os.path.join(des_path, 'T2')
    roi_0_save_path = os.path.join(des_path, 'BG')
    roi_1_save_path = os.path.join(des_path, 'PZ')
    roi_2_save_path = os.path.join(des_path, 'CZ')
    roi_3_save_path = os.path.join(des_path, 'DPU')
    roi_4_save_path = os.path.join(des_path, 'ASF')

    for slice in range(t2.shape[0]):
        if slice == range(t2.shape[0])[0] or slice == range(t2.shape[0])[-1]:
            continue

        # t2_slice_before = t2[slice - 1, ...]
        # t2_slice_before = CropData(t2_slice_before, crop_shape)
        # t2_slice_after = t2[slice + 1, ...]
        # t2_slice_after = CropData(t2_slice_after, crop_shape)
        # t2_slice = t2[slice, ...]
        # t2_slice = CropData(t2_slice, crop_shape)
        # t2_slice = np.array([t2_slice_before, t2_slice, t2_slice_after])

        t2_slice = t2[slice-1: slice + 2, ...]
        t2_slice = Crop3DData(t2_slice, (3, crop_shape[0], crop_shape[1]))

        # t2_slice = t2_slice[np.newaxis, ...]

        t2_npy_path = os.path.join(t2_save_path, '{}_-_slice{}.npy'.format(case, slice))


        roi_slice = roi[slice, ...]
        if np.sum(roi_slice) == 0:
            continue
        roi_crop = CropData(roi_slice, crop_shape, is_roi=True)

        roi_0 = (roi_crop == 0).astype(int)
        roi_1 = (roi_crop == 1).astype(int)
        roi_2 = (roi_crop == 2).astype(int)
        roi_3 = (roi_crop == 3).astype(int)
        roi_4 = (roi_crop == 4).astype(int)

        roi_0_npy_path = os.path.join(roi_0_save_path,  '{}_-_slice{}.npy'.format(case, slice))
        roi_1_npy_path = os.path.join(roi_1_save_path,  '{}_-_slice{}.npy'.format(case, slice))
        roi_2_npy_path = os.path.join(roi_2_save_path,  '{}_-_slice{}.npy'.format(case, slice))
        roi_3_npy_path = os.path.join(roi_3_save_path,  '{}_-_slice{}.npy'.format(case, slice))
        roi_4_npy_path = os.path.join(roi_4_save_path,  '{}_-_slice{}.npy'.format(case, slice))

        np.save(t2_npy_path, t2_slice)
        np.save(roi_0_npy_path, roi_0[np.newaxis])
        np.save(roi_1_npy_path, roi_1[np.newaxis])
        np.save(roi_2_npy_path, roi_2[np.newaxis])
        np.save(roi_3_npy_path, roi_3[np.newaxis])
        np.save(roi_4_npy_path, roi_4[np.newaxis])


def SaveNPY3D(case, data_path, des_path, crop_shape):
    print(case)
    from MeDIT.ArrayProcess import Crop3DArray

    t2_path = os.path.join(data_path, 't2_resize.nii')
    roi_path = os.path.join(data_path, 'roi_resize.nii')

    image, t2, _ = LoadImage(t2_path)
    _, roi, _ = LoadImage(roi_path, dtype=np.int32)
    t2 = t2.transpose((2, 0, 1))
    roi = roi.transpose((2, 0, 1))

    t2_save_path = os.path.join(des_path, 'T2')
    roi_0_save_path = os.path.join(des_path, 'BG')
    roi_1_save_path = os.path.join(des_path, 'PZ')
    roi_2_save_path = os.path.join(des_path, 'CZ')
    roi_3_save_path = os.path.join(des_path, 'DPU')
    roi_4_save_path = os.path.join(des_path, 'ASF')

    t2_crop = Crop3DArray(t2, crop_shape)
    roi_crop = Crop3DArray(roi, crop_shape)

    roi_0 = (roi_crop == 0).astype(int)
    roi_1 = (roi_crop == 1).astype(int)
    roi_2 = (roi_crop == 2).astype(int)
    roi_3 = (roi_crop == 3).astype(int)
    roi_4 = (roi_crop == 4).astype(int)

    t2_crop = NormalizeZ(t2_crop)

    t2_npy_path = os.path.join(t2_save_path, '{}.npy'.format(case))
    roi_0_npy_path = os.path.join(roi_0_save_path, '{}.npy'.format(case))
    roi_1_npy_path = os.path.join(roi_1_save_path, '{}.npy'.format(case))
    roi_2_npy_path = os.path.join(roi_2_save_path, '{}.npy'.format(case))
    roi_3_npy_path = os.path.join(roi_3_save_path, '{}.npy'.format(case))
    roi_4_npy_path = os.path.join(roi_4_save_path, '{}.npy'.format(case))

    np.save(t2_npy_path, t2_crop)
    np.save(roi_0_npy_path, roi_0)
    np.save(roi_1_npy_path, roi_1)
    np.save(roi_2_npy_path, roi_2)
    np.save(roi_3_npy_path, roi_3)
    np.save(roi_4_npy_path, roi_4)


if __name__ == '__main__':
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
    des_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/Three_CorrectNorm'
    # os.mkdir(des_folder)
    os.mkdir(os.path.join(des_folder, 'BG'))
    os.mkdir(os.path.join(des_folder, 'PZ'))
    os.mkdir(os.path.join(des_folder, 'CZ'))
    os.mkdir(os.path.join(des_folder, 'DPU'))
    os.mkdir(os.path.join(des_folder, 'ASF'))
    os.mkdir(os.path.join(des_folder, 'T2'))

    for case in sorted(os.listdir(data_root)):
        case_path = os.path.join(data_root, case)
        SaveNPY(case, case_path, des_folder, crop_shape=(250, 250))
        # SaveNPY3D(case, case_path, des_folder, crop_shape=(20, 250, 250))







