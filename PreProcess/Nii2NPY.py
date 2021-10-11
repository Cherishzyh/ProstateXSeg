import numpy as np
import os
import matplotlib.pyplot as plt

from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Normalize import Normalize01
from MeDIT.Visualization import Imshow3DArray


def CropData(t2_data, crop_shape, center, is_roi=False):
    from BasicTool.MeDIT.ArrayProcess import ExtractPatch

    # Crop
    t2_crop, _ = ExtractPatch(t2_data, crop_shape, center_point=center)

    if not is_roi:
        # Normalization
        t2_crop -= np.mean(t2_crop)
        t2_crop /= np.std(t2_crop)

    return t2_crop


def GetCenter(roi):
    roi_row = []
    roi_column = []
    for row in range(roi.shape[0]):
        roi_row.append(np.sum(roi[row, ...]))
    for column in range(roi.shape[1]):
        roi_column.append(np.sum(roi[..., column]))

    max_row = max(roi_row)
    max_column = max(roi_column)
    row_index = roi_row.index(max_row)
    column_index = roi_column.index(max_column)

    column = np.argmax(roi[row_index])
    row = np.argmax(roi[..., column_index])
    # center = [int(column + max_row // 2), int(row + max_column // 2)]
    center = [int(row + max_column // 2), int(column + max_row // 2)]
    return center


def GetROICenter(roi):
    '''
    :param roi: 2D roi include multi-class
    :return: the center of entire roi
    '''

    assert len(roi.shape) == 2
    roi_binary = (roi >= 1).astype(int)
    center = GetCenter(roi_binary)
    return center


def ShapeofROI(data_root):
    length_list = []
    width_list = []
    for case in os.listdir(data_root):
        print(case)
        case_path = os.path.join(data_root, case)
        roi_path = os.path.join(case_path, 'roi.nii.gz')
        _, roi, _ = LoadImage(roi_path, dtype=np.int)
        roi = (roi >= 1).astype(int)
        for slice in range(roi.shape[-1]):
            _, shape = GetCenter(roi[..., slice])
            length_list.append(shape[3]-shape[2])
            width_list.append(shape[1]-shape[0])
    print(max(length_list), max(width_list))


def ROIOneHot(roi, roi_class=(0, 1, 2, 3, 4)):
    '''
    :param roi:
    :return:
    '''
    roi_list = []
    # roi_class = [0, 1, 2, 3, 4]
    if len(roi.shape) == 2:
        for index in roi_class:
            roi_list.append((roi == index).astype(int))
        return np.array(roi_list, dtype=np.int)
    elif len(roi.shape) == 3:
        roi_list_3d = []
        for slice in range(roi.shape[0]):
            roi_list = []
            for index in roi_class:
                roi_list.append((roi[slice] == index).astype(int))
            roi_list_3d.append(roi_list)
        return np.array(roi_list_3d, dtype=np.int)



def SaveNPY(case, data_path, des_path, crop_shape):
    print(case)

    t2_path = os.path.join(data_path, 't2.nii')
    roi_path = os.path.join(data_path, 'roi.nii.gz')

    _, t2, _ = LoadImage(t2_path)
    _, roi, _ = LoadImage(roi_path, dtype=np.int)
    t2 = t2.transpose((2, 0, 1))
    roi = roi.transpose((2, 0, 1))

    t2_save_path = os.path.join(des_path, 'T2Slice')
    roi_save_path = os.path.join(des_path, 'RoiSlice')

    for slice in range(t2.shape[0]):
        # if slice == range(t2.shape[0])[0] or slice == range(t2.shape[0])[-1]:
        #     continue

        # t2_slice_before = t2[slice-1, ...]
        # t2_slice_after = t2[slice + 1, ...]
        t2_slice = t2[slice, ...]
        roi_slice = roi[slice, ...]

        if np.sum(roi_slice) == 0:
            continue

        center = GetROICenter(roi_slice)

        # t2_slice_before = CropData(t2_slice_before, crop_shape, center=center)
        # t2_slice_after = CropData(t2_slice_after, crop_shape, center=center)
        t2_slice = CropData(t2_slice, crop_shape, center=center)
        roi_slice = CropData(roi_slice, crop_shape, center=center, is_roi=True)


        t2_slice = t2_slice[np.newaxis, ...]
        # t2_slice = np.array([t2_slice_before, t2_slice, t2_slice_after])
        roi_slice = ROIOneHot(roi_slice)


        t2_npy_path = os.path.join(t2_save_path, '{}_-_slice{}.npy'.format(case, slice))
        roi_npy_path = os.path.join(roi_save_path, '{}_-_slice{}.npy'.format(case, slice))

        np.save(t2_npy_path, t2_slice)
        np.save(roi_npy_path, roi_slice)


if __name__ == '__main__':
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
    des_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # ShapeofROI(data_root)
    for case in os.listdir(data_root):
        case_path = os.path.join(data_root, case)
        SaveNPY(case, case_path, des_folder, crop_shape=(250, 250))