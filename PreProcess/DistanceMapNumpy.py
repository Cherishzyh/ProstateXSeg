import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.spatial.distance import cdist
from scipy.ndimage import label

from MeDIT.SaveAndLoad import LoadImage
from MeDIT.Normalize import Normalize01


# def IntraSliceFilter(attention, diff_value, is_show=False):
#     raw_attention = deepcopy(attention)
#     raw_attention[raw_attention >= 1] = 1
#
#     while True:
#         new_attention = maximum_filter(raw_attention, size=(3, 3))
#         new_attention[new_attention > raw_attention] -= diff_value
#         # new_attention[new_attention < 0] = 0
#
#         if not (new_attention > raw_attention).any():
#             break
#
#         raw_attention = new_attention
#
#     new_attention1 = median_filter(new_attention, size=(3, 3))
#     new_attention2 = median_filter(new_attention1, size=(3, 3))
#
#     if is_show:
#         plt.subplot(221)
#         plt.imshow(attention, cmap='gray')
#         plt.subplot(222)
#         plt.imshow(new_attention, cmap='jet')
#         plt.subplot(223)
#         plt.imshow((new_attention1-new_attention), cmap='gray')
#         plt.subplot(224)
#         plt.imshow((new_attention2-new_attention1), cmap='gray')
#         plt.show()
#
#     return new_attention


def KeepLargest(mask):

    label_im, nb_labels = label(mask)
    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
    if len(max_volume) == 0:
        new_mask = np.ones(mask.shape)
        return label_im, nb_labels, new_mask
    else:
        index = np.argmax(max_volume)
        new_mask = np.zeros(mask.shape)
        new_mask[label_im == index+1] = 1
        # print(max_volume, np.sum(new_mask))
        return label_im, nb_labels, new_mask


def GetRoiEdge(attention):
    raw_attention = deepcopy(attention)
    raw_attention[raw_attention >= 1] = 1
    return maximum_filter(raw_attention, size=(3, 3)) - raw_attention


def GetEdgeIndex(edge):
    x, y = np.where(edge == 1)
    return x.tolist(), y.tolist()


def GetShortestDis(dot, edge):
    '''

    :param dot: (x, y)
    :param edge: a 2d array only include 0 and 1, 1 is the edge
    :return: the shortest distance of dot to edge
    '''

    dis = cdist(dot, edge, metric='euclidean')
    return np.min(dis)


def DistanceMap(roi, is_show=False, is_decrease=True):
    # assert np.ndim(roi) == 2
    # assert np.squeeze(np.ndim(roi)) == 2
    roi = np.squeeze(roi)
    assert np.ndim(roi) == 2
    if len(np.unique(roi)) != 2:
        th = np.mean(roi)
        roi[roi >= th] = 1
        roi[roi < th] = 0
    _, _, new_roi = KeepLargest(roi)

    edge = GetRoiEdge(new_roi)
    edge_x, edge_y = GetEdgeIndex(edge)

    edge_index = np.stack((edge_x, edge_y), axis=1)

    dis_map = np.zeros_like(new_roi)

    for x_index in range(new_roi.shape[0]):
        for y_index in range(new_roi.shape[1]):
            if new_roi[x_index][y_index] == 1:
                continue
            else:
                indexs = np.array([(x_index, y_index)])
                value = GetShortestDis(indexs, edge_index)
                if is_decrease:
                    dis_map[x_index][y_index] = - value
                else:
                    dis_map[x_index][y_index] = value
                # dis_map[x_index][y_index] = 1. / (value+1)
    dis_map = Normalize01(dis_map)
    # print(np.min(dis_map), np.max(dis_map))

    if is_show:
        plt.subplot(131)
        plt.imshow(roi, cmap='gray')
        plt.subplot(132)
        plt.imshow(edge, cmap='gray')
        plt.subplot(133)
        plt.imshow(dis_map, cmap='jet')
        plt.show()
    return dis_map

if __name__ == '__main__':

    # roi_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData/ProstateX-0340/roi.nii.gz'
    # base_rate = 0.025
    #
    # image, roi, _ = LoadImage(roi_path)
    #
    # # result = IntraSliceFilter(roi[..., 10], base_rate)  #result.shape=(23, 399, 399)
    # roi = roi[..., 10]
    # dis_map = DistanceMap(roi, is_show=True)
    #
    roi_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/RoiSlice'
    dis_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/DistanceMap'

    case_num = len(os.listdir(roi_folder))
    for index, case in enumerate(os.listdir(roi_folder)):
        dis_list = []
        roi = np.load(os.path.join(roi_folder, case))
        for slice in range(roi.shape[0]):
            dis = DistanceMap(roi[slice, ...], is_decrease=False)
            dis_list.append(dis)
        dis_map = np.array(dis_list)
        print('**************save {} | {} / {}****************'.format(case[:case.index('.npy')], index+1, case_num))
        np.save(os.path.join(dis_root, case), dis_map)



