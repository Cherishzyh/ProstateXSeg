import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter, median_filter
from scipy.spatial.distance import cdist

from SSHProject.BasicTool.MeDIT.SaveAndLoad import LoadImage
from SSHProject.BasicTool.MeDIT.Normalize import Normalize01


def GetRoiEdge(attention):
    raw_attention = deepcopy(attention)
    raw_attention[raw_attention >= 1] = 1
    dilate_attention = maximum_filter(raw_attention, size=(3, 3))
    if isinstance(dilate_attention, np.ndarray):
        dilate_attention = torch.from_numpy(dilate_attention)
    return dilate_attention - raw_attention


def GetEdgeIndex(edge):
    x, y = torch.where(edge == 1)
    return x.tolist(), y.tolist()


def GetShortestDis(dot, edge):
    '''

    :param dot: (x, y)
    :param edge: a 2d array only include 0 and 1, 1 is the edge
    :return: the shortest distance of dot to edge
    '''

    dis = cdist(dot, edge, metric='euclidean')
    return np.min(dis)


def DistanceMap(roi, is_show=False):
    assert len(roi.shape) == 2
    if len(torch.unique(roi)) != 2:
        roi[roi >= 1] = 1

    edge = GetRoiEdge(roi)
    edge_x, edge_y = GetEdgeIndex(edge)

    edge_index = torch.stack((torch.tensor(edge_x), torch.tensor(edge_y)), dim=1)

    dis_map = torch.zeros_like(roi)

    for x_index in range(roi.shape[0]):
        for y_index in range(roi.shape[1]):
            if roi[x_index][y_index] == 1:
                continue
            else:
                indexs = torch.tensor([(x_index, y_index)])
                value = GetShortestDis(indexs, edge_index)
                dis_map[x_index][y_index] = - value
    dis_map = Normalize01(dis_map)\

    # dis_map = torch.from_numpy(dis_map)
    # print(torch.min(dis_map), torch.max(dis_map))

    if is_show:
        plt.subplot(131)
        plt.imshow(roi, cmap='gray')
        plt.subplot(132)
        plt.imshow(edge, cmap='gray')
        plt.subplot(133)
        plt.imshow(dis_map, cmap='gray')
        plt.show()

    return dis_map


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    roi_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData/ProstateX-0340/roi.nii.gz'
    t2_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData/ProstateX-0340/t2.nii'
    base_rate = 0.025

    _, roi, _ = LoadImage(roi_path)
    _, t2, _ = LoadImage(t2_path)

    # result = IntraSliceFilter(roi[..., 10], base_rate)  #result.shape=(23, 399, 399)
    roi = roi[..., 10]
    roi_tensor = torch.from_numpy(roi)

    dis_map = DistanceMap(roi_tensor, is_show=False)


    plt.subplot(131)
    plt.imshow(t2[..., 10], cmap='gray')
    plt.contour(roi, colors='r')
    plt.subplot(132)
    plt.imshow(dis_map, cmap='gray')
    plt.subplot(133)
    plt.imshow(Normalize01(t2[..., 10]) * dis_map, cmap='gray')
    plt.show()

