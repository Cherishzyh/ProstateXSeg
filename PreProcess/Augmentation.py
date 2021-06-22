import os
import cv2
import numpy as np
from copy import deepcopy

from BasicTool.MeDIT.ArrayProcess import Crop2DArray
from BasicTool.MeDIT.SaveAndLoad import LoadImage
from BasicTool.MeDIT.Normalize import Normalize01
from BasicTool.MeDIT.Visualization import Imshow3DArray


def Rotate(data, param):
    row, col = data.shape
    alpha = np.cos(param['theta'] / 180 * np.pi)
    beta = np.sin(param['theta'] / 180 * np.pi)

    matrix = np.array([
        [alpha, beta, (1 - alpha) * (col // 2) - beta * (row // 2)],
        [-beta, alpha, beta * (col // 2) + (1 - alpha) * (row // 2)]
    ])
    return cv2.warpAffine(data, matrix, (col, row), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)


def Zoom(data, param):
    result = cv2.resize(data, None, fx=param['vertical_zoom'], fy=param['horizontal_zoom'],
                        interpolation=cv2.INTER_LINEAR)
    result = Crop2DArray(result, data.shape)
    return result


def Flip(data, param):
    result = deepcopy(data)
    if param['horizontal_flip']:
        result = np.flip(result, axis=1)
    if param['vertical_flip']:
        result = np.flip(result, axis=0)
    return result


def test():
    root = r'Z:\PM\Prostate301_test'
    for case in os.listdir(root):
        case_folder = os.path.join(root, case)
        t2_path = os.path.join(case_folder, 't2.nii')
        _, t2, _ = LoadImage(t2_path)
        t2_rotate = []
        for slice in range(t2.shape[-1]):
            t2_rotate.append(Rotate(t2[:, :, slice], {'theta': -10}))
        t2_rotate = np.transpose(np.array(t2_rotate), axes=(1, 2, 0))

        t2_rotate_1 = []
        for slice in range(t2.shape[-1]):
            t2_rotate_1.append(Rotate(t2_rotate[:, :, slice], {'theta': 10}))
        t2_rotate_1 = np.transpose(np.array(t2_rotate_1), axes=(1, 2, 0))

        Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(t2_rotate), Normalize01(t2_rotate_1)], axis=1))


def AugmentPred():
    root = r'Z:\PM\Prostate301_test'
    for case in os.listdir(root):
        case_folder = os.path.join(root, case)
        t2_path = os.path.join(case_folder, 't2.nii')
        _, t2, _ = LoadImage(t2_path)

        pred_list = []
        for idx in range(18):
            pred_id = 'ProstateX_UNet_{}.nii.gz'.format(str(idx))
            pred_path = os.path.join(case_folder, pred_id)
            _, pred, _ = LoadImage(pred_path)
            pred_list.append(pred)

        augment_pred = np.array(pred_list)
        augment_pred = np.sum(augment_pred, axis=0) / 18
        augment_pred = np.around(augment_pred)

        roi_1 = (augment_pred == 1).astype(int)
        roi_2 = (augment_pred == 2).astype(int)
        roi_3 = (augment_pred == 3).astype(int)
        roi_4 = (augment_pred == 4).astype(int)

        # Imshow3DArray(np.concatenate([Normalize01(t2), Normalize01(t2)], axis=1),
        #               roi=[np.concatenate([Normalize01(roi_1), Normalize01(roi_1_flip)], axis=1),
        #                    np.concatenate([Normalize01(roi_2), Normalize01(roi_2_flip)], axis=1),
        #                    np.concatenate([Normalize01(roi_3), Normalize01(roi_3_flip)], axis=1),
        #                    np.concatenate([Normalize01(roi_4), Normalize01(roi_4_flip)], axis=1)])
        Imshow3DArray(Normalize01(t2),
                      roi=[Normalize01(roi_1), Normalize01(roi_2), Normalize01(roi_3), Normalize01(roi_4)])


if __name__ == '__main__':
    # pass
    test()
    # AugmentPred()