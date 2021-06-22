import os
import cv2
import numpy as np
from copy import deepcopy

from scipy import ndimage
import SimpleITK as sitk
import scipy.signal as signal
import torch
from skimage.morphology import ball, binary_dilation, binary_closing, disk
import matplotlib.pyplot as plt

from BasicTool.MeDIT.ImageProcess import GetDataFromSimpleITK, GetImageFromArrayByImage
from BasicTool.MeDIT.Normalize import NormalizeForTorch
from BasicTool.MeDIT.SaveAndLoad import SaveNiiImage, LoadImage
from BasicTool.MeDIT.ImageProcess import ReformatAxis
from BasicTool.MeDIT.Visualization import FlattenAllSlices
from BasicTool.MeDIT.ArrayProcess import Crop2DArray

from Result.ConfigInterpretor import ConfigInterpretor, BaseImageOutModel


class ProstatePzCgSegmentationInput_3(BaseImageOutModel):
    def __init__(self):
        super(ProstatePzCgSegmentationInput_3, self).__init__()
        self._image_preparer = ConfigInterpretor()

    def __KeepLargest(self, mask, class_num=3):
        new_mask = np.zeros(mask.shape)
        if mask.max() != 0:
            for position in range(1, class_num):
                if not np.any(mask == position):
                    continue
                temp_mask = (mask == position).astype(int)
                temp_mask = binary_closing(temp_mask, ball(1))
                temp_mask = binary_dilation(temp_mask, disk(3)[..., np.newaxis])

                label_im, nb_labels = ndimage.label(temp_mask.astype(int))
                max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
                index = np.argmax(max_volume)
                new_mask[label_im == index + 1] = position
        return new_mask

    def __FilterResult(self, mask):
        result = np.zeros_like(mask)
        for slice_index in range(mask.shape[-1]):
            one_slice = mask[..., slice_index]
            one_slice = signal.medfilt(one_slice, 5)
            result[..., slice_index] = one_slice
        return result

    def TransOneDataFor2_5DModel(self, data):
        # Here needs to be set according to config
        data_list = [data[..., :-2], data[..., 1:-1], data[..., 2:]]
        for input_data_index in range(len(data_list)):
            temp = data_list[input_data_index]
            temp = np.transpose(temp, (2, 0, 1))
            temp = temp[:, np.newaxis, :, :]
            temp = temp.astype(np.float32)
            temp = NormalizeForTorch(temp)
            data_list[input_data_index] = temp

        return data_list

    def invTransDataFor2_5DModel(self, preds):
        preds = np.squeeze(preds)
        preds = np.transpose(preds, (1, 2, 0))
        preds = np.concatenate((np.zeros((self._config.GetShape()[0], self._config.GetShape()[1], 1)),
                                preds,
                                np.zeros((self._config.GetShape()[0], self._config.GetShape()[1], 1))),
                               axis=-1)
        return preds

    def Rotate2d(self, data, param):
        assert len(data.shape) == 2
        row, col = data.shape
        alpha = np.cos(param['theta'] / 180 * np.pi)
        beta = np.sin(param['theta'] / 180 * np.pi)

        matrix = np.array([
            [alpha, beta, (1 - alpha) * (col // 2) - beta * (row // 2)],
            [-beta, alpha, beta * (col // 2) + (1 - alpha) * (row // 2)]
        ])
        return cv2.warpAffine(data, matrix, (col, row), flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    def Rotate3d(self, data, param):
        assert len(data.shape) == 3
        rotate_data = []
        row, col, slice = data.shape
        if min(row, col, slice) == slice:
            for idx in range(slice):
                rotate_data.append(self.Rotate2d(data[:, :, idx], param))
            result = np.transpose(np.array(rotate_data), axes=(1, 2, 0))
        elif min(row, col, slice) == row:
            slice, row, col = data.shape
            for idx in range(slice):
                rotate_data.append(self.Rotate2d(data[idx, ...], param))
            result = np.array(rotate_data)
        else:
            result = None

        return result

    def Rotate4d(self, data, param):
        assert len(data.shape) == 4
        rotate_data = []
        slice, channel, row, col = data.shape

        for idx in range(channel):
            rotate_data.append(self.Rotate3d(data[:, idx, :, :], param))
        result = np.transpose(np.array(rotate_data), axes=(1, 0, 2, 3))
        return result

    def Zoom2d(self, data, param):
        assert len(data.shape) == 2
        result = cv2.resize(data, None, fx=param['vertical_zoom'], fy=param['horizontal_zoom'],
                            interpolation=cv2.INTER_LINEAR)
        result = Crop2DArray(result, data.shape)
        return result

    def Zoom3d(self, data, param):
        assert len(data.shape) == 3
        zoom_data = []
        row, col, slice = data.shape
        if min(row, col, slice) == slice:
            for idx in range(slice):
                zoom_data.append(self.Zoom2d(data[:, :, idx], param))
            result = np.transpose(np.array(zoom_data), axes=(1, 2, 0))
        elif min(row, col, slice) == row:
            slice, row, col = data.shape
            for idx in range(slice):
                zoom_data.append(self.Zoom2d(data[idx, ...], param))
            result = np.array(zoom_data)
        else:
            result = None

        return result

    def Zoom4d(self, data, param):
        assert len(data.shape) == 4
        zoom_data = []
        slice, channel, row, col = data.shape

        for idx in range(channel):
            zoom_data.append(self.Zoom3d(data[:, idx, :, :], param))
        result = np.transpose(np.array(zoom_data), axes=(1, 0, 2, 3))
        return result

    def Flip(self, data, param):
        result = deepcopy(data)
        if len(result.shape) == 2:
            if param['horizontal_flip']:
                result = np.flip(result, axis=1)
            if param['vertical_flip']:
                result = np.flip(result, axis=0)
        if len(result.shape) == 3:
            if min(result.shape) == result.shape[0]:
                if param['horizontal_flip']:
                    result = np.flip(result, axis=2)
                if param['vertical_flip']:
                    result = np.flip(result, axis=1)
            elif min(result.shape) == result.shape[-1]:
                if param['horizontal_flip']:
                    result = np.flip(result, axis=1)
                if param['vertical_flip']:
                    result = np.flip(result, axis=0)
        return result

    def Flip4d(self, data, param):
        assert len(data.shape) == 4
        # result = deepcopy(data)
        flip_data = []
        slice, channel, row, col = data.shape

        for idx in range(channel):
            flip_data.append(self.Flip(data[:, idx, :, :], param))
        result = np.transpose(np.array(flip_data), axes=(1, 0, 2, 3))
        return result

    def Run(self, image, param, classnum, store_folder=''):
        ref = ReformatAxis()
        if isinstance(image, str):
            image = sitk.ReadImage(image)

        resolution = image.GetSpacing()

        raw_data = ref.Run(image)
        # data = raw_data
        ################################################################################################################

        # raw_data = self.Flip(raw_data, param)
        # raw_data = self.Zoom3d(raw_data,  param)
        # raw_data = self.Rotate3d(raw_data,  param)

        ################################################################################################################
        # Preprocess Data
        data = self._config.CropDataShape(raw_data, resolution)
        input_list = self.TransOneDataFor2_5DModel(data)

        with torch.no_grad():
            input_0, input_1, input_2 = torch.from_numpy(input_list[0]), torch.from_numpy(input_list[1]), torch.from_numpy(input_list[2])
            inputs = torch.cat([input_0, input_1, input_2], dim=1)
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            preds_list = self._model(inputs)

        pred = preds_list.cpu().data.numpy()
        pred = np.concatenate([np.zeros(shape=(1, pred.shape[1], pred.shape[2], pred.shape[3])),
                               pred,
                               np.zeros(shape=(1, pred.shape[1], pred.shape[2], pred.shape[3]))], axis=0)
        ################################################################################################################
        # oppo_horizontal_flip_param = param['horizontal_flip']
        # oppo_vertical_flip_param = param['vertical_flip']
        #
        # oppo_horizontal_zoom_param = 1. / param['horizontal_zoom']
        # oppo_vertical_zoom = 1. / param['vertical_zoom']
        #
        # oppo_theta_param = 0 - param['theta']
        #
        # oppo_param = {'horizontal_flip': oppo_horizontal_flip_param,
        #               'vertical_flip': oppo_vertical_flip_param,
        #               'horizontal_zoom': oppo_horizontal_zoom_param,
        #               'vertical_zoom': oppo_vertical_zoom,
        #               'theta': oppo_theta_param}
        #
        # pred = self.Rotate4d(pred, oppo_param)
        # pred = self.Zoom4d(pred, oppo_param)
        # pred = self.Flip4d(pred, oppo_param)

        ################################################################################################################

        # np.save(os.path.join(store_folder, 'prediction_{}.npy'.format(classnum)), pred)
        # np.save(os.path.join(store_folder, 't2.npy'), data.transpose(2, 0, 1))
        # print()
        pred = np.argmax(pred, axis=1)

        pred = self.invTransDataFor2_5DModel(pred)

        pred = self._config.RecoverDataShape(pred, resolution)

        if pred.max() > 0:
            new_pred = self.__KeepLargest(pred.astype(int), class_num=5)
        else:
            new_pred = np.zeros_like(pred)

        mask_image = ref.BackToImage(new_pred)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, '{}.nii.gz'.format(self._config.GetName()))
            SaveNiiImage(store_folder, mask_image)

        return new_pred, mask_image
        # return 0, 0


if __name__ == '__main__':
    from MeDIT.UsualUse import *
    from pathlib import Path
    segmentor = ProstatePzCgSegmentationInput_3()
    segmentor.LoadConfigAndModel(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/Config')
    root_folder = Path(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/external/Prostate301_normal')

    # classnum = 17
    # param = {'horizontal_flip': False, 'vertical_flip': False, 'horizontal_zoom': 1, 'vertical_zoom': 1, 'theta': 10}

    for case in sorted(root_folder.iterdir()):
        print(case.name)
        if not os.path.exists(str(case / 't2.nii')):
            print('no t2')
            continue

        image, _, show_data = LoadImage(str(case / 't2.nii'))
        # mask, mask_image = segmentor.Run(image, param, classnum, store_folder=os.path.join(root_folder, str(case)))
        mask, mask_image = segmentor.Run(image, store_folder=os.path.join(root_folder, str(case)))




