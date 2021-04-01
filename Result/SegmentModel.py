import numpy as np
import os
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

from Result.ConfigInterpretor import ConfigInterpretor, BaseImageOutModel

class ProstatePzCgSegmentationInput_3(BaseImageOutModel):
    def __init__(self):
        super(ProstatePzCgSegmentationInput_3, self).__init__()
        self._image_preparer = ConfigInterpretor()

    def __KeepLargest(self, mask):
        new_mask = np.zeros(mask.shape)
        if mask.max() != 0:
            for position in range(1, 3):
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


    def Run(self, image, store_folder=''):
        ref = ReformatAxis()
        if isinstance(image, str):
            image = sitk.ReadImage(image)

        resolution = image.GetSpacing()

        # data1 = ref.Run(image)
        data = sitk.GetArrayFromImage(image)
        data = data.transpose(1, 2, 0)
        # flip_log = [0, 0, 0]
        # data, _ = GetDataFromSimpleITK(image, dtype=np.float32)

        # Preprocess Data
        data = self._config.CropDataShape(data, resolution)
        input_list = self.TransOneDataFor2_5DModel(data)

        with torch.no_grad():
            input_0, input_1, input_2 = torch.from_numpy(input_list[0]), torch.from_numpy(input_list[1]), torch.from_numpy(input_list[2])
            inputs = torch.cat([input_0, input_1, input_2], dim=1)
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            preds_list = self._model(inputs)

        pred = preds_list.cpu().data.numpy()
        pred = np.argmax(pred, axis=1)

        pred = self.invTransDataFor2_5DModel(pred)

        pred = self._config.RecoverDataShape(pred, resolution)
        # pred = self.__FilterResult(pred)



        if pred.max() > 0:
            new_pred = self.__KeepLargest(pred.astype(int))
        else:
            new_pred = np.zeros_like(pred)


        # mask_image = ref.BackToImage(new_pred)
        new_pred = new_pred.transpose(2, 1, 0)
        mask_image = sitk.GetImageFromArray(new_pred)
        mask_image.SetDirection(image.GetDirection())
        mask_image.SetSpacing(image.GetSpacing())
        mask_image.SetOrigin(image.GetOrigin())
        # mask_image = GetImageFromArrayByImage(new_pred, image, flip_log=flip_log)
        if store_folder:
            if os.path.isdir(store_folder):
                store_folder = os.path.join(store_folder, '{}.nii.gz'.format(self._config.GetName()))
            SaveNiiImage(store_folder, mask_image)

        return new_pred, mask_image

if __name__ == '__main__':
    from MeDIT.UsualUse import *
    segmentor = ProstatePzCgSegmentationInput_3()
    segmentor.LoadConfigAndModel(r'/home/zhangyihong/Documents/ProstateX')

    from pathlib import Path
    root_folder = Path(r'/home/zhangyihong/Documents/PM')
    for case in sorted(root_folder.iterdir()):
        # if case.name < '2019-CA-formal-CHANG XIAN YUN':
        #     continue
        print(case.name)
        # image, _, show_data = LoadImage(str(case / 't2.nii'))
        image = sitk.ReadImage(str(case / 't2.nii'))
        mask, mask_image = segmentor.Run(image, store_folder=os.path.join(r'/home/zhangyihong/Documents/ProstateX/test', str(case)))
        break

        # _, _, roi = LoadNiiData(str(case / 'merge_pz1_cg2.nii'), dtype=int)
        # Imshow3DArray(Normalize01(show_data), roi=[
        #     (roi == 1).astype(int),
        #     (mask == 1).astype(int),
        #     (roi == 2).astype(int),
        #     (mask == 2).astype(int)
        # ])


