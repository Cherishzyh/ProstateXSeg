import os.path

import torch

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from T4T.Utility.Data import *
from MeDIT.Others import IterateCase

from Statistics.Metric import Dice


class ProstateXSeg():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def CropData(self, data, crop_shape, is_roi=False, slice_num=1):
        from MeDIT.ArrayProcess import ExtractPatch
        from MeDIT.ArrayProcess import ExtractBlock

        # Crop
        if np.ndim(data) == 2:
            data_crop, _ = ExtractPatch(data, crop_shape)
        else:
            data_crop, _ = ExtractBlock(data, (slice_num, crop_shape[0], crop_shape[0]))

        if not is_roi:
            # Normalization
            data_crop -= np.mean(data_crop)
            data_crop /= np.std(data_crop)

        return data_crop

    def ROIOneHot(self, roi):
        '''
        :param roi:
        :return:
        '''
        roi_list = []
        roi_class = [0, 1, 2, 3, 4]
        for index in roi_class:
            roi_list.append((roi == index).astype(int))
        return np.array(roi_list, dtype=np.int32)

    def KeepLargest(self, roi):
        from scipy import ndimage
        new_mask = np.zeros(roi.shape)
        if np.all(roi == 0):
            return new_mask
        else:
            for cls in range(1, roi.shape[1]):
                if np.all(roi[:, cls] == 0):
                    new_mask[:, cls] = np.zeros_like(roi[:, cls])
                else:
                    label_im, nb_labels = ndimage.label(roi[:, cls])
                    max_volume = [(label_im == index).sum() for index in range(1, nb_labels + 1)]
                    index = np.argmax(max_volume)
                    new_mask[:, cls][label_im == index + 1] = 1
            new_mask[:, 0] = 1 - np.sum(new_mask, axis=1)
            assert np.all(np.sum(new_mask, axis=1) == 1)
            return new_mask

    def Nii2NPY(self, case, data_path, slice_num=1):
        from MeDIT.Visualization import FlattenImages
        from MeDIT.SaveAndLoad import LoadImage

        t2_path = os.path.join(data_path, 't2_resize.nii')
        roi_path = os.path.join(data_path, 'roi_resize.nii')

        image, t2, _ = LoadImage(t2_path)
        _, roi, _ = LoadImage(roi_path, dtype=np.int32)
        t2 = t2.transpose((2, 0, 1))
        roi = roi.transpose((2, 0, 1))

        start_slice = int((slice_num - 1) / 2)
        end_slice = t2.shape[0] - start_slice
        t2_list, roi_list = [], []
        for slice in range(start_slice, end_slice):
            t2_slice = np.squeeze(t2[int(slice - start_slice): int(slice + start_slice)+1])
            t2_slice = self.CropData(t2_slice, self.input_shape, slice_num=slice_num)
            t2_slice = np.squeeze(t2_slice)

            roi_slice = roi[slice]
            roi_slice = self.CropData(roi_slice, self.input_shape, is_roi=True)

            roi_slice = self.ROIOneHot(roi_slice)

            if len(t2_slice.shape) == 2:
                t2_slice = t2_slice[np.newaxis, ...]

            t2_list.append(t2_slice)
            roi_list.append(roi_slice)

        return np.array(t2_list), np.array(roi_list, dtype=np.int32)

    def run(self, case, model, model_path, inputs, outputs, is_save=False, weights_list=None, is_multipy=False):
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        if weights_list is None:
            weights_list = [one for one in IterateCase(model_path, only_folder=False, verbose=0) if one.is_file()]
            weights_list = [one for one in weights_list if str(one).endswith('.pt')]
            if len(weights_list) == 0:
                raise Exception
            weights_list = sorted(weights_list, key=lambda x: os.path.getctime(str(x)))
            weights_path = weights_list[-1]
        else:
            weights_path = weights_list

        weights_path = os.path.join(model_path, weights_path)
        model.to(device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        inputs = MoveTensorsToDevice(inputs, device)

        # print('****** predicting {} | (｀・ω・´) ****** '.format(case))

        with torch.no_grad():
            preds = model(inputs, epoch=10, is_multipy=is_multipy)

        if is_save:
            result_folder = os.path.join(model_path, 'CaseResult')
            if not os.path.exists(os.path.join(model_path, 'CaseResult')):
                os.mkdir(os.path.join(model_path, 'CaseResult'))

            np.save(os.path.join(result_folder, '{}.npy'.format(case)), np.squeeze(preds.cpu().data.numpy()))
            # np.save(os.path.join(result_folder, '{}_label.npy'.format(case)), outputs.cpu().data.numpy())

        return preds


if __name__ == '__main__':
    from PreProcess.Nii2NPY import ROIOneHot
    from SegModel.SuccessfulWNet import WNet2_5D
    from SegModel.WNet import WNet2_5D_channel
    from Statistics.Metric import BinarySegmentation

    seg = ProstateXSeg((192, 192))
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
    model_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    model_name = 'WNet_1221_mse'

    model = WNet2_5D(1, 3, 1, 5)

    # df_train = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'train_case_name.csv'))
    # df_val = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'val_case_name.csv'))
    df_train = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'all_train_case_name.csv'))
    df_test = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'test_case_name.csv'))
    for i, df in enumerate([df_train, df_test]):
        HD_dict = {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}
        Dice_dict = {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}
        case_list = df.values.tolist()[0]
        bs = BinarySegmentation(is_show=False)
        for case in sorted(case_list):
            t2_arr, roi_arr = seg.Nii2NPY(case, os.path.join(data_path, case), slice_num=3)
            preds_list = []
            # plt.figure(dpi=500)
            # plt.imshow(t2_arr[10, 1], cmap='gray')
            # plt.contour(np.argmax(roi_arr[10], axis=0))
            # plt.axis('off')
            # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/WNet_1220_mse/Image', '{}.jpg'.format(case)))
            # plt.close()
            for cv_index in range(5):
                preds = seg.run(case, model,
                                model_path=os.path.join(model_folder, '{}/CV_{}'.format(model_name, cv_index)),
                                inputs=t2_arr,
                                outputs=roi_arr,
                                is_save=False,
                                is_multipy=True)
                if isinstance(preds, tuple):
                    preds = preds[-1]
                preds = torch.softmax(preds, dim=1).cpu().numpy()
                # preds = torch.argmax(preds, dim=1)
                # preds = ROIOneHot(preds)
                # preds = preds.transpose(1, 0, 2, 3)
                # preds = seg.KeepLargest(preds)
                # preds = median_filter(preds, size=(1, 1, 3, 3))
                preds_list.append(np.asarray(preds, dtype=np.float32))

            mean_pred = np.mean(np.array(preds_list), axis=0)
            mean_pred = np.argmax(mean_pred, axis=1)
            mean_pred = ROIOneHot(mean_pred)
            mean_pred = mean_pred.transpose(1, 0, 2, 3)
            mean_pred = seg.KeepLargest(mean_pred)
            mean_pred = median_filter(mean_pred, size=(1, 1, 3, 3))
            mean_pred = np.asarray(mean_pred, dtype=np.int32)

            # plt.figure()
            # plt.imshow(t2_arr[10, 1], cmap='gray')
            # plt.contour(np.argmax(mean_pred[10], axis=0))
            # plt.axis('off')
            # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/WNet_1220_mse_multiply/Image', '{}.jpg'.format(case)))
            # plt.close()

            for index, classes in enumerate(['BG', 'PZ', 'TZ', 'DPU', 'AFS']):
                if classes == 'BG': continue
                metric = bs.Run(mean_pred[:, index], roi_arr[:, index])
                Dice_dict[classes].append(metric['Dice'])
                if 'HD' in metric.keys():
                    HD_dict[classes].append(metric['HD'])

        plt.figure(figsize=(8, 6))
        plt.subplot(221)
        plt.title('mean dice of PZ: {:.3f}'.format(sum(Dice_dict['PZ']) / len(Dice_dict['PZ'])))
        plt.hist(Dice_dict['PZ'], bins=20)
        plt.subplot(222)
        plt.title('mean dice of TZ: {:.3f}'.format(sum(Dice_dict['TZ']) / len(Dice_dict['TZ'])))
        plt.hist(Dice_dict['TZ'], bins=20)
        plt.subplot(223)
        plt.title('mean dice of DPU: {:.3f}'.format(sum(Dice_dict['DPU']) / len(Dice_dict['DPU'])))
        plt.hist(Dice_dict['DPU'], bins=20)
        plt.subplot(224)
        plt.title('mean dice of AFS: {:.3f}'.format(sum(Dice_dict['AFS']) / len(Dice_dict['AFS'])))
        plt.hist(Dice_dict['AFS'], bins=20)
        plt.show()

        print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(Dice_dict['PZ']) / len(Dice_dict['PZ']), sum(Dice_dict['TZ']) / len(Dice_dict['TZ']),
                                                         sum(Dice_dict['DPU']) / len(Dice_dict['DPU']), sum(Dice_dict['AFS']) / len(Dice_dict['AFS'])))
        print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(HD_dict['PZ']) / len(HD_dict['PZ']), sum(HD_dict['TZ']) / len(HD_dict['TZ']),
                                                         sum(HD_dict['DPU']) / len(HD_dict['DPU']), sum(HD_dict['AFS']) / len(HD_dict['AFS'])))



