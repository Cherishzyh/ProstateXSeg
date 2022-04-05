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


    def GetCenter(self, mask):
        assert (np.ndim(mask) == 2)
        roi_row = np.sum(mask, axis=1)
        roi_column = np.sum(mask, axis=0)

        row = np.nonzero(roi_row)[0]
        column = np.nonzero(roi_column)[0]

        center = [int(np.mean(row)), int(np.mean(column))]
        return center


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

    def LoadModel(self, model, model_path, weights_list=None):
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
        model.to(self.device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()

    def run(self, case, model, model_path, inputs, is_multipy=False, weights_list=None):
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
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
        model.to(self.device)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        inputs = MoveTensorsToDevice(inputs, self.device)
        with torch.no_grad():
            try:
                preds = model(inputs, epoch=10, is_multipy=is_multipy)
            except Exception:
                preds = model(inputs)

        # if is_save:
        #     result_folder = os.path.join(model_path, 'CaseResult')
        #     if not os.path.exists(os.path.join(model_path, 'CaseResult')):
        #         os.mkdir(os.path.join(model_path, 'CaseResult'))
        #
        #     np.save(os.path.join(result_folder, '{}.npy'.format(case)), np.squeeze(preds.cpu().data.numpy()))
            # np.save(os.path.join(result_folder, '{}_label.npy'.format(case)), outputs.cpu().data.numpy())
        return preds


if __name__ == '__main__':
    from PreProcess.Nii2NPY import ROIOneHot
    from SegModel.SuccessfulWNet import WNet2_5D, WNet2_5D_channelcombine, UNet, WNet2_5D_channelcombine_share
    from Statistics.Metric import BinarySegmentation
    from MeDIT.Visualization import FlattenImages

    seg = ProstateXSeg((192, 192))
    data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
    model_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'

    # ['UNet_20220104', 'WNet_1221_mse', 'WNet_1221_mse_multiply', 'WNet_1221_multiply', 'WNet_1222_mse', 'WNet_1222_mse_multiply', 'WNet_1222_multiply', 'WNet_1230_mse_share']:
    for model_name in ['UNet_20220104', 'WNet_1230_mse_share', 'WNet_1230_mse']:
        if 'UNet' in model_name:
            model = UNet(3, 5)
            print('testing the <{}> with <UNet>'.format(model_name))
        elif '1221' in model_name:
            model = WNet2_5D(1, 3, 1, 5)
            print('testing the <{}> with <WNet>'.format(model_name))
        elif 'share' in model_name:
            model = WNet2_5D_channelcombine_share(3, 3, 5)
            print('testing the <{}> with <WNet>'.format(model_name))
        else:
            model = WNet2_5D_channelcombine(3, 3, 3, 5)
            print('testing the <{}> with <WNet_channelcombine>'.format(model_name))


        if 'multiply' in model_name:
            is_multiply = True
        else:
            is_multiply = False
        if not os.path.exists(os.path.join(model_folder, '{}/{}'.format(model_name, model_name))):
            os.mkdir(os.path.join(model_folder, '{}/{}'.format(model_name, model_name)))

        df_train = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'all_train_case_name.csv'))
        df_test = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH', 'test_case_name.csv'))
        for i, df in enumerate([df_train, df_test]):
            if i == 0: continue
            save_image = os.path.join(model_folder, '{}/{}/{}'.format(model_name, model_name, ['all_train', 'test'][i]))
            if not os.path.exists(os.path.join(save_image, 'PZ')):
                os.mkdir(os.path.join(save_image, 'PZ'))
            if not os.path.exists(os.path.join(save_image, 'TZ')):
                os.mkdir(os.path.join(save_image, 'TZ'))
            if not os.path.exists(os.path.join(save_image, 'DPU')):
                os.mkdir(os.path.join(save_image, 'DPU'))
            if not os.path.exists(os.path.join(save_image, 'ASF')):
                os.mkdir(os.path.join(save_image, 'ASF'))
            print('<{}>'.format(['all_train', 'test'][i]))
            HD_dict = {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}
            Dice_dict = {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}
            CV_HD_dict = [{'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                          {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                          {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                          {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                          {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}]
            CV_Dice_dict = [{'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                            {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                            {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                            {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []},
                            {'PZ': [], 'TZ': [], 'DPU': [], 'AFS': []}]
            case_list = df.values.tolist()[0]
            bs = BinarySegmentation(is_show=False)
            no_label_predict = 0
            label_no_predict = 0
            for case in sorted(case_list):
                t2_arr, roi_arr = seg.Nii2NPY(case, os.path.join(data_path, case), slice_num=3)
                arr_new = np.copy(roi_arr)
                arr_new = arr_new[:, [0, 2, 1, 3, 4]]
                preds_list = []
                # plt.figure(dpi=500)
                # plt.imshow(t2_arr[10, 1], cmap='gray')
                # plt.contour(np.argmax(roi_arr[10], axis=0))
                # plt.axis('off')
                # plt.savefig(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/WNet_1220_mse/Image', '{}.jpg'.format(case)))
                # plt.close()
                for cv_index in range(5):
                    result = seg.run(case, model,
                                    model_path=os.path.join(model_folder, '{}/CV_{}'.format(model_name, cv_index)),
                                    inputs=t2_arr,
                                    is_multipy=is_multiply)
                    if isinstance(result, tuple):
                        result = result[-1]
                    preds = torch.softmax(result, dim=1).cpu().numpy()
                    preds_list.append(np.asarray(preds, dtype=np.float32))
                    preds = np.argmax(preds, axis=1)
                    preds = ROIOneHot(preds)
                    preds = preds.transpose(1, 0, 2, 3)
                    preds = seg.KeepLargest(preds)
                    preds = np.asarray(preds, dtype=np.int32)
                    # for index, classes in enumerate(['BG', 'TZ', 'PZ', 'DPU', 'AFS']):
                    #     if classes == 'BG': continue
                    #     else:
                    #         metric = bs.Run(preds[:, index], arr_new[:, index])
                    #     CV_Dice_dict[cv_index][classes].append(metric['Dice'])
                    #     if 'HD' in metric.keys():
                    #         CV_HD_dict[cv_index][classes].append(metric['HD'])

                mean_pred = np.mean(np.array(preds_list), axis=0)
                mean_pred = np.argmax(mean_pred, axis=1)
                mean_pred = ROIOneHot(mean_pred)
                mean_pred = mean_pred.transpose(1, 0, 2, 3)
                mean_pred = seg.KeepLargest(mean_pred)
                mean_pred = np.asarray(mean_pred, dtype=np.int32)

                # t2_flatten = FlattenImages(t2_arr[:, 1])
                # pred_flatten_TZ = FlattenImages(mean_pred[:, 1])
                # label_flatten_TZ = FlattenImages(arr_new[:, 1])
                # pred_flatten_PZ = FlattenImages(mean_pred[:, 2])
                # label_flatten_PZ = FlattenImages(arr_new[:, 2])
                # pred_flatten_DPU = FlattenImages(mean_pred[:, 3])
                # label_flatten_DPU = FlattenImages(arr_new[:, 3])
                # pred_flatten_ASF = FlattenImages(mean_pred[:, 4])
                # label_flatten_ASF = FlattenImages(arr_new[:, 4])
                #
                # plt.imshow(t2_flatten, cmap='gray')
                # plt.contour(pred_flatten_TZ, linewidths=0.15, colors='g')
                # plt.contour(label_flatten_TZ, linewidths=0.15, colors='r')
                # plt.axis('off')
                # plt.gca().set_axis_off()
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.savefig(os.path.join(os.path.join(save_image, 'PZ'), '{}.jpg'.format(case)), dpi=500)
                # plt.close()
                # plt.imshow(t2_flatten, cmap='gray')
                # plt.contour(pred_flatten_PZ, linewidths=0.15, colors='g')
                # plt.contour(label_flatten_PZ, linewidths=0.15, colors='r')
                # plt.axis('off')
                # plt.gca().set_axis_off()
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.savefig(os.path.join(os.path.join(save_image, 'TZ'), '{}.jpg'.format(case)), dpi=500)
                # plt.close()
                # plt.imshow(t2_flatten, cmap='gray')
                # plt.contour(pred_flatten_DPU, linewidths=0.15, colors='g')
                # plt.contour(label_flatten_DPU, linewidths=0.15, colors='r')
                # plt.axis('off')
                # plt.gca().set_axis_off()
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.savefig(os.path.join(os.path.join(save_image, 'DPU'), '{}.jpg'.format(case)), dpi=500)
                # plt.close()
                # plt.imshow(t2_flatten, cmap='gray')
                # plt.contour(pred_flatten_ASF, linewidths=0.15, colors='g')
                # plt.contour(label_flatten_ASF, linewidths=0.15, colors='r')
                # plt.axis('off')
                # plt.gca().set_axis_off()
                # plt.gca().xaxis.set_major_locator(plt.NullLocator())
                # plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.savefig(os.path.join(os.path.join(save_image, 'ASF'), '{}.jpg'.format(case)), dpi=500)
                # plt.close()
                for i in range(mean_pred.shape[0]):
                    if (np.unique(mean_pred[i, -1]).size == 2 and np.unique(arr_new[i, -1]).size == 1):
                        no_label_predict += 1
                    elif (np.unique(mean_pred[i, -1]).size == 1 and np.unique(arr_new[i, -1]).size == 2):
                        label_no_predict += 1

                for index, classes in enumerate(['BG', 'TZ', 'PZ', 'DPU', 'AFS']):
                    if classes == 'BG': continue
                    else:
                        metric = bs.Run(mean_pred[:, index], arr_new[:, index])
                    Dice_dict[classes].append(metric['Dice'])
                    if 'HD' in metric.keys():
                        HD_dict[classes].append(metric['HD'])
            # plt.figure(figsize=(8, 6))
            # plt.subplot(221)
            # plt.title('mean dice of PZ: {:.3f}'.format(sum(Dice_dict['PZ']) / len(Dice_dict['PZ'])))
            # plt.hist(Dice_dict['PZ'], bins=20)
            # plt.subplot(222)
            # plt.title('mean dice of TZ: {:.3f}'.format(sum(Dice_dict['TZ']) / len(Dice_dict['TZ'])))
            # plt.hist(Dice_dict['TZ'], bins=20)
            # plt.subplot(223)
            # plt.title('mean dice of DPU: {:.3f}'.format(sum(Dice_dict['DPU']) / len(Dice_dict['DPU'])))
            # plt.hist(Dice_dict['DPU'], bins=20)
            # plt.subplot(224)
            # plt.title('mean dice of AFS: {:.3f}'.format(sum(Dice_dict['AFS']) / len(Dice_dict['AFS'])))
            # plt.hist(Dice_dict['AFS'], bins=20)
            # plt.show()

            # for idx in range(len(CV_HD_dict)):
            #     print('CV-{}:'.format(idx), end='\t')
            #     print('Dice:', end=' ')
            #     print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(CV_Dice_dict[idx]['PZ']) / len(CV_Dice_dict[idx]['PZ']),
            #                                                      sum(CV_Dice_dict[idx]['TZ']) / len(CV_Dice_dict[idx]['TZ']),
            #                                                      sum(CV_Dice_dict[idx]['DPU']) / len(CV_Dice_dict[idx]['DPU']),
            #                                                      sum(CV_Dice_dict[idx]['AFS']) / len(CV_Dice_dict[idx]['AFS'])), end='\t')
            #     print('HD:', end=' ')
            #     print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(CV_HD_dict[idx]['PZ']) / len(CV_HD_dict[idx]['PZ']),
            #                                                      sum(CV_HD_dict[idx]['TZ']) / len(CV_HD_dict[idx]['TZ']),
            #                                                      sum(CV_HD_dict[idx]['DPU']) / len(CV_HD_dict[idx]['DPU']),
            #                                                      sum(CV_HD_dict[idx]['AFS']) / len(CV_HD_dict[idx]['AFS'])))
            # print('mean:', end='\t')
            # print('Dice:', end=' ')
            # print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(Dice_dict['PZ']) / len(Dice_dict['PZ']),
            #                                                  sum(Dice_dict['TZ']) / len(Dice_dict['TZ']),
            #                                                  sum(Dice_dict['DPU']) / len(Dice_dict['DPU']),
            #                                                  sum(Dice_dict['AFS']) / len(Dice_dict['AFS'])), end='\t')
            # print('HD:', end=' ')
            # print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(HD_dict['PZ']) / len(HD_dict['PZ']),
            #                                                  sum(HD_dict['TZ']) / len(HD_dict['TZ']),
            #                                                  sum(HD_dict['DPU']) / len(HD_dict['DPU']),
            #                                                  sum(HD_dict['AFS']) / len(HD_dict['AFS'])))
            print(len(HD_dict['PZ']), len(HD_dict['TZ']), len(HD_dict['DPU']), len(HD_dict['AFS']))
            print(df_test)
            print(HD_dict['AFS'])
            print('no_label_predict: {}, label_no_predict: {}'.format(no_label_predict, label_no_predict))



