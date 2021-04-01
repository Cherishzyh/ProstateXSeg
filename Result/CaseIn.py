import torch
import matplotlib.pyplot as plt

from CnnTools.T4T.Utility.Data import *


from Statistics.Metric import Dice

class ProstateXSeg():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def CropData(self, data, crop_shape, center, is_roi=False, slice_num=1):
        from BasicTool.MeDIT.ArrayProcess import ExtractPatch
        from BasicTool.MeDIT.ArrayProcess import ExtractBlock

        # Crop
        if len(data.shape) == 2:
            t2_crop, _ = ExtractPatch(data, crop_shape, center_point=center)
        elif len(data.shape) == 3:
            center = [center[0], center[1], -1]
            crop_shape = [crop_shape[0], crop_shape[1], slice_num]
            t2_crop, _ = ExtractBlock(data, crop_shape, center_point=center)
        else:
            raise Exception

        if not is_roi:
            # Normalization
            t2_crop -= np.mean(t2_crop)
            t2_crop /= np.std(t2_crop)

        return t2_crop

    def GetCenter(self, roi):
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

    def GetROICenter(self, roi):
        '''
        :param roi: 2D roi include multi-class
        :return: the center of entire roi
        '''

        assert len(roi.shape) == 2
        roi_binary = (roi >= 1).astype(int)
        center = self.GetCenter(roi_binary)
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
        return np.array(roi_list, dtype=np.int)

    def Nii2NPY(self, case, data_path, des_path, slice_num=1, is_save=False):
        from BasicTool.MeDIT.SaveAndLoad import LoadImage

        t2_path = os.path.join(data_path, 't2.nii')
        roi_path = os.path.join(data_path, 'roi.nii.gz')

        _, t2, _ = LoadImage(t2_path)
        _, roi, _ = LoadImage(roi_path, dtype=np.int)
        # t2 = t2.transpose((2, 0, 1))
        # roi = roi.transpose((2, 0, 1))

        t2_save_path = os.path.join(des_path, 'T2Slice')
        roi_save_path = os.path.join(des_path, 'RoiSlice')

        start_slice = int((slice_num - 1) / 2)
        end_slice = t2.shape[-1] - start_slice
        t2_list, roi_list = [], []
        for slice in range(start_slice, end_slice):

            # t2 slice
            t2_slice = np.squeeze(t2[..., int(slice - start_slice): int(slice + start_slice)+1])

            # roi slice
            roi_slice = roi[..., slice]

            if np.sum(roi_slice) == 0:
                continue

            center = self.GetROICenter(roi_slice)

            t2_slice = self.CropData(t2_slice, self.input_shape, center=center, slice_num=slice_num)
            roi_slice = self.CropData(roi_slice, self.input_shape, center=center, is_roi=True)

            roi_slice = self.ROIOneHot(roi_slice)

            if len(t2_slice.shape) == 3:
                t2_slice = t2_slice.transpose((2, 0, 1))
            if len(t2_slice.shape) == 2:
                t2_slice = t2_slice[np.newaxis, ...]

            t2_list.append(t2_slice)
            roi_list.append(roi_slice)

            if is_save:
                t2_npy_path = os.path.join(t2_save_path, '{}_-_slice{}.npy'.format(case, slice))
                roi_npy_path = os.path.join(roi_save_path, '{}_-_slice{}.npy'.format(case, slice))
                np.save(t2_npy_path, t2_slice)
                np.save(roi_npy_path, roi_slice)

        return np.array(t2_list), np.array(roi_list, dtype=int)

    def NPY2NPY(self, case, data_path, n_class=5):
        t2_path = os.path.join(data_path, 'T2Slice/{}.npy'.format(case))
        roi_path = os.path.join(data_path, 'RoiSlice/{}.npy'.format(case))
        if not os.path.exists(t2_path):
            t2_arr = None
            roi_arr = None
            return t2_arr, roi_arr
        else:
            t2_arr = np.load(t2_path)
            roi_arr = np.load(roi_path)

            if np.ndim(t2_arr) == 3:
                t2_arr = t2_arr[np.newaxis, ...]
            if np.ndim(roi_arr) == 3:

                if n_class == 5:
                    roi_arr = roi_arr[np.newaxis, ...]
                else:
                    other_roi = np.sum(roi_arr[2:, ...], axis=0)
                    roi_arr = np.concatenate([roi_arr[0:1, ...], roi_arr[1:2, ...], other_roi[np.newaxis, ...]], axis=0)
                    roi_arr = roi_arr[np.newaxis, ...]

            return t2_arr, roi_arr

    def run(self, case, model, model_path, weights_path, inputs, outputs, is_save=False):
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        weights_path = os.path.join(model_path, weights_path)

        model.load_state_dict(torch.load(weights_path))

        model.eval()

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        # if isinstance(outputs, np.ndarray):
        #     outputs = torch.from_numpy(outputs)

        inputs = MoveTensorsToDevice(inputs, device)
        # outputs = MoveTensorsToDevice(outputs, device)


        print('******predicting {}, (｀・ω・´)******'.format(case))
        preds = model(inputs)

        if is_save:
            result_folder = os.path.join(model_path, 'CaseResult')
            if not os.path.exists(os.path.join(model_path, 'CaseResult')):
                os.mkdir(os.path.join(model_path, 'CaseResult'))

            np.save(os.path.join(result_folder, '{}.npy'.format(case)), np.squeeze(preds.cpu().data.numpy()))
            # np.save(os.path.join(result_folder, '{}_label.npy'.format(case)), outputs.cpu().data.numpy())

        return preds


if __name__ == '__main__':

    # from SegModel.UNet import UNet, UNet25D
    from ModelfromGitHub.UNet.unet_model import UNet25D
    from PreProcess.Nii2NPY import ROIOneHot

    # data_path = r'W:\Public Datasets\PROSTATEx_Seg\Seg'

    # model = UNet(1, 5)          # one slice
    # model = UNet(3, 5)        # three slice
    # model = MultiSeg(1, 1)    # one slice
    # model = MSUNet(1, 5)      # one slice
    # model = WNet(1, 3, 1, 5)  # one slice
    # model = UNet25D(1, 5)
    # model = AttenUNet(1, 5)
    # model = TwoUNet(1, 5)

    def case_test():
        seg = ProstateXSeg((200, 200))
        data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OriginalData'
        model_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'

        model = UNet25D(n_channels=1, n_classes=5, bilinear=True, factor=2)

        dice1_list, dice2_list, dice3_list, dice4_list, dice5_list = [], [], [], [], []

        # df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'train_case_name.csv'))
        # df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'val_case_name.csv'))
        df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'test_case_name.csv'))
        case_list = df.values.tolist()[0]
        for case in case_list:
            t2_arr, roi_arr = seg.Nii2NPY(case, os.path.join(data_path, case), r'', slice_num=3)
            # UNet
            # 13-2.679143.pt
            # 26-1.161448.pt   UNet_0330
            preds = seg.run(case, model,
                            model_path=os.path.join(model_folder, 'UNet_0330_weightedloss'),
                            weights_path='13-2.679143.pt',
                            inputs=t2_arr,
                            outputs=roi_arr,
                            is_save=False)
            if isinstance(preds, tuple):
                preds = preds[-1]
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1).cpu().data.numpy()
            preds = ROIOneHot(preds)
            # roi_arr = np.argmax(roi_arr, axis=1)
            # roi_arr = np.clip(roi_arr, a_min=0, a_max=2)
            # roi_arr = ROIOneHot(roi_arr, roi_class=[0, 1, 2])

            dice1_list.append(Dice(preds[:, 0, ...], roi_arr[:, 0, ...]))
            dice2_list.append(Dice(preds[:, 1, ...], roi_arr[:, 1, ...]))
            dice3_list.append(Dice(preds[:, 2, ...], roi_arr[:, 2, ...]))
            dice4_list.append(Dice(preds[:, 3, ...], roi_arr[:, 3, ...]))
            dice5_list.append(Dice(preds[:, 4, ...], roi_arr[:, 4, ...]))

        print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(dice1_list) / len(dice1_list), sum(dice2_list) / len(dice2_list),
                                                              sum(dice3_list) / len(dice3_list), sum(dice4_list) / len(dice4_list),
                                                              sum(dice5_list) / len(dice5_list)))



    case_test()


    def slice_test():
        seg = ProstateXSeg((200, 200))
        data_path = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
        model_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
        result_folder = os.path.join(os.path.join(model_folder, 'UNet_0311_step1', 'CaseResult'))
        if not os.path.exists(os.path.join(os.path.join(model_folder, 'UNet_0311_step1', 'CaseResult'))):
            os.mkdir(os.path.join(os.path.join(model_folder, 'UNet_0311_step1', 'CaseResult')))

        model = UNet25D(n_channels=1, n_classes=3, bilinear=True, factor=2)

        dice1_list, dice2_list, dice3_list, dice4_list, dice5_list = [], [], [], [], []

        df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'train_name.csv'))
        # df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'val_name.csv'))
        # df = pd.read_csv(os.path.join(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice', 'test_name.csv'))
        case_list = df.values.tolist()[0]
        for index, case in enumerate(case_list):
            t2_arr, roi_arr = seg.NPY2NPY(case, data_path, n_class=3)

            if not isinstance(t2_arr, np.ndarray):
                continue
            # UNet
            preds = seg.run(case, model,
                            model_path=os.path.join(model_folder, 'UNet_0311_step1'),
                            weights_path='18--11.511773.pt',
                            inputs=t2_arr,
                            outputs=roi_arr,
                            is_save=False)
            if isinstance(preds, tuple):
                preds = preds[-1]
            preds = torch.argmax(preds, dim=1).cpu().data.numpy()
            preds = ROIOneHot(preds, roi_class=(0, 1, 2))

            # plt.subplot(221)
            # plt.imshow(t2_arr[0, 1, ...], cmap='gray')
            # plt.contour(np.argmax(roi_arr, axis=1)[0, ...])
            # plt.subplot(222)
            # plt.imshow(preds[0, 0, ...], cmap='gray')
            # # plt.contour(outputs_roi[index])
            # plt.subplot(223)
            # plt.imshow(preds[0, 1, ...], cmap='gray')
            # # plt.contour(outputs_roi[index])
            # plt.subplot(224)
            # plt.imshow(preds[0, 2, ...], cmap='gray')
            # # plt.contour(outputs_roi[index])
            # plt.show()

            # print('#############   {} / 1727  #############'.format(index + 1))
            np.save(os.path.join(result_folder, '{}.npy'.format(case)), np.squeeze(preds))
        #
        #     dice1_list.append(Dice(preds[:, 0, ...], roi_arr[:, 0, ...]))
        #     dice2_list.append(Dice(preds[:, 1, ...], roi_arr[:, 1, ...]))
        #     dice3_list.append(Dice(preds[:, 2, ...], roi_arr[:, 2, ...]))
        #
        # print('{:.3f}, {:.3f}, {:.3f}'.format(sum(dice1_list) / len(dice1_list), sum(dice2_list) / len(dice2_list),
        #                                       sum(dice3_list) / len(dice3_list)))
    # slice_test()


    # plt.subplot(221)
    # plt.title('aver: {:.3f}'.format(sum(dice2_list) / len(dice2_list)))
    # plt.hist(dice2_list, bins=20)
    # plt.subplot(222)
    # plt.title('aver: {:.3f}'.format(sum(dice3_list) / len(dice3_list)))
    # plt.hist(dice3_list, bins=20)
    # plt.subplot(223)
    # plt.title('aver: {:.3f}'.format(sum(dice4_list) / len(dice4_list)))
    # plt.hist(dice4_list, bins=20)
    # plt.subplot(224)
    # plt.title('aver: {:.3f}'.format(sum(dice5_list) / len(dice5_list)))
    # plt.hist(dice5_list, bins=20)
    # plt.show()






