import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from CnnTools.T4T.Utility.Data import *

from BasicTool.MeDIT.Statistics import BinarySegmentation
from BasicTool.MeDIT.ArrayProcess import ExtractPatch

# from SegModel.UNet import UNet, UNet25D
# from SegModel.AttenUnet import AttenUNet
from SegModel.MultiSeg import MultiSeg, MultiAttenSeg
from SegModel.WNet import WNet
from SegModel.MSUNet import MSUNet
from SegModel.TwoUNet import TwoUNet
from ModelfromGitHub.UNet.unet_model import UNet, UNet25D
from SegModel.Atten import AttU_Net

from Statistics.Metric import Dice
from PreProcess.Nii2NPY import ROIOneHot
from PreProcess.DistanceMapNumpy import DistanceMap
from PreProcess.DistanceMapNumpy import KeepLargest


def ModelTest(model, model_folder, epoch, data_type='train'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    batch_size = 2

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}_name.csv'.format(data_type))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    # data.AddOne(Image2D(data_root + '/RoiSliceNoOneHot', shape=input_shape), is_input=False)
    # data.AddOne(Image2D(model_step1_pred, shape=input_shape, is_roi=True))
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model.to(device)

    model.load_state_dict(torch.load(model_folder + epoch))

    pred_list, label_list = [], []
    model.eval()
    with torch.no_grad():
        for inputs, outputs in data_loader:
            # mask = torch.squeeze(outputs[1], dim=1)
            # roi_dilate = torch.from_numpy(binary_dilation(mask, structure=np.ones((1, 11, 11))))
            # inputs = inputs * torch.unsqueeze(roi_dilate, dim=1)

            inputs = MoveTensorsToDevice(inputs, device)
            # outputs = MoveTensorsToDevice(outputs, device)

            # inputs = MoveTensorsToDevice(inputs, device)

            preds2 = model(inputs)
            preds2 = torch.softmax(preds2, dim=1)
            if isinstance(preds2, tuple):
                preds2 = preds2[-1]

            pred_list.extend(list(preds2.cpu().data.numpy()))
            # label_list.extend(list(outputs.cpu().data.numpy()))
            label_list.extend(list(outputs.numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_preds.npy'.format(data_type)), np.array(pred_list))
    np.save(os.path.join(result_folder, '{}_label.npy'.format(data_type)), np.array(label_list))


def ShowAtten(model_folder, data_type='train'):
    from BasicTool.MeDIT.Normalize import Normalize01
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    batch_size = 1

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}_name.csv'.format(data_type))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = AttenUNet(1, 5).to(device)
    model.load_state_dict(torch.load(model_folder + '/39--12.661804.pt'))

    model.eval()
    for inputs, outputs in data_loader:
        inputs = MoveTensorsToDevice(inputs, device)
        outputs = MoveTensorsToDevice(outputs, device)

        atten5, atten6, atten7, preds = model(inputs)
        index5 = torch.argmax(F.adaptive_avg_pool2d(atten5, 1))
        index6 = torch.argmax(F.adaptive_avg_pool2d(atten6, 1))
        index7 = torch.argmax(F.adaptive_avg_pool2d(atten7, 1))

        plt.figure(figsize=(12, 8))
        # label
        plt.subplot(241)
        plt.axis('off')
        plt.imshow(np.squeeze(inputs.cpu().data.numpy()), cmap='gray')
        plt.contour(np.squeeze(outputs[:, 0, ...].cpu().data.numpy()), colors='r')
        plt.contour(np.squeeze(outputs[:, 1, ...].cpu().data.numpy()), colors='y')
        plt.contour(np.squeeze(outputs[:, 2, ...].cpu().data.numpy()), colors='g')
        plt.contour(np.squeeze(outputs[:, 3, ...].cpu().data.numpy()), colors='b')

        plt.subplot(242)
        plt.axis('off')
        plt.imshow(Normalize01(atten5[0, index5, ...].cpu().data.numpy()), cmap='jet')

        plt.subplot(243)
        plt.axis('off')
        plt.imshow(Normalize01(atten6[0, index6, ...].cpu().data.numpy()), cmap='jet')

        plt.subplot(244)
        plt.axis('off')
        plt.imshow(Normalize01(atten7[0, index7, ...].cpu().data.numpy()), cmap='jet')

        plt.subplot(245)
        plt.axis('off')
        plt.imshow(np.squeeze(preds[:, 1, ...].cpu().data.numpy()), cmap='gray')
        plt.subplot(246)
        plt.axis('off')
        plt.imshow(np.squeeze(preds[:, 2, ...].cpu().data.numpy()), cmap='gray')
        plt.subplot(247)
        plt.axis('off')
        plt.imshow(np.squeeze(preds[:, 3, ...].cpu().data.numpy()), cmap='gray')
        plt.subplot(248)
        plt.axis('off')
        plt.imshow(np.squeeze(preds[:, 4, ...].cpu().data.numpy()), cmap='gray')

        plt.show()

        # atten_pz, atten_cg, atten_u, atten_as, preds = model(inputs)
        # for index in range(len(atten_as)):
        #     index5 = torch.argmax(F.adaptive_avg_pool2d(atten_pz[index], output_size=1))
        #     index6 = torch.argmax(F.adaptive_avg_pool2d(atten_cg[index], output_size=1))
        #     index7 = torch.argmax(F.adaptive_avg_pool2d(atten_u [index], output_size=1))
        #     index8 = torch.argmax(F.adaptive_avg_pool2d(atten_as[index], output_size=1))
        #
        #     plt.figure(figsize=(12, 13))
        #     # label
        #     plt.subplot(331)
        #     plt.title('label')
        #     plt.axis('off')
        #     plt.imshow(np.squeeze(inputs.cpu().data.numpy()), cmap='gray')
        #     plt.contour(np.squeeze(outputs[:, 0, ...].cpu().data.numpy()), colors='r')
        #     plt.contour(np.squeeze(outputs[:, 1, ...].cpu().data.numpy()), colors='y')
        #     plt.contour(np.squeeze(outputs[:, 2, ...].cpu().data.numpy()), colors='g')
        #     plt.contour(np.squeeze(outputs[:, 3, ...].cpu().data.numpy()), colors='b')
        #
        #     plt.subplot(332)
        #     plt.title('attention map of pz')
        #     plt.axis('off')
        #     plt.imshow(Normalize01(atten_pz[index][0, index5, ...].cpu().data.numpy()), cmap='jet')
        #
        #     plt.subplot(333)
        #     plt.title('attention map of cg')
        #     plt.axis('off')
        #     plt.imshow(Normalize01(atten_cg[index][0, index6, ...].cpu().data.numpy()), cmap='jet')
        #
        #     plt.subplot(334)
        #     plt.title('attention map of u')
        #     plt.axis('off')
        #     plt.imshow(Normalize01(atten_u[index][0, index7, ...].cpu().data.numpy()), cmap='jet')
        #
        #     plt.subplot(335)
        #     plt.title('attention map of asmf')
        #     plt.axis('off')
        #     plt.imshow(Normalize01(atten_as[index][0, index8, ...].cpu().data.numpy()), cmap='jet')
        #
        #     plt.subplot(336)
        #     plt.title('prediction of pz')
        #     plt.axis('off')
        #     plt.imshow(np.squeeze(preds[:, 0, ...].cpu().data.numpy()), cmap='gray')
        #     plt.subplot(337)
        #     plt.title('prediction of cg')
        #     plt.axis('off')
        #     plt.imshow(np.squeeze(preds[:, 1, ...].cpu().data.numpy()), cmap='gray')
        #     plt.subplot(338)
        #     plt.title('prediction of u')
        #     plt.axis('off')
        #     plt.imshow(np.squeeze(preds[:, 2, ...].cpu().data.numpy()), cmap='gray')
        #     plt.subplot(339)
        #     plt.title('prediction of asmf')
        #     plt.axis('off')
        #     plt.imshow(np.squeeze(preds[:, 3, ...].cpu().data.numpy()), cmap='gray')
        #
        #     plt.show()


def ShoweResult(model_folder, data_type='train', num_pred=1, save_path=r''):
    if save_path and not os.path.exists(save_path):
        os.mkdir(save_path)
    result_folder = os.path.join(model_folder, 'Result')
    label_path = os.path.join(result_folder, '{}_label.npy'.format(data_type))
    if num_pred == 1:
        pred_path = os.path.join(result_folder, '{}_preds.npy'.format(data_type))

        label = np.load(label_path)
        pred = np.load(pred_path)

        for index in range(label.shape[0]):
            pred_index = ROIOneHot(np.argmax(pred[index], axis=0))

            plt.figure(figsize=(12, 4))

            plt.subplot(251)
            plt.axis('off')
            plt.imshow(label[index][0, ...], cmap='gray')

            plt.subplot(252)
            plt.axis('off')
            plt.imshow(label[index][1, ...], cmap='gray')

            plt.subplot(253)
            plt.axis('off')
            plt.imshow(label[index][2, ...], cmap='gray')

            plt.subplot(254)
            plt.axis('off')
            plt.imshow(label[index][3, ...], cmap='gray')

            plt.subplot(255)
            plt.axis('off')
            plt.imshow(label[index][4, ...], cmap='gray')

            plt.subplot(256)
            plt.title('{:.3f}'.format(Dice(pred_index[0], label[index][0])))
            plt.axis('off')
            plt.imshow(pred[index][0, ...], cmap='gray')

            plt.subplot(257)
            plt.title('{:.3f}'.format(Dice(pred_index[1], label[index][1])))
            plt.axis('off')
            plt.imshow(pred[index][1, ...], cmap='gray')

            plt.subplot(258)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(pred_index[2], label[index][2])))
            plt.imshow(pred[index][2, ...], cmap='gray')

            plt.subplot(259)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(pred_index[3], label[index][3])))
            plt.imshow(pred[index][3, ...], cmap='gray')

            plt.subplot(2, 5, 10)
            plt.axis('off')
            plt.title('{:.3f}'.format(Dice(pred_index[4], label[index][4])))
            plt.imshow(pred[index][4, ...], cmap='gray')
            if save_path:
                plt.savefig(os.path.join(save_path, 'test_{}.jpg'.format(index)))
                plt.close()
            else:
                plt.show()

    elif num_pred == 2:
        pred_path1 = os.path.join(result_folder, '{}_preds1.npy'.format(data_type))
        pred_path2 = os.path.join(result_folder, '{}_preds2.npy'.format(data_type))

        label = np.load(label_path)
        pred1 = np.load(pred_path1)
        pred2 = np.load(pred_path2)
        for index in range(label.shape[0]):
            pred_index = ROIOneHot(np.argmax(pred2[index], axis=0))
            plt.figure(figsize=(12, 4))

            ############################################################################################################
            plt.subplot(351)
            plt.axis('off')
            plt.imshow(label[index][0, ...], cmap='gray')
            plt.subplot(352)
            plt.axis('off')
            plt.imshow(label[index][1, ...], cmap='gray')
            plt.subplot(353)
            plt.axis('off')
            plt.imshow(label[index][2, ...], cmap='gray')
            plt.subplot(354)
            plt.axis('off')
            plt.imshow(label[index][3, ...], cmap='gray')
            plt.subplot(355)
            plt.axis('off')
            plt.imshow(label[index][4, ...], cmap='gray')

            ############################################################################################################
            plt.subplot(356)
            plt.axis('off')
            plt.imshow(pred1[index][0, ...], cmap='gray')
            plt.subplot(357)
            plt.axis('off')
            plt.imshow(pred1[index][1, ...], cmap='gray')
            plt.subplot(358)
            plt.axis('off')
            plt.imshow(pred1[index][2, ...], cmap='gray')

            ############################################################################################################
            plt.subplot(3, 5, 11)
            plt.title('{:.3f}'.format(Dice(pred_index[0], label[index][0])))
            plt.axis('off')
            plt.imshow(pred2[index][0, ...], cmap='gray')

            plt.subplot(3, 5, 12)
            plt.title('{:.3f}'.format(Dice(pred_index[1], label[index][1])))
            plt.axis('off')
            plt.imshow(pred2[index][1, ...], cmap='gray')

            plt.subplot(3, 5, 13)
            plt.title('{:.3f}'.format(Dice(pred_index[2], label[index][2])))
            plt.axis('off')
            plt.imshow(pred2[index][2, ...], cmap='gray')

            plt.subplot(3, 5, 14)
            plt.title('{:.3f}'.format(Dice(pred_index[3], label[index][3])))
            plt.axis('off')
            plt.imshow(pred2[index][3, ...], cmap='gray')

            plt.subplot(3, 5, 15)
            plt.title('{:.3f}'.format(Dice(pred_index[4], label[index][4])))
            plt.axis('off')
            plt.imshow(pred2[index][4, ...], cmap='gray')

            # plt.savefig()

            plt.show()


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'
    # model_step1_pred = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/CaseResultDilate'
    # model_step1_pred = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/UNet_0311_step1/DistanceMap/DisMap'


    model_path = model_root + '/UNet_0330_weightedloss'
    epoch = '/13-2.679143.pt'
    model = UNet25D(n_channels=1, n_classes=5, bilinear=True, factor=2)


    ModelTest(model, model_path, epoch, 'train')
    ModelTest(model, model_path, epoch, 'val')
    ModelTest(model, model_path, epoch, 'test')

    ShoweResult(model_path, data_type='test', num_pred=1, save_path=os.path.join(model_path, 'Image'))
    # ShoweResult(model_path, data_type='train', num_pred=1)


    # ShowAtten(model_path, data_type='train')
    # ShowAtten(model_path, data_type='val')
    # ShowAtten(model_path, data_type='test')










