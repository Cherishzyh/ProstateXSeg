import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from SSHProject.CnnTools.T4T.Utility.Data import *

from MeDIT.Statistics import BinarySegmentation
from MeDIT.ArrayProcess import ExtractPatch

from Statistics.Metric import Dice
from PreProcess.Nii2NPY import ROIOneHot
from MeDIT.Others import IterateCase


def ShoweResult(model_folder, data_type='train', num_pred=1, save_path=r''):
    if save_path and not os.path.exists(save_path):
        os.mkdir(save_path)
    result_folder = os.path.join(model_folder, 'Result')
    label_path = os.path.join(result_folder, '{}_label.npy'.format(data_type))
    t2_path = os.path.join(result_folder, '{}_t2.npy'.format(data_type))
    if num_pred == 1:
        pred_path = os.path.join(result_folder, '{}_pred_2.npy'.format(data_type))
        label = np.load(label_path)
        pred = np.load(pred_path)
        t2 = np.load(t2_path)

        for index in range(label.shape[0]):
            pred_pro = np.clip(np.argmax(pred[index], axis=0), a_min=0, a_max=1.)
            label_pro = np.clip(np.argmax(label[index], axis=0), a_min=0, a_max=1.)
            pred_index = ROIOneHot(np.argmax(pred[index], axis=0))
            t2_index = t2[index][1]
            label_index = label[index]

            plt.figure(figsize=(12, 6))

            plt.subplot(231)
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_index[1, ...], colors='r')
            plt.contour(label_index[2, ...], colors='g')
            plt.contour(label_index[3, ...], colors='b')
            plt.contour(label_index[4, ...], colors='y')

            plt.subplot(232)
            plt.title('Pro: {:.3f}'.format(Dice(label_pro, pred_pro)))
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_pro, colors='darkorange')
            plt.contour(pred_pro, colors='m')

            plt.subplot(233)
            plt.title('PZ: {:.3f}'.format(Dice(pred_index[1], label_index[1])))
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_index[1], colors='r')
            plt.contour(pred_index[1], colors='m')

            plt.subplot(234)
            plt.title('CG: {:.3f}'.format(Dice(pred_index[2], label_index[2])))
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_index[2], colors='g')
            plt.contour(pred_index[2], colors='m')

            plt.subplot(235)
            plt.title('U: {:.3f}'.format(Dice(pred_index[3], label_index[3])))
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_index[3], colors='b')
            plt.contour(pred_index[3], colors='m')

            plt.subplot(236)
            plt.title('AMSF: {:.3f}'.format(Dice(pred_index[4], label_index[4])))
            plt.axis('off')
            plt.imshow(t2_index, cmap='gray')
            plt.contour(label_index[4], colors='y')
            plt.contour(pred_index[4], colors='m')

            if save_path:
                plt.savefig(os.path.join(save_path, 'test_{}.jpg'.format(index)), bbox_inches='tight', pad_inches=0.05)
                plt.close()
            else:
                plt.show()
    #
    # elif num_pred == 2:
    #     pred_path1 = os.path.join(result_folder, '{}_preds1.npy'.format(data_type))
    #     pred_path2 = os.path.join(result_folder, '{}_preds2.npy'.format(data_type))
    #
    #     label = np.load(label_path)
    #     pred1 = np.load(pred_path1)
    #     pred2 = np.load(pred_path2)
    #     for index in range(label.shape[0]):
    #         pred_index = ROIOneHot(np.argmax(pred2[index], axis=0))
    #         plt.figure(figsize=(12, 4))
    #
    #         ############################################################################################################
    #         plt.subplot(351)
    #         plt.axis('off')
    #         plt.imshow(label[index][0, ...], cmap='gray')
    #         plt.subplot(352)
    #         plt.axis('off')
    #         plt.imshow(label[index][1, ...], cmap='gray')
    #         plt.subplot(353)
    #         plt.axis('off')
    #         plt.imshow(label[index][2, ...], cmap='gray')
    #         plt.subplot(354)
    #         plt.axis('off')
    #         plt.imshow(label[index][3, ...], cmap='gray')
    #         plt.subplot(355)
    #         plt.axis('off')
    #         plt.imshow(label[index][4, ...], cmap='gray')
    #
    #         ############################################################################################################
    #         plt.subplot(356)
    #         plt.axis('off')
    #         plt.imshow(pred1[index][0, ...], cmap='gray')
    #         plt.subplot(357)
    #         plt.axis('off')
    #         plt.imshow(pred1[index][1, ...], cmap='gray')
    #         plt.subplot(358)
    #         plt.axis('off')
    #         plt.imshow(pred1[index][2, ...], cmap='gray')
    #
    #         ############################################################################################################
    #         plt.subplot(3, 5, 11)
    #         plt.title('{:.3f}'.format(Dice(pred_index[0], label[index][0])))
    #         plt.axis('off')
    #         plt.imshow(pred2[index][0, ...], cmap='gray')
    #
    #         plt.subplot(3, 5, 12)
    #         plt.title('{:.3f}'.format(Dice(pred_index[1], label[index][1])))
    #         plt.axis('off')
    #         plt.imshow(pred2[index][1, ...], cmap='gray')
    #
    #         plt.subplot(3, 5, 13)
    #         plt.title('{:.3f}'.format(Dice(pred_index[2], label[index][2])))
    #         plt.axis('off')
    #         plt.imshow(pred2[index][2, ...], cmap='gray')
    #
    #         plt.subplot(3, 5, 14)
    #         plt.title('{:.3f}'.format(Dice(pred_index[3], label[index][3])))
    #         plt.axis('off')
    #         plt.imshow(pred2[index][3, ...], cmap='gray')
    #
    #         plt.subplot(3, 5, 15)
    #         plt.title('{:.3f}'.format(Dice(pred_index[4], label[index][4])))
    #         plt.axis('off')
    #         plt.imshow(pred2[index][4, ...], cmap='gray')
    #
    #         # plt.savefig()
    #
    #         plt.show()


def ShowAtten(model, model_folder, epoch, data_type='train', save_path=r''):
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

    model = model.to(device)
    model.load_state_dict(torch.load(model_folder + epoch))

    if save_path:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    model.eval()
    for index, (inputs, outputs) in enumerate(data_loader):
        inputs = MoveTensorsToDevice(inputs, device)
        outputs = MoveTensorsToDevice(outputs, device)

        preds0, preds1, atten5, atten6, atten7 = model(inputs, 15)
        atten5 = F.interpolate(atten5, size=preds1.shape[2:], mode='bilinear')
        atten6 = F.interpolate(atten6, size=preds1.shape[2:], mode='bilinear')
        atten7 = F.interpolate(atten7, size=preds1.shape[2:], mode='bilinear')
        # index5 = torch.argmax(F.adaptive_avg_pool2d(atten5, 1))
        # index6 = torch.argmax(F.adaptive_avg_pool2d(atten6, 1))
        # index7 = torch.argmax(F.adaptive_avg_pool2d(atten7, 1))

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
        plt.imshow(Normalize01(atten5[0, 0, ...].cpu().data.numpy()), cmap='jet', vmin=0, vmax=1)

        plt.subplot(243)
        plt.axis('off')
        plt.imshow(Normalize01(atten6[0, 0, ...].cpu().data.numpy()), cmap='jet', vmin=0, vmax=1)

        plt.subplot(244)
        plt.axis('off')
        plt.imshow(Normalize01(atten7[0, 0, ...].cpu().data.numpy()), cmap='jet', vmin=0, vmax=1)

        plt.subplot(245)
        plt.axis('off')
        plt.imshow(np.squeeze(torch.softmax(preds1[:, 1, ...], dim=1).cpu().data.numpy()), cmap='gray')
        plt.subplot(246)
        plt.axis('off')
        plt.imshow(np.squeeze(torch.softmax(preds1[:, 2, ...], dim=1).cpu().data.numpy()), cmap='gray')
        plt.subplot(247)
        plt.axis('off')
        plt.imshow(np.squeeze(torch.softmax(preds1[:, 3, ...], dim=1).cpu().data.numpy()), cmap='gray')
        plt.subplot(248)
        plt.axis('off')
        plt.imshow(np.squeeze(torch.softmax(preds1[:, 4, ...], dim=1).cpu().data.numpy()), cmap='gray')

        if save_path:
            plt.savefig(os.path.join(save_path, 'test_{}.jpg'.format(index)))
            plt.close()
        else:
            plt.show()


def ComputeDice(result_path, t2_folder, data_type):
    if len(os.listdir(result_path)) == 9:
        preds_arr = np.load(os.path.join(result_path, '{}_preds2.npy'.format(data_type)))
        label_arr = np.load(os.path.join(result_path, '{}_label.npy'.format(data_type)))
    else:
        preds_arr = np.load(os.path.join(result_path, '{}_preds.npy'.format(data_type)))
        label_arr = np.load(os.path.join(result_path, '{}_label.npy'.format(data_type)))

    dice1_list, dice2_list, dice3_list, dice4_list, dice5_list = [], [], [], [], []
    # case_name = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/{}_name.csv'.format(data_type))
    # case_list = sorted(case_name.loc[0].tolist())
    for index in range(preds_arr.shape[0]):
        pred = ROIOneHot(np.argmax(preds_arr[index], axis=0))
        label = label_arr[index]
        # if os.path.exists(os.path.join(t2_folder, case_list[index] + '.npy')):
            # t2 = np.squeeze(np.load(os.path.join(t2_folder, case_list[index] + '.npy')))
            # t2_crop, _ = ExtractPatch(t2, (200, 200))

        dice1_list.append(Dice(pred[0], label[0]))
        dice2_list.append(Dice(pred[1], label[1]))
        dice3_list.append(Dice(pred[2], label[2]))
        dice4_list.append(Dice(pred[3], label[3]))
        dice5_list.append(Dice(pred[4], label[4]))

            ########################################show result#############################################################
            # plt.subplot(231)
            # plt.axis('off')
            # plt.imshow(pred[0], cmap='gray', vmin=0., vmax=1.)
            # plt.contour(label[0], colors='r')
            # plt.subplot(232)
            # plt.axis('off')
            # plt.imshow(pred[1], cmap='gray', vmin=0., vmax=1.)
            # plt.contour(label[1], colors='r')
            # plt.subplot(234)
            # plt.axis('off')
            # plt.imshow(pred[2], cmap='gray', vmin=0., vmax=1.)
            # plt.contour(label[2], colors='r')
            # plt.subplot(235)
            # plt.axis('off')
            # plt.imshow(pred[3], cmap='gray', vmin=0., vmax=1.)
            # plt.contour(label[3], colors='r')
            # plt.subplot(233)
            # plt.axis('off')
            # plt.imshow(t2_crop, cmap='gray')
            # plt.contour(label[0], colors='r')
            # plt.contour(label[1], colors='y')
            # plt.contour(label[2], colors='b')
            # plt.contour(label[3], colors='g')
            # plt.show()

    plt.subplot(221)
    plt.title('aver: {:.3f}'.format(sum(dice2_list) / len(dice2_list)))
    plt.hist(dice2_list, bins=20)
    plt.subplot(222)
    plt.title('aver: {:.3f}'.format(sum(dice3_list) / len(dice3_list)))
    plt.hist(dice3_list, bins=20)
    plt.subplot(223)
    plt.title('aver: {:.3f}'.format(sum(dice4_list) / len(dice4_list)))
    plt.hist(dice4_list, bins=20)
    plt.subplot(224)
    plt.title('aver: {:.3f}'.format(sum(dice5_list) / len(dice5_list)))
    plt.hist(dice5_list, bins=20)
    plt.show()
    return dice1_list, dice2_list, dice3_list, dice4_list, dice5_list


def ComputeDiceWNet(result_path, t2_folder, data_type, th=0.3):
    label_arr = np.load(os.path.join(result_path, '{}_label1.npy'.format(data_type)))
    preds_arr1 = np.load(os.path.join(result_path, '{}_preds1.npy'.format(data_type)))
    preds_arr2 = np.load(os.path.join(result_path, '{}_preds2.npy'.format(data_type)))

    dice1_list, dice2_list, dice3_list, dice4_list = [], [], [], []
    case_name = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/{}_name.csv'.format(data_type))
    case_list = sorted(case_name.loc[0].tolist())
    for index in range(preds_arr1.shape[0]):
        # t2 = np.squeeze(np.load(os.path.join(t2_folder, case_list[index] + '.npy')))
        # t2_crop, _ = ExtractPatch(t2, (200, 200))
        pred1 = ROIOneHot(np.argmax(preds_arr1[index], axis=0))
        pred2 = ROIOneHot(np.argmax(preds_arr2[index], axis=0))
        label = label_arr[index]

        dice1_list.append(Dice((preds_arr1[index] > th).astype(int), label_arr1[index]))
        dice2_list.append(Dice((preds_arr2[index] > th).astype(int), label_arr2[index]))
        dice3_list.append(Dice((preds_arr3[index] > th).astype(int), label_arr3[index]))
        dice4_list.append(Dice((preds_arr4[index] > th).astype(int), label_arr4[index]))

    return dice1_list, dice2_list, dice3_list, dice4_list


def ModelCompare(model_path1, model_path2, t2_folder, data_type):
    case_name = pd.read_csv(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/{}_name.csv'.format(data_type))
    case_list = sorted(case_name.loc[0].tolist())

    preds_arr_1 = np.load(os.path.join(model_path1, '{}_preds.npy'.format(data_type)))
    label_arr_1 = np.load(os.path.join(model_path1, '{}_label.npy'.format(data_type)))

    preds_arr_2 = np.load(os.path.join(model_path2, '{}_preds.npy'.format(data_type)))
    label_arr_2 = np.load(os.path.join(model_path2, '{}_label.npy'.format(data_type)))

    for index, case in enumerate(case_list):
        t2 = np.squeeze(np.load(os.path.join(t2_folder, case_list[index] + '.npy')))
        t2_crop, _ = ExtractPatch(t2, (200, 200))
        plt.figure(figsize=(16, 8))

        pred_1 = preds_arr_1[index]
        label_1 = label_arr_1[index]
        pred_2 = preds_arr_2[index]
        label_2 = label_arr_2[index]

        plt.subplot(251)
        plt.axis('off')
        plt.imshow(pred_1[0], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_1[0], colors='r')
        plt.subplot(252)
        plt.axis('off')
        plt.imshow(pred_1[1], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_1[1], colors='r')
        plt.subplot(256)
        plt.axis('off')
        plt.imshow(pred_1[2], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_1[2], colors='r')
        plt.subplot(257)
        plt.axis('off')
        plt.imshow(pred_1[3], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_1[3], colors='r')

        plt.subplot(253)
        plt.axis('off')
        plt.imshow(pred_2[0], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_2[0], colors='r')
        plt.subplot(254)
        plt.axis('off')
        plt.imshow(pred_2[1], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_2[1], colors='r')
        plt.subplot(258)
        plt.axis('off')
        plt.imshow(pred_2[2], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_2[2], colors='r')
        plt.subplot(259)
        plt.axis('off')
        plt.imshow(pred_2[3], cmap='gray', vmin=0., vmax=1.)
        plt.contour(label_2[3], colors='r')

        plt.subplot(2, 5, 10)
        plt.axis('off')
        plt.imshow(t2_crop, cmap='gray')
        plt.contour(label_1[0], colors='r')
        plt.contour(label_1[1], colors='y')
        plt.contour(label_1[2], colors='b')
        plt.contour(label_1[3], colors='g')
        plt.show()


def ShowBestWorseResult(dice1_list, dice2_list, dice3_list, dice4_list, preds_arr, label_arr):
    index1 = dice1_list.index(min(dice1_list))
    index2 = dice2_list.index(min(dice2_list))
    index3 = dice3_list.index(min(dice3_list))
    index4 = dice4_list.index(min(dice4_list))

    plt.suptitle('Worse')
    plt.subplot(221)
    plt.imshow(preds_arr[index1][0], cmap='gray')
    plt.contour(label_arr[index1][0], colors='r')
    plt.subplot(222)
    plt.imshow(preds_arr[index2][1], cmap='gray')
    plt.contour(label_arr[index2][1], colors='r')
    plt.subplot(223)
    plt.imshow(preds_arr[index3][2], cmap='gray')
    plt.contour(label_arr[index3][2], colors='r')
    plt.subplot(224)
    plt.imshow(preds_arr[index4][3], cmap='gray')
    plt.contour(label_arr[index4][3], colors='r')
    plt.show()

    index1 = dice1_list.index(max(dice1_list))
    index2 = dice2_list.index(max(dice2_list))
    index3 = dice3_list.index(max(dice3_list))
    index4 = dice4_list.index(max(dice4_list))

    plt.suptitle('Best')
    plt.subplot(221)
    plt.imshow(preds_arr[index1][0], cmap='gray')
    plt.contour(label_arr[index1][0], colors='r')
    plt.subplot(222)
    plt.imshow(preds_arr[index2][1], cmap='gray')
    plt.contour(label_arr[index2][1], colors='r')
    plt.subplot(223)
    plt.imshow(preds_arr[index3][2], cmap='gray')
    plt.contour(label_arr[index3][2], colors='r')
    plt.subplot(224)
    plt.imshow(preds_arr[index4][3], cmap='gray')
    plt.contour(label_arr[index4][3], colors='r')
    plt.show()


if __name__ == '__main__':
    from SegModel.ResNet50 import *
    from SegModel.WNet import *

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'


    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = WNet2_5D_weightshared(3, 3, 5).to(device)

    model_folder = os.path.join(model_root, 'WNet_WeightShare_1202_ce_mse')
    weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
    weights_list = [one for one in weights_list if str(one).endswith('.pt')]
    if len(weights_list) == 0:
        raise Exception
    weights_list = sorted(weights_list, key=lambda x: os.path.getctime(str(x)))
    weights_path = weights_list[-1]

    weights_path = os.path.join(model_folder, weights_path)

    ShoweResult(model_folder, data_type='test', num_pred=1, save_path=os.path.join(model_folder, 'Image'))

    # ShowAtten(model_path, data_type='train')
    # ShowAtten(model_path, data_type='val')
    # ShowAtten(model, model_path, epoch, data_type='test', save_path=r'')
    # ShowAtten(model, model_path, epoch, data_type='test', save_path=os.path.join(model_path, 'SpatialAttention'))


    # ************************************  2D Dice  *****************************************
    # train_1, train_2, train_3, train_4, train_5 = ComputeDice(result_path1, t2_folder, 'train')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(train_1) / len(train_1), sum(train_2) / len(train_2),
    #                                                       sum(train_3) / len(train_3), sum(train_4) / len(train_4), sum(train_5) / len(train_5)))
    # val_1, val_2, val_3, val_4, val_5 = ComputeDice(result_path1, t2_folder, 'val')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(val_1) / len(val_1), sum(val_2) / len(val_2),
    #                                                       sum(val_3) / len(val_3), sum(val_4) / len(val_4), sum(val_5) / len(val_5)))
    # test_1, test_2, test_3, test_4, test_5 = ComputeDice(result_path1, t2_folder, 'test')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(test_1) / len(test_1), sum(test_2) / len(test_2),
    #                                                       sum(test_3) / len(test_3), sum(test_4) / len(test_4), sum(test_5) / len(test_5)))

    # train_1, train_2, train_3, train_4 = ComputeDiceWNet(result_path1, t2_folder, 'train')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(train_1) / len(train_1), sum(train_2) / len(train_2), sum(train_3) / len(train_3), sum(train_4) / len(train_4)))
    # val_1, val_2, val_3, val_4 = ComputeDiceWNet(result_path1, t2_folder, 'val')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(val_1) / len(val_1), sum(val_2) / len(val_2), sum(val_3) / len(val_3), sum(val_4) / len(val_4)))
    # test_1, test_2, test_3, test_4 = ComputeDiceWNet(result_path1, t2_folder, 'test')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(test_1) / len(test_1), sum(test_2) / len(test_2), sum(test_3) / len(test_3), sum(test_4) / len(test_4)))
    # ************************************  2D Dice  *****************************************
