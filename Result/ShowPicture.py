import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from SSHProject.CnnTools.T4T.Utility.Data import *

from SSHProject.BasicTool.MeDIT.Statistics import BinarySegmentation
from SSHProject.BasicTool.MeDIT.ArrayProcess import ExtractPatch

from Statistics.Metric import Dice
from PreProcess.Nii2NPY import ROIOneHot


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
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'

    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'

    model_path1 = os.path.join(model_root, 'TwoUNet_bce')
    result_path1 = os.path.join(model_path1, 'Result')

    train_1, train_2, train_3, train_4, train_5 = ComputeDice(result_path1, t2_folder, 'train')
    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(train_1) / len(train_1), sum(train_2) / len(train_2),
                                                          sum(train_3) / len(train_3), sum(train_4) / len(train_4), sum(train_5) / len(train_5)))
    val_1, val_2, val_3, val_4, val_5 = ComputeDice(result_path1, t2_folder, 'val')
    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(val_1) / len(val_1), sum(val_2) / len(val_2),
                                                          sum(val_3) / len(val_3), sum(val_4) / len(val_4), sum(val_5) / len(val_5)))
    test_1, test_2, test_3, test_4, test_5 = ComputeDice(result_path1, t2_folder, 'test')
    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(test_1) / len(test_1), sum(test_2) / len(test_2),
                                                          sum(test_3) / len(test_3), sum(test_4) / len(test_4), sum(test_5) / len(test_5)))

    # train_1, train_2, train_3, train_4 = ComputeDiceWNet(result_path1, t2_folder, 'train')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(train_1) / len(train_1), sum(train_2) / len(train_2), sum(train_3) / len(train_3), sum(train_4) / len(train_4)))
    # val_1, val_2, val_3, val_4 = ComputeDiceWNet(result_path1, t2_folder, 'val')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(val_1) / len(val_1), sum(val_2) / len(val_2), sum(val_3) / len(val_3), sum(val_4) / len(val_4)))
    # test_1, test_2, test_3, test_4 = ComputeDiceWNet(result_path1, t2_folder, 'test')
    # print('{:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(sum(test_1) / len(test_1), sum(test_2) / len(test_2), sum(test_3) / len(test_3), sum(test_4) / len(test_4)))

    # ModelCompare(result_path1, result_path2, t2_folder, 'test')
