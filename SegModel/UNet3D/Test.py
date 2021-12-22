import os
import shutil
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from MeDIT.Others import IterateCase
from T4T.Utility.Data import *

from Statistics.Metric import Dice4Torch


def Test(model_folder, device, model, data_type):
    input_shape = (192, 192)
    batch_size = 1

    spliter = DataSpliter()
    sub_list = spliter.LoadName(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH' + '/{}_name.csv'.format(data_type))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2', shape=input_shape))
    data.AddOne(Image2D(data_root + '/BG', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/PZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/CZ', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/DPU', shape=input_shape, is_roi=True), is_input=False)
    data.AddOne(Image2D(data_root + '/ASF', shape=input_shape, is_roi=True), is_input=False)

    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model.to(device)
    one_fold_weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
    one_fold_weights_list = sorted(one_fold_weights_list, key=lambda x: os.path.getctime(str(x)))
    weights_path = one_fold_weights_list[-1]
    print(weights_path.name, end='\t')
    model.load_state_dict(torch.load(str(weights_path)))

    pred_list, label_list = [], []
    input_list = []
    model.eval()
    with torch.no_grad():
        for inputs, outputs in loader:
            inputs = MoveTensorsToDevice(torch.unsqueeze(inputs, dim=1), device)

            preds = model(inputs)

            input_list.append(list(inputs.cpu().detach().numpy()))
            pred_list.append(list(preds.cpu().detach().numpy()))
            label_list.append(list(torch.stack(outputs, dim=1).numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_t2.npy'.format(data_type)), np.array(input_list))
    np.save(os.path.join(result_folder, '{}_preds.npy'.format(data_type)), np.array(pred_list))
    np.save(os.path.join(result_folder, '{}_label.npy'.format(data_type)), np.array(label_list))


def SaveFig():
    from MeDIT.Visualization import FlattenImages
    save_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Image'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    for data_type in ['train_case', 'val_case', 'test_case']:
        t2 = np.squeeze(np.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/{}_t2.npy'.format(data_type)))
        label = np.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/{}_label.npy'.format(data_type))
        preds = np.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/{}_preds.npy'.format(data_type))
        label = np.squeeze(np.argmax(label, axis=2))
        preds = np.squeeze(np.argmax(preds, axis=2))
        # pz cz dpu afs
        for case in range(t2.shape[0]):
            print(case)

            data_flatten = FlattenImages(t2[case])
            roi_flatten = FlattenImages(label[case])
            preds_flatten = FlattenImages(preds[case])
            plt.figure(figsize=(8, 8), dpi=300)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.subplot(221)
            plt.axis('off')
            plt.imshow(data_flatten, cmap='gray')
            # , linestyles='dashed'
            plt.contour((roi_flatten == 1).astype(int), colors='r', linewidths=0.5)
            plt.contour((preds_flatten == 1).astype(int), colors='g', linewidths=0.5)
            plt.subplot(222)
            plt.axis('off')
            plt.imshow(data_flatten, cmap='gray')
            plt.contour((roi_flatten == 2).astype(int), colors='r', linewidths=0.5)
            plt.contour((preds_flatten == 2).astype(int), colors='g', linewidths=0.5)
            plt.subplot(223)
            plt.axis('off')
            plt.imshow(data_flatten, cmap='gray')
            plt.contour((roi_flatten == 3).astype(int), colors='r', linewidths=0.5)
            plt.contour((preds_flatten == 3).astype(int), colors='g', linewidths=0.5)
            plt.subplot(224)
            plt.axis('off')
            plt.imshow(data_flatten, cmap='gray')
            plt.contour((roi_flatten == 4).astype(int), colors='r', linewidths=0.5)
            plt.contour((preds_flatten == 4).astype(int), colors='g', linewidths=0.5)
            plt.savefig(os.path.join(save_folder, '{}_{}.jpg'.format(data_type, str(case))), pad_inches=0)
            plt.close()


def ComputDice():
    from Statistics.Metric import BinarySegmentation
    store_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/AFS'
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)
    for data_type in ['test_case', 'val_case', 'train_case']:
        print(data_type)
        bs = BinarySegmentation(is_show=False)
        label = np.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/{}_label.npy'.format(data_type))
        preds = np.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D/UNet3D_1214/Result/{}_preds.npy'.format(data_type))
        label = np.argmax(np.squeeze(label), axis=1)
        preds = np.argmax(np.squeeze(preds), axis=1)
        HD_dict = {'PZ': [], 'CZ': [], 'DPU': [], 'AFS': []}
        Dice_dict = {'PZ': [], 'CZ': [], 'DPU': [], 'AFS': []}
        for case in range(label.shape[0]):
            for index, classes in enumerate(['BG', 'PZ', 'CZ', 'DPU', 'AFS']):
                if classes == 'BG': continue
                metric = bs.Run((preds[case] == index).astype(int), (label[case] == index).astype(int))
                Dice_dict[classes].append(metric['Dice'])
                if 'HD' in metric.keys():
                    HD_dict[classes].append(metric['HD'])
        print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(Dice_dict['PZ']) / len(Dice_dict['PZ']),
                                                         sum(Dice_dict['CZ']) / len(Dice_dict['CZ']),
                                                         sum(Dice_dict['DPU']) / len(Dice_dict['DPU']),
                                                         sum(Dice_dict['AFS']) / len(Dice_dict['AFS'])))
        print('{:.3f} / {:.3f} / {:.3f} / {:.3f}'.format(sum(HD_dict['PZ']) / len(HD_dict['PZ']),
                                                         sum(HD_dict['CZ']) / len(HD_dict['CZ']),
                                                         sum(HD_dict['DPU']) / len(HD_dict['DPU']),
                                                         sum(HD_dict['AFS']) / len(HD_dict['AFS'])))


if __name__ == '__main__':
    from SegModel.UNet3D.unet3d import UNet

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model3D'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Data/3D'

    model = UNet(1, 5, num_filters=16)

    # Test(os.path.join(model_root, 'UNet3D_1214'), device, model, data_type='train_case')
    # Test(os.path.join(model_root, 'UNet3D_1214'), device, model, data_type='val_case')
    # Test(os.path.join(model_root, 'UNet3D_1214'), device, model, data_type='test_case')
    # CheckInput()
    ComputDice()