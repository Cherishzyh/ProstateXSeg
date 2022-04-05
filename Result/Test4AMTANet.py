import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from CnnTools.T4T.Utility.Data import *

from BasicTool.MeDIT.Statistics import BinarySegmentation
from BasicTool.MeDIT.ArrayProcess import ExtractPatch

from SegModel.UNet import UNet, UNet25D
from SegModel.AttenUnet import AttenUNet2_5D
from SegModel.MultiSeg import MultiSeg, MultiAttenSeg
from SegModel.WNet import WNet, WNet2_5D
from SegModel.MSUNet import MSUNet
from SegModel.TwoUNet import TwoUNet
# from ModelfromGitHub.UNet_Git.unet_model import UNet_Git, UNet25D
from SegModel.Atten import AttU_Net

from Statistics.Metric import Dice
from PreProcess.Nii2NPY import ROIOneHot
from PreProcess.DistanceMapNumpy import DistanceMap
from PreProcess.DistanceMapNumpy import KeepLargest


def ModelTest(model, model_folder, epoch, data_type='train'):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_shape = (200, 200)
    batch_size = 2

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}_name.csv'.format(data_type))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model.to(device)

    model.load_state_dict(torch.load(model_folder + epoch))

    pro_list, region_list, label_list, onelabel_list = [], [], [], []
    model.eval()
    with torch.no_grad():
        for inputs, outputs in data_loader:
            outputs_onelabel = torch.argmax(outputs, dim=1, keepdim=True)
            outputs_onelabel = torch.clip(outputs_onelabel, min=0, max=1)

            inputs = MoveTensorsToDevice(inputs, device)

            pro, region, _ = model(inputs)

            pro_list.extend(list(pro.cpu().data.numpy()))
            region_list.extend(list(region.cpu().data.numpy()))
            onelabel_list.extend(list(outputs_onelabel.numpy()))
            label_list.extend(list(outputs.numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_pro_preds.npy'.format(data_type)), np.array(pro_list))
    np.save(os.path.join(result_folder, '{}_pro_label.npy'.format(data_type)), np.array(onelabel_list))
    np.save(os.path.join(result_folder, '{}_reg_preds.npy'.format(data_type)), np.array(region_list))
    np.save(os.path.join(result_folder, '{}_reg_label.npy'.format(data_type)), np.array(label_list))


def ShoweResult(model_folder, data_type='train', save_path=r''):
    if save_path and not os.path.exists(save_path):
        os.mkdir(save_path)
    result_folder = os.path.join(model_folder, 'Result')
    pro_label_path = os.path.join(result_folder, '{}_pro_label.npy'.format(data_type))
    pro_preds_path = os.path.join(result_folder, '{}_pro_preds.npy'.format(data_type))
    reg_label_path = os.path.join(result_folder, '{}_reg_label.npy'.format(data_type))
    reg_preds_path = os.path.join(result_folder, '{}_reg_preds.npy'.format(data_type))

    pro_label = np.load(pro_label_path)
    pro_preds = np.load(pro_preds_path)

    reg_label = np.load(reg_label_path)
    reg_preds = np.load(reg_preds_path)

    for index in range(pro_label.shape[0]):
        reg_index = ROIOneHot(np.argmax(reg_preds[index], axis=0))
        pro_index = np.squeeze(pro_preds[index])
        reg_label_index = reg_label[index]
        pro_label_index = np.squeeze(pro_label[index])

        plt.figure(figsize=(12, 4))

        plt.subplot(251)
        plt.axis('off')
        plt.title('{:.3f}'.format(Dice(pro_index, pro_label_index)))
        plt.imshow(pro_index, cmap='gray')
        plt.contour(pro_label_index, colors='r')

        plt.subplot(254)
        plt.axis('off')
        plt.imshow(reg_label_index[3], cmap='gray')

        plt.subplot(255)
        plt.axis('off')
        plt.imshow(reg_label_index[4], cmap='gray')

        plt.subplot(256)
        plt.title('{:.3f}'.format(Dice(reg_index[0], reg_label_index[0])))
        plt.axis('off')
        plt.imshow(reg_index[0], cmap='gray')
        plt.contour(reg_label_index[0], colors='r')

        plt.subplot(257)
        plt.title('{:.3f}'.format(Dice(reg_index[1], reg_label_index[1])))
        plt.axis('off')
        plt.imshow(reg_index[1], cmap='gray')
        plt.contour(reg_label_index[1], colors='r')

        plt.subplot(258)
        plt.axis('off')
        plt.title('{:.3f}'.format(Dice(reg_index[2], reg_label_index[2])))
        plt.imshow(reg_index[2], cmap='gray')
        plt.contour(reg_label_index[2], colors='r')

        plt.subplot(259)
        plt.axis('off')
        plt.title('{:.3f}'.format(Dice(reg_index[3], reg_label_index[3])))
        plt.imshow(reg_index[3], cmap='gray')

        plt.subplot(2, 5, 10)
        plt.axis('off')
        plt.title('{:.3f}'.format(Dice(reg_index[4], reg_label_index[4])))
        plt.imshow(reg_index[4], cmap='gray')
        if save_path:
            plt.savefig(os.path.join(save_path, 'test_{}.jpg'.format(index)))
            plt.close()
        else:
            plt.show()



if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'

    from SegModel.AMTANet import *

    model_path = model_root + '/AMTANet_0623_25D'
    epoch = '/48-3.206361.pt'

    model = AMTA_Net(3, 1, 5)

    # ModelTest(model, model_path, epoch, 'train')
    # ModelTest(model, model_path, epoch, 'val')
    # ModelTest(model, model_path, epoch, 'test')
    #
    ShoweResult(model_path, data_type='test', save_path=os.path.join(model_path, 'Image'))
    # ShoweResult(model_path, data_type='test')

    # ShowAtten(model_path, data_type='train')
    # ShowAtten(model_path, data_type='val')
    # ShowAtten(model, model_path, epoch, data_type='test', save_path=r'')
    # ShowAtten(model, model_path, epoch, data_type='test', save_path=os.path.join(model_path, 'SpatialAttention'))









