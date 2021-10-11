import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from T4T.Utility.Data import *

from MeDIT.Statistics import BinarySegmentation
from MeDIT.ArrayProcess import ExtractPatch

from SegModel.UNet import UNet, UNet25D
from SegModel.AttenUnet import AttenUNet2_5D
from SegModel.MultiSeg import MultiSeg, MultiAttenSeg
from SegModel.WNet import WNet, WNet2_5D
from SegModel.MSUNet import MSUNet
from SegModel.TwoUNet import TwoUNet
from SegModel.UNet_Git.unet_model import UNet25D
from SegModel.Atten import AttU_Net

from Statistics.Metric import Dice
from PreProcess.Nii2NPY import ROIOneHot
from PreProcess.DistanceMapNumpy import DistanceMap
from PreProcess.DistanceMapNumpy import KeepLargest


def ModelTest(model, model_folder, epoch, data_type='train'):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    input_shape = (100, 100)
    batch_size = 32

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
            if isinstance(preds2, tuple):
                preds2 = preds2[-1]
            # preds2 = torch.softmax(preds2, dim=1)
            preds2 = torch.squeeze(torch.sigmoid(preds2))

            pred_list.extend(list(preds2.cpu().detach().numpy()))
            # label_list.extend(list(outputs.cpu().data.numpy()))
            label_list.extend(list(outputs[:, -1].numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_preds.npy'.format(data_type)), np.array(pred_list))
    np.save(os.path.join(result_folder, '{}_label.npy'.format(data_type)), np.array(label_list))







if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'

    from SegModel.MultiTask import *

    model_path = model_root + '/UNet25D_0914'
    epoch = '/45-4.118648.pt'

    model = UNet25D(1, 1)

    ModelTest(model, model_path, epoch, 'train')
    ModelTest(model, model_path, epoch, 'val')
    ModelTest(model, model_path, epoch, 'test')









