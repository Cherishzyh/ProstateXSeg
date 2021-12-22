import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from T4T.Utility.Data import *

from MeDIT.Statistics import BinarySegmentation
from MeDIT.ArrayProcess import ExtractPatch
from MeDIT.Others import IterateCase

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


def ModelTest(model, model_folder, data_type='train'):

    input_shape = (192, 192)
    batch_size = 32

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}_name.csv'.format(data_type))

    data = DataManager(sub_list=sub_list)
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/RoiSlice', shape=input_shape, is_roi=True), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model.to(device)
    one_fold_weights_list = [one for one in IterateCase(model_folder, only_folder=False, verbose=0) if one.is_file()]
    one_fold_weights_list = [one for one in one_fold_weights_list if str(one).endswith('.pt')]
    one_fold_weights_list = sorted(one_fold_weights_list,  key=lambda x: os.path.getctime(str(x)))
    weights_path = one_fold_weights_list[-1]
    print(weights_path.name, end='\t')
    model.load_state_dict(torch.load(str(weights_path)))

    pred_list_1, pred_list_2, label_list = [], [], []
    input_list = []
    model.eval()
    with torch.no_grad():
        for inputs, outputs in data_loader:

            inputs = MoveTensorsToDevice(inputs, device)

            preds = model(inputs)

            pred_1, pred_2 = preds[0], preds[1]

            input_list.extend(list(inputs.cpu().detach().numpy()))
            pred_list_1.extend(list(pred_1.cpu().detach().numpy()))
            pred_list_2.extend(list(pred_2.cpu().detach().numpy()))
            label_list.extend(list(outputs.numpy()))

    result_folder = os.path.join(model_folder, 'Result')
    if not os.path.exists(os.path.join(model_folder, 'Result')):
        os.mkdir(os.path.join(model_folder, 'Result'))

    np.save(os.path.join(result_folder, '{}_t2.npy'.format(data_type)), np.array(pred_list_1))
    np.save(os.path.join(result_folder, '{}_pred_1.npy'.format(data_type)), np.array(pred_list_1))
    np.save(os.path.join(result_folder, '{}_pred_2.npy'.format(data_type)), np.array(pred_list_2))
    np.save(os.path.join(result_folder, '{}_label.npy'.format(data_type)), np.array(label_list))


if __name__ == '__main__':
    model_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model'
    # data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice'
    # t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/OneSlice/T2Slice'
    data_root = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice'
    t2_folder = r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/ThreeSlice/T2Slice'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    from SegModel.MultiTask import *
    from SegModel.WNet import *

    model_path = model_root + '/WNet_WeightShare_1202_ce_mse'

    model = WNet2_5D_weightshared(3, 3, 5)

    ModelTest(model, model_path, 'train')
    ModelTest(model, model_path, 'val')
    ModelTest(model, model_path, 'test')









