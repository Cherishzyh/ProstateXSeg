import torch
import numpy as np
import torch.nn as nn
from SegModel.UNet import UNet

from PreProcess.DistanceMapNumpy import DistanceMap


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class TwoUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(TwoUNet, self).__init__()
        self.unet1 = UNet(in_channels, out_channels-2, filters=32)
        self.unet2 = UNet(in_channels, out_channels, filters=32)

    def forward(self, x, epoch):

        if epoch > 10:
            out1 = self.unet1(x)

            out1_copy = out1[:, 2, ...].cpu().data.numpy()
            dis_map_list = []
            for batch in range(out1_copy.shape[0]):
                dis_map_list.append(DistanceMap(out1_copy[batch, ...]))
            dis_map = np.array(dis_map_list)
            dis_map = dis_map[:, np.newaxis, ...]
            dis_map = torch.from_numpy(dis_map).to(device)
            assert dis_map.shape == x.shape

            out2 = self.unet2(dis_map*x)

            return out1, out2
        else:
            out1 = self.unet1(x)
            return out1