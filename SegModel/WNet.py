import torch
import torch.nn as nn
import torch.nn.functional as F

from SegModel.UNet import DoubleConv
from SegModel.UNet_Git.unet_model import *
from SegModel.UNet_Git.unet_parts import *
from SegModel.Block import *


class WNet(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2, filters=32):
        super(WNet, self).__init__()
        self.Pool = nn.MaxPool2d(2)

        self.Conv1_1 = DoubleConv(in_ch1, filters)
        self.Conv1_2 = DoubleConv(filters, filters * 2)
        self.Conv1_3 = DoubleConv(filters * 2, filters * 4)
        self.Conv1_4 = DoubleConv(filters * 4, filters * 8)

        self.Up1_1 = nn.ConvTranspose2d(filters * 8, filters * 4, 2, stride=2)
        self.Conv1_5 = DoubleConv(filters * 8, filters * 4)
        self.Up1_2 = nn.ConvTranspose2d(filters * 4, filters * 2, 2, stride=2)
        self.Conv1_6 = DoubleConv(filters * 4, filters * 2)
        self.Up1_3 = nn.ConvTranspose2d(filters * 2, filters, 2, stride=2)
        self.Conv1_7 = DoubleConv(filters * 2, filters)

        self.Out1 = nn.Conv2d(filters, out_ch1, 1)

        self.Conv2_1 = DoubleConv(in_ch2, filters)
        self.Conv2_2 = DoubleConv(filters, filters * 2)
        self.Conv2_3 = DoubleConv(filters * 2, filters * 4)
        self.Conv2_4 = DoubleConv(filters * 4, filters * 8)

        self.Up2_1 = nn.ConvTranspose2d(filters * 8, filters * 4, 2, stride=2)
        self.Conv2_5 = DoubleConv(filters * 8, filters * 4)
        self.Up2_2 = nn.ConvTranspose2d(filters * 4, filters * 2, 2, stride=2)
        self.Conv2_6 = DoubleConv(filters * 4, filters * 2)
        self.Up2_3 = nn.ConvTranspose2d(filters * 2, filters, 2, stride=2)
        self.Conv2_7 = DoubleConv(filters * 2, filters)

        self.Out2 = nn.Conv2d(filters, out_ch2, 1)

    def forward(self, x):
        input1 = x
        x1_1 = self.Conv1_1(input1)
        x1_2 = self.Pool(x1_1)
        x1_2 = self.Conv1_2(x1_2)
        x1_3 = self.Pool(x1_2)
        x1_3 = self.Conv1_3(x1_3)
        x1_4 = self.Pool(x1_3)
        x1_4 = self.Conv1_4(x1_4)

        x1_5 = self.Up1_1(x1_4)
        x1_5 = torch.cat((x1_5, x1_3), dim=1)
        x1_5 = self.Conv1_5(x1_5)
        x1_6 = self.Up1_2(x1_5)
        x1_6 = torch.cat((x1_6, x1_2), dim=1)
        x1_6 = self.Conv1_6(x1_6)
        x1_7 = self.Up1_3(x1_6)
        x1_7 = torch.cat((x1_7, x1_1), dim=1)
        x1_7 = self.Conv1_7(x1_7)

        out1 = self.Out1(x1_7)

        input2 = out1*input1

        x2_1 = self.Conv2_1(input1)
        x2_2 = self.Pool(x2_1)
        x2_2 = self.Conv2_2(x2_2)

        x2_3 = self.Pool(x2_2)
        x2_3 = self.Conv2_3(x2_3)

        x2_4 = self.Pool(x2_3)
        x2_4 = self.Conv2_4(x2_4)

        x2_5 = self.Up2_1(x2_4)
        x2_5 = torch.cat((x2_5, x2_3), dim=1)
        x2_5 = self.Conv2_5(x2_5)

        x2_6 = self.Up2_2(x2_5)
        x2_6 = torch.cat((x2_6, x2_2), dim=1)
        x2_6 = self.Conv2_6(x2_6)

        x2_7 = self.Up2_3(x2_6)
        x2_7 = torch.cat((x2_7, x2_1), dim=1)
        x2_7 = self.Conv2_7(x2_7)

        out2 = self.Out2(x2_7)

        return torch.softmax(out1, dim=1), torch.softmax(out2, dim=1)


class WNet2_5D(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2, filters=32):
        super(WNet2_5D, self).__init__()
        self.UNet1 = UNet25D(in_ch1, out_ch1)
        self.UNet2 = UNet25D(in_ch2, out_ch2)

    def forward(self, x):
        # input1 = x
        input1 = input2 = x
        out1 = self.UNet1(input1)
        # out1_softmax = F.softmax(out1, dim=1)
        # other = torch.argmax(out1_softmax, dim=1, keepdim=True)
        # input2 = x * other
        out2 = self.UNet2(input2)
        return out1, out2


class WNet2_5D_channel(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2, filters=32):
        super(WNet2_5D_channel, self).__init__()
        self.UNet1 = UNet(in_ch1, out_ch1, channels=filters)
        self.UNet2 = UNet(in_ch2, out_ch2, channels=filters)

    def forward(self, x):
        input1 = x
        out1 = self.UNet1(input1)
        out1_softmax = F.softmax(out1, dim=1)
        other = torch.clamp(torch.argmax(out1_softmax, dim=1, keepdim=True), max=1, min=0)
        input2 = x * other
        out2 = self.UNet2(input2)
        return out1, out2


class WNet2_5D_weightshared(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2, filters=32):
        super(WNet2_5D_weightshared, self).__init__()
        self.UNet = UNet_NoOut(in_ch, channels=filters)
        self.Out1 = OutConv(filters, out_ch1)
        self.Out2 = OutConv(filters, out_ch2)

    def forward(self, x):
        input1 = input2 = x
        out1 = self.UNet(input1)
        out2 = self.UNet(input2)
        # out1_softmax = F.softmax(out1, dim=1)
        # other = torch.argmax(out1_softmax, dim=1, keepdim=True)
        out1 = self.Out1(out1)
        out2 = self.Out2(out2)
        return torch.softmax(out1, dim=1), torch.softmax(out2, dim=1)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = WNet2_5D(in_ch1=1, out_ch1=3, in_ch2=1, out_ch2=5)
    model = model.to(device)
    # parameters infomation of network
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    model.load_state_dict(torch.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/WNet_0408/22-2.948752.pt'))
    # print(model)
    # inputs = torch.randn(12, 3, 200, 200).to(device)
    # prediction1, prediction2 = model(inputs)
    # print(prediction1.shape, prediction2.shape)


if __name__ == '__main__':
    test()