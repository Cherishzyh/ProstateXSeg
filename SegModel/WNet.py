import torch
import torch.nn as nn

from SegModel.UNet import DoubleConv
from PreProcess.DistanceMapNumpy import DistanceMap


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


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = WNet(in_ch1=1, out_ch1=2, in_ch2=2, out_ch2=3)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 1, 200, 200).to(device)
    prediction1, prediction2 = model(inputs)
    print(prediction1.shape, prediction2.shape)


if __name__ == '__main__':
    test()