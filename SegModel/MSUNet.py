import torch
import torch.nn as nn

from SegModel.UNet import DoubleConv

'''MSUNet = Multi Supervision UNet_Git'''
class MSUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(MSUNet, self).__init__()
        self.Conv1 = DoubleConv(in_channels, filters)
        self.pool1 = nn.MaxPool2d(2)
        self.Conv2 = DoubleConv(filters, filters * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.Conv3 = DoubleConv(filters * 2, filters * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.Conv4 = DoubleConv(filters * 4, filters * 8)

        self.up5 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.Conv6 = DoubleConv(filters * 8, filters * 4)
        self.Conv1x1_1 = nn.Conv2d(filters * 4, out_channels-2, 1)

        self.up6 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.Conv7 = DoubleConv(filters * 4, filters * 2)
        self.Conv1x1_2 = nn.Conv2d(filters * 2, out_channels-2, 1)

        self.up7 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.Conv8 = DoubleConv(filters * 2, filters)
        self.Conv1x1_3 = nn.Conv2d(filters, out_channels, 1)


    def forward(self, x):
        c1 = self.Conv1(x)
        p1 = self.pool1(c1)
        c2 = self.Conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.Conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.Conv4(p3)

        up_5 = self.up5(c4)
        merge6 = torch.cat((up_5, c3), dim=1)
        c5 = self.Conv6(merge6)
        out1 = self.Conv1x1_1(c5)

        up_6 = self.up6(c5)
        merge7 = torch.cat((up_6, c2), dim=1)
        c7 = self.Conv7(merge7)
        out2 = self.Conv1x1_2(c7)

        up_7 = self.up7(c7)
        merge8 = torch.cat((up_7, c1), dim=1)
        c8 = self.Conv8(merge8)
        out3 = self.Conv1x1_3(c8)

        return torch.softmax(out1, dim=1), torch.softmax(out2, dim=1), torch.softmax(out3, dim=1)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MSUNet(in_channels=1, out_channels=5)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 1, 200, 200).to(device)
    prediction1, prediction2, prediction3 = model(inputs)
    print(prediction1.shape, prediction2.shape, prediction3.shape)



if __name__ == '__main__':

    test()