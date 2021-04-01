""" Full assembly of the parts to form the complete network """

''' 
https://github.com/milesial/Pytorch-UNet
'''

import torch.nn.functional as F

from ModelfromGitHub.UNet.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels=32, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, channels)
        self.down1 = Down(channels, channels * 2)
        self.down2 = Down(channels * 2, channels * 4)
        self.down3 = Down(channels * 4, channels * 8)
        self.down4 = Down(channels * 8, channels * 16 // factor)
        self.up1 = Up(channels * 16, channels * 8 // factor, bilinear)
        self.up2 = Up(channels * 8, channels * 4 // factor, bilinear)
        self.up3 = Up(channels * 4, channels * 2 // factor, bilinear)
        self.up4 = Up(channels * 2, channels, bilinear)
        self.outc = OutConv(channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.softmax(logits, dim=1)


class UNet25D(nn.Module):
    def __init__(self, n_channels, n_classes, filters=32, bilinear=True, factor=2):
        super(UNet25D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoding1 = Encoding(n_channels, filters, factor=factor)
        self.encoding2 = Encoding(n_channels, filters, factor=factor)
        self.encoding3 = Encoding(n_channels, filters, factor=factor)

        self.decoding  = Decoding(n_classes, filters, factor=factor, bilinear=bilinear)


    def forward(self, x):
        x1 = self.encoding1(x[:, 0:1, ...])
        x2 = self.encoding1(x[:, 1:2, ...])
        x3 = self.encoding1(x[:, 2:, ...])

        logits = self.decoding([x1, x2, x3])

        return torch.softmax(logits, dim=1)




def test():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = UNet25D(n_channels=1, n_classes=5, bilinear=True, factor=2)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 3, 200, 200).to(device)
    prediction = model(inputs)
    print(prediction.shape)

if __name__ == '__main__':
    test()
