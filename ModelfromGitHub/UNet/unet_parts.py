""" Parts of the U-Net model """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels1+in_channels2, out_channels, in_channels1 // 2)
        # else:
        #     self.up = nn.ConvTranspose2d(in_channels1, in_channels1 // 2, kernel_size=2, stride=2)
        #     self.conv = DoubleConv(in_channels1//2+in_channels2, out_channels, in_channels1 // 2)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Encoding(nn.Module):
    def __init__(self, in_channels, filters=32, factor=2):
        super(Encoding, self).__init__()
        self.in_channels = in_channels

        self.inc = DoubleConv(in_channels, filters)

        self.down1 = Down(filters, filters * 2)
        self.down2 = Down(filters * 2, filters * 4)
        self.down3 = Down(filters * 4, filters * 8)
        self.down4 = Down(filters * 8, filters * 16 // factor)


    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x5, x4, x3, x2, x1]


class Decoding(nn.Module):
    def __init__(self, n_classes, filters, factor, bilinear=True):
        super(Decoding, self).__init__()
        if bilinear:
            self.up1 = Up(filters * 16 * 3, filters * 8 // factor, bilinear)
            self.up2 = Up(filters * 8 * 2, filters * 4 // factor, bilinear)
            self.up3 = Up(filters * 4 * 2, filters * 2 // factor, bilinear)
            self.up4 = Up(filters * 2 * 2, filters, bilinear)
        # else:
        #     self.up1 = Up(filters * (4+8) * 3, filters * 8 // factor, bilinear)
        #     self.up2 = Up(filters * 8 * 2, filters * 4 // factor, bilinear)
        #     self.up3 = Up(filters * 4 * 2, filters * 2 // factor, bilinear)
        #     self.up4 = Up(filters * 2 * 2, filters, bilinear)
        # self.up1 = Up(filters * 16// factor * 3, filters * 8 * 3, filters * 8 // factor, False)
        # self.up2 = Up(filters * 8 // factor,     filters * 4 * 3, filters * 4 // factor, bilinear)
        # self.up3 = Up(filters * 4 // factor,     filters * 2 * 3, filters * 2 // factor, bilinear)
        # self.up4 = Up(filters * 2 // factor,     filters * 1 * 3, filters, bilinear)
        self.outc = OutConv(filters, n_classes)

    def forward(self, encoding_result):
        # if isinstance(encoding_result[0], list):
        #     encoding_result = np.array(encoding_result)
        #     encoding_result = encoding_result.T
        # else:
        #     pass
        result1, result2, result3 = encoding_result[0], encoding_result[1], encoding_result[2]
        x5 = torch.cat([result1[0], result2[0], result3[0]], dim=1)
        x4 = torch.cat([result1[1], result2[1], result3[1]], dim=1)
        x3 = torch.cat([result1[2], result2[2], result3[2]], dim=1)
        x2 = torch.cat([result1[3], result2[3], result3[3]], dim=1)
        x1 = torch.cat([result1[4], result2[4], result3[4]], dim=1)



        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits