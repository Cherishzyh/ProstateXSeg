import torch
import torch.nn as nn
import torch.nn.functional as F

from SegModel.Block import *


class BottleneckWithCBAM(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckWithCBAM, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AttenUNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(AttenUNet, self).__init__()

        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_channels, filters)
        self.Conv2 = DoubleConv(filters, filters*2)
        self.Conv3 = DoubleConv(filters*2, filters*4)
        self.Conv4 = DoubleConv(filters*4, filters*8)

        self.Up5 = nn.ConvTranspose2d(filters*8, filters*8, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=filters*8, F_l=filters*4, F_int=filters*8)
        self.Conv5 = DoubleConv(filters*(8+4), filters*4)

        self.Up6 = nn.ConvTranspose2d(filters*4, filters*4, kernel_size=2, stride=2)
        self.Att6 = AttentionBlock(F_g=filters*4, F_l=filters*2, F_int=filters*4)
        self.Conv6 = DoubleConv(filters*(4+2), filters*2)

        self.Up7 = nn.ConvTranspose2d(filters*2, filters*2, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=filters*2, F_l=filters, F_int=filters*2)
        self.Conv7 = DoubleConv(filters*(2+1), filters)

        self.Conv_1x1 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        conv_1 = self.Conv1(x)
        # 32*shape
        pool_1 = self.Pool(conv_1)         # 32*shape/2
        conv_2 = self.Conv2(pool_1)         # 64*shape/2

        pool_2 = self.Pool(conv_2)         # 64*shape/4
        conv_3 = self.Conv3(pool_2)         # 128*shape/4

        pool_3 = self.Pool(conv_3)         # 128*shape/8
        conv_4 = self.Conv4(pool_3)         # 256*shape/8

        # encoding path
        up_5 = self.Up5(conv_4)             # 256*shape/4

        atten_5 = self.Att5(g=up_5, x=conv_3)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        conv_5 = self.Conv5(merge_5)        # 256*46*46=

        up_6 = self.Up6(conv_5)             # 256*shape/2
        atten_6 = self.Att6(g=up_6, x=conv_2)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        conv_6 = self.Conv6(merge_6)

        up_7 = self.Up7(conv_6)
        atten_7 = self.Att7(g=up_7, x=conv_1)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        conv_7 = self.Conv7(merge_7)

        out = self.Conv_1x1(conv_7)
        return F.interpolate(atten_5, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_6, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_7, size=(200, 200), mode='bilinear', align_corners=True), \
               torch.softmax(out, dim=1)


class UNetWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNetWithCBAM, self).__init__()

        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = DoubleConv(in_channels, filters)
        self.Conv2 = DoubleConv(filters, filters*2)
        self.Conv3 = DoubleConv(filters*2, filters*4)
        self.Conv4 = DoubleConv(filters*4, filters*8)

        self.Up5 = nn.ConvTranspose2d(filters*8, filters*8, kernel_size=2, stride=2)

        self.Conv5 = DoubleConv(filters*(8+4), filters*4)

        self.Up6 = nn.ConvTranspose2d(filters*4, filters*4, kernel_size=2, stride=2)

        self.Conv6 = DoubleConv(filters*(4+2), filters*2)

        self.Up7 = nn.ConvTranspose2d(filters*2, filters*2, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=filters*2, F_l=filters, F_int=filters*2)
        self.Conv7 = DoubleConv(filters*(2+1), filters)

        self.Conv_1x1 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        conv_1 = self.Conv1(x)
        # 32*shape
        pool_1 = self.Pool(conv_1)         # 32*shape/2
        conv_2 = self.Conv2(pool_1)         # 64*shape/2

        pool_2 = self.Pool(conv_2)         # 64*shape/4
        conv_3 = self.Conv3(pool_2)         # 128*shape/4

        pool_3 = self.Pool(conv_3)         # 128*shape/8
        conv_4 = self.Conv4(pool_3)         # 256*shape/8

        # encoding path
        up_5 = self.Up5(conv_4)             # 256*shape/4

        atten_5 = self.Att5(g=up_5, x=conv_3)        # 256*46*46 + 512*46*46 = 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)        # 512*46*46
        conv_5 = self.Conv5(merge_5)        # 256*46*46=

        up_6 = self.Up6(conv_5)             # 256*shape/2
        atten_6 = self.Att6(g=up_6, x=conv_2)        # 128*92*92 + 256*92*92
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        conv_6 = self.Conv6(merge_6)

        up_7 = self.Up7(conv_6)
        atten_7 = self.Att7(g=up_7, x=conv_1)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        conv_7 = self.Conv7(merge_7)

        out = self.Conv_1x1(conv_7)
        return F.interpolate(atten_5, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_6, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_7, size=(200, 200), mode='bilinear', align_corners=True), \
               torch.softmax(out, dim=1)


def test():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = AttenUNet(in_channels=1, out_channels=4)
    model = model.to(device)
    print(model)

    inputs = torch.randn(1, 1, 200, 200).to(device)
    prediction = model(inputs)
    print(prediction.shape)


if __name__ == '__main__':
    test()