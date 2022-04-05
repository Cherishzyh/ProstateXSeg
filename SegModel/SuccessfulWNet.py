import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

    # def __init__(self, in_channels1, in_channels2, out_channels, bilinear=True):
    #     super().__init__()
    #     if bilinear:
    #         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    #         self.conv = DoubleConv(in_channels1, out_channels, in_channels1 // 2)
    #         # self.conv = DoubleConv(in_channels1, out_channels)
    #     else:
    #         self.up = nn.ConvTranspose2d(in_channels1, in_channels1 // 2, kernel_size=2, stride=2)
    #         self.conv = DoubleConv(in_channels2 + in_channels1 // 2, out_channels)

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
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
    def __init__(self, n_classes, filters, bilinear=True):
        super(Decoding, self).__init__()

        self.up1 = Up(512, 256//2, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(filters, n_classes)

    def forward(self, encoding_result):
        x = self.up1(encoding_result[0], encoding_result[1]) # 128
        x = self.up2(x, encoding_result[2])
        x = self.up3(x, encoding_result[3])
        x = self.up4(x, encoding_result[4])
        logits = self.outc(x)
        return logits


class Decoding4SuccessfulModel(nn.Module):
    def __init__(self, n_classes, filters, bilinear=True):
        super(Decoding4SuccessfulModel, self).__init__()

        self.up1 = Up(512, 256//2, bilinear)
        self.up2 = Up(256, 64, bilinear)
        self.up3 = Up(128, 32, bilinear)
        self.up4 = Up(64, 32, bilinear)

        self.conv1 = nn.Conv2d(in_channels=256*3, out_channels=256, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=256*3, out_channels=256, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=128*3, out_channels=128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=64*3, out_channels=64, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=32*3, out_channels=32, kernel_size=1, bias=False)

        self.outc = OutConv(filters, n_classes)

    def forward(self, encoding_result):
        result1, result2, result3 = encoding_result[0], encoding_result[1], encoding_result[2]

        x5 = torch.cat([result1[0], result2[0], result3[0]], dim=1)
        x4 = torch.cat([result1[1], result2[1], result3[1]], dim=1)
        x3 = torch.cat([result1[2], result2[2], result3[2]], dim=1)
        x2 = torch.cat([result1[3], result2[3], result3[3]], dim=1)
        x1 = torch.cat([result1[4], result2[4], result3[4]], dim=1)

        x5 = self.conv1(x5)
        x4 = self.conv2(x4)
        x3 = self.conv3(x3)
        x2 = self.conv4(x2)
        x1 = self.conv5(x1)

        x = self.up1(x5, x4) # 128
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNet25D(nn.Module):
    def __init__(self, n_channels, n_classes, filters=32, bilinear=True, factor=2):
        super(UNet25D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoding1 = Encoding(n_channels, filters, factor=factor)
        self.encoding2 = Encoding(n_channels, filters, factor=factor)
        self.encoding3 = Encoding(n_channels, filters, factor=factor)

        self.decoding = Decoding4SuccessfulModel(n_classes, filters, bilinear=bilinear)


    def forward(self, x):
        x1 = self.encoding1(x[:, 0:1, ...])
        x2 = self.encoding1(x[:, 1:2, ...])
        x3 = self.encoding1(x[:, 2:, ...])

        logits = self.decoding([x1, x2, x3])

        return logits


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=32, bilinear=True, factor=2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.encoding = Encoding(n_channels, filters, factor=factor)
        self.decoding = Decoding(n_classes, filters, bilinear=bilinear)


    def forward(self, x):
        encoding_result = self.encoding(x)
        logits = self.decoding(encoding_result)
        return logits


class WNet2_5D(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2):
        super(WNet2_5D, self).__init__()
        self.UNet1 = UNet25D(in_ch1, out_ch1)
        self.UNet2 = UNet25D(in_ch2, out_ch2)

    def forward(self, x, epoch, is_multipy=False):
        input1 = input2 = x
        out1 = self.UNet1(input1)
        if (is_multipy and epoch > 5):
            out1_softmax = F.softmax(out1, dim=1)
            other = torch.argmax(out1_softmax, dim=1, keepdim=True)
            input2 = x * other
        out2 = self.UNet2(input2)
        return out1, out2


class WNet2_5D_channelcombine(nn.Module):
    def __init__(self, in_ch1, out_ch1, in_ch2, out_ch2):
        super(WNet2_5D_channelcombine, self).__init__()
        self.UNet1 = UNet(in_ch1, out_ch1)
        self.UNet2 = UNet(in_ch2, out_ch2)

    def forward(self, x, epoch, is_multipy=False):
        input1 = input2 = x
        out1 = self.UNet1(input1)
        if (is_multipy and epoch > 5):
            out1_softmax = F.softmax(out1, dim=1)
            other = torch.argmax(out1_softmax, dim=1, keepdim=True)
            input2 = x * other
        out2 = self.UNet2(input2)
        return out1, out2


class WNet2_5D_channelcombine_share(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch2):
        super(WNet2_5D_channelcombine_share, self).__init__()

        self.encoding = Encoding(in_ch)
        self.up1 = Up(512, 256 // 2)
        self.up2 = Up(256, 64)
        self.up3 = Up(128, 32)
        self.up4 = Up(64, 32)
        self.outc_1 = OutConv(32, out_ch1)
        self.outc_2 = OutConv(32, out_ch2)

    def forward(self, x):
        encoding_result = self.encoding(x)
        x = self.up1(encoding_result[0], encoding_result[1])  # 128
        x = self.up2(x, encoding_result[2])
        x = self.up3(x, encoding_result[3])
        x = self.up4(x, encoding_result[4])
        out1 = self.outc_1(x)
        out2 = self.outc_2(x)
        return out1, out2


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model =WNet2_5D_channelcombine(3, 3, 3, 5)
    print(model)
    inputs = torch.rand(size=(1, 3, 192, 192))
    prediction = model(inputs, epoch=10, is_multipy=True)
    print(prediction[0].shape, prediction[1].shape)
    # model = model.to(device)
    # model.load_state_dict(torch.load(r'/home/zhangyihong/Documents/ProstateX_Seg_ZYH/Model/WNet_0408/22-2.948752.pt'))

if __name__ == '__main__':
    test()