import torch
import torch.nn as nn
from SegModel.Block import SpatialAttention


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Multi_UNet(nn.Module):
    '''
        inchannel1: for the whole gland segmentation
        outchannel1: for the whole gland segmentation

        inchannel2: for the multi-class segmentations
        outchannel2: for the multi-class segmentations
    '''
    def __init__(self, inchannels1, outchannels1, inchannels2, outchannels2, filters=32):
        super(Multi_UNet, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # encoding for the whole gland segmentation
        self.conv1_1 = DoubleConv(inchannels1, filters)
        self.conv1_2 = DoubleConv(filters, filters * 2)
        self.conv1_3 = DoubleConv(filters*2, filters*4)
        self.conv1_4 = DoubleConv(filters*4, filters*8)

        # encoding the multi-class segmentations
        self.conv2_1 = DoubleConv(inchannels2, filters)
        self.conv2_2 = DoubleConv(filters*2, filters * 2)
        self.conv2_3 = DoubleConv(filters*2*2, filters*4)
        self.conv2_4 = DoubleConv(filters*4*2, filters*8)

        self.spatialattention = SpatialAttention()

        # decoding for the whole gland segmentation
        self.up1_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1_5 = DoubleConv(filters * 8, filters * 4)
        self.up1_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv1_6 = DoubleConv(filters * 4, filters * 2)
        self.up1_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv1_7 = DoubleConv(filters * 2, filters)
        self.conv1_8 = nn.Conv2d(filters, outchannels1, 1)

        # decoding for multi-class segmentations
        self.up2_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv2_5 = DoubleConv(filters * 8, filters * 4)
        self.up2_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2_6 = DoubleConv(filters * 4, filters * 2)
        self.up2_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv2_7 = DoubleConv(filters * 2, filters)
        self.conv2_8 = nn.Conv2d(filters, outchannels2, 1)

    def forward(self, x, epoch):
        if epoch <= 10:
            x1_1 = self.conv1_1(x)                                 # torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)                             # torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)                          # torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)                             # torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)                          # torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)                             # torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)                          # torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)  # torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)                          # torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)  # torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)                          # torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)  # torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))            # torch.Size([1, 2, 184, 184])

            return out1
        else:
            x1_1 = self.conv1_1(x)                                      #torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)                                  #torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)                               #torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)                                  #torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)                               #torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)                                  #torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)                               #torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)       #torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)                               #torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)       #torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)                               #torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)       #torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))                 #torch.Size([1, 2, 184, 184])

            ##############################################################################################
            x2_1 = self.conv2_1(x)                                      #torch.Size([1, 32, 184, 184])
            x2_pool1 = self.pool(x2_1)                                  #torch.Size([1, 32, 92, 92])
            x2_2 = torch.cat([x2_pool1, x1_pool1], dim=1)
            x2_2 = self.conv2_2(x2_2)                                   #torch.Size([1, 64, 92, 92])
            x2_pool2 = self.pool(x2_2)                                  #torch.Size([1, 64, 46, 46])
            x2_3 = torch.cat([x2_pool2, x1_pool2], dim=1)               #torch.Size([1, 128, 46, 46])
            x2_3 = self.conv2_3(x2_3)                                   #torch.Size([1, 128, 46, 46])

            x2_pool3 = self.pool(x2_3)                                  #torch.Size([1, 128, 23, 23])
            x2_4 = torch.cat([x2_pool3, x1_pool3], dim=1)               #torch.Size([1, 256, 23, 23])
            x2_4 = self.conv2_4(x2_4)                                   #torch.Size([1, 256, 23, 23])

            merge2_1 = torch.cat((self.up2_1(x2_4), x2_3), dim=1)       #torch.Size([1, 256, 46, 46])
            merge2_1 = self.spatialattention(merge1_1) * merge2_1
            x2_5 = self.conv2_5(merge2_1)                               #torch.Size([1, 128, 46, 46])
            merge2_2 = torch.cat((self.up2_2(x2_5), x2_2), dim=1)
            merge2_2 = self.spatialattention(merge1_2) * merge2_2
            x2_6 = self.conv2_6(merge2_2)
            merge2_3 = torch.cat((self.up2_3(x2_6), x2_1), dim=1)
            merge2_3 = self.spatialattention(merge1_3) * merge2_3
            out2 = self.conv2_8(self.conv2_7(merge2_3))

            return out1, out2, self.spatialattention(merge1_1), self.spatialattention(merge1_2), self.spatialattention(merge1_3)


class Multi_UNet1(nn.Module):
    '''
        inchannel1: for the whole gland segmentation
        outchannel1: for the whole gland segmentation

        inchannel2: for the multi-class segmentations
        outchannel2: for the multi-class segmentations
    '''
    def __init__(self, inchannels1, outchannels1, inchannels2, outchannels2, filters=32):
        super(Multi_UNet1, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # encoding for the whole gland segmentation
        self.conv1_1 = DoubleConv(inchannels1, filters)
        self.conv1_2 = DoubleConv(filters, filters * 2)
        self.conv1_3 = DoubleConv(filters*2, filters*4)
        self.conv1_4 = DoubleConv(filters*4, filters*8)

        # encoding the multi-class segmentations
        self.conv2_1 = DoubleConv(inchannels2, filters)
        self.conv2_2 = DoubleConv(filters, filters * 2)
        self.conv2_3 = DoubleConv(filters*2, filters*4)
        self.conv2_4 = DoubleConv(filters*4, filters*8)

        self.spatialattention = SpatialAttention()

        # decoding for the whole gland segmentation
        self.up1_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1_5 = DoubleConv(filters * 8, filters * 4)
        self.up1_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv1_6 = DoubleConv(filters * 4, filters * 2)
        self.up1_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv1_7 = DoubleConv(filters * 2, filters)
        self.conv1_8 = nn.Conv2d(filters, outchannels1, 1)

        # decoding for multi-class segmentations
        self.up2_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv2_5 = DoubleConv(filters * 8, filters * 4)
        self.up2_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2_6 = DoubleConv(filters * 4, filters * 2)
        self.up2_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv2_7 = DoubleConv(filters * 2, filters)
        self.conv2_8 = nn.Conv2d(filters, outchannels2, 1)

    def forward(self, x, epoch):
        if epoch <= 10:
            x1_1 = self.conv1_1(x)                                 # torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)                             # torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)                          # torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)                             # torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)                          # torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)                             # torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)                          # torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)  # torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)                          # torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)  # torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)                          # torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)  # torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))            # torch.Size([1, 2, 184, 184])

            return out1
        else:
            x1_1 = self.conv1_1(x)                                      #torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)                                  #torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)                               #torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)                                  #torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)                               #torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)                                  #torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)                               #torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)       #torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)                               #torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)       #torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)                               #torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)       #torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))                 #torch.Size([1, 2, 184, 184])

            ##############################################################################################
            x2_1 = self.conv2_1(x)                                      #torch.Size([1, 32, 184, 184])
            x2_pool1 = self.pool(x2_1)                                  #torch.Size([1, 32, 92, 92])
            x2_2 = self.conv2_2(x2_pool1)                               #torch.Size([1, 64, 92, 92])
            x2_pool2 = self.pool(x2_2)                                  #torch.Size([1, 64, 46, 46])
            x2_3 = self.conv2_3(x2_pool2)                               #torch.Size([1, 128, 46, 46])

            x2_pool3 = self.pool(x2_3)                                  #torch.Size([1, 128, 23, 23])
            x2_4 = self.conv2_4(x2_pool3)                               #torch.Size([1, 256, 23, 23])

            merge2_1 = torch.cat((self.up2_1(x2_4), x2_3), dim=1)       #torch.Size([1, 256, 46, 46])
            merge2_1 = self.spatialattention(merge1_1) * merge2_1
            x2_5 = self.conv2_5(merge2_1)                               #torch.Size([1, 128, 46, 46])
            merge2_2 = torch.cat((self.up2_2(x2_5), x2_2), dim=1)
            merge2_2 = self.spatialattention(merge1_2) * merge2_2
            x2_6 = self.conv2_6(merge2_2)
            merge2_3 = torch.cat((self.up2_3(x2_6), x2_1), dim=1)
            merge2_3 = self.spatialattention(merge1_3) * merge2_3
            out2 = self.conv2_8(self.conv2_7(merge2_3))

            return out1, out2


class Multi_UNet2(nn.Module):
    '''
        inchannel1: for the whole gland segmentation
        outchannel1: for the whole gland segmentation

        inchannel2: for the multi-class segmentations
        outchannel2: for the multi-class segmentations
    '''
    def __init__(self, inchannels1, outchannels1, inchannels2, outchannels2, filters=32):
        super(Multi_UNet2, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # encoding for the whole gland segmentation
        self.conv1_1 = DoubleConv(inchannels1, filters)
        self.conv1_2 = DoubleConv(filters, filters * 2)
        self.conv1_3 = DoubleConv(filters*2, filters*4)
        self.conv1_4 = DoubleConv(filters*4, filters*8)

        # encoding the multi-class segmentations
        self.conv2_1 = DoubleConv(inchannels2, filters)
        self.conv2_2 = DoubleConv(filters*2, filters * 2)
        self.conv2_3 = DoubleConv(filters*2*2, filters*4)
        self.conv2_4 = DoubleConv(filters*4*2, filters*8)

        self.spatialattention = SpatialAttention()

        # decoding for the whole gland segmentation
        self.up1_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1_5 = DoubleConv(filters * 8, filters * 4)
        self.up1_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv1_6 = DoubleConv(filters * 4, filters * 2)
        self.up1_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv1_7 = DoubleConv(filters * 2, filters)
        self.conv1_8 = nn.Conv2d(filters, outchannels1, 1)

        # decoding for multi-class segmentations
        self.up2_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv2_5 = DoubleConv(filters * 8, filters * 4)
        self.up2_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2_6 = DoubleConv(filters * 4, filters * 2)
        self.up2_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv2_7 = DoubleConv(filters * 2, filters)
        self.conv2_8 = nn.Conv2d(filters, outchannels2, 1)

    def forward(self, x, epoch):
        if epoch <= 10:
            x1_1 = self.conv1_1(x)                                 # torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)                             # torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)                          # torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)                             # torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)                          # torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)                             # torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)                          # torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)  # torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)                          # torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)  # torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)                          # torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)  # torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))            # torch.Size([1, 2, 184, 184])

            return out1
        else:
            x1_1 = self.conv1_1(x)  # torch.Size([1, 32, 184, 184])
            x1_pool1 = self.pool(x1_1)  # torch.Size([1, 32, 92, 92])
            x1_2 = self.conv1_2(x1_pool1)  # torch.Size([1, 64, 92, 92])
            x1_pool2 = self.pool(x1_2)  # torch.Size([1, 64, 46, 46])
            x1_3 = self.conv1_3(x1_pool2)  # torch.Size([1, 128, 46, 46])
            x1_pool3 = self.pool(x1_3)  # torch.Size([1, 128, 23, 23])
            x1_4 = self.conv1_4(x1_pool3)  # torch.Size([1, 256, 23, 23])

            merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)  # torch.Size([1, 256, 46, 46])
            x1_5 = self.conv1_5(merge1_1)  # torch.Size([1, 128, 46, 46])
            merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)  # torch.Size([1, 128, 92, 92])
            x1_6 = self.conv1_6(merge1_2)  # torch.Size([1, 64, 92, 92])
            merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)  # torch.Size([1, 64, 184, 184])
            out1 = self.conv1_8(self.conv1_7(merge1_3))  # torch.Size([1, 2, 184, 184])

            ##############################################################################################
            x2_1 = self.conv2_1(x)  # torch.Size([1, 32, 184, 184])
            x2_pool1 = self.pool(x2_1)  # torch.Size([1, 32, 92, 92])
            x2_2 = torch.cat([x2_pool1, x1_pool1], dim=1)
            x2_2 = self.conv2_2(x2_2)  # torch.Size([1, 64, 92, 92])
            x2_pool2 = self.pool(x2_2)  # torch.Size([1, 64, 46, 46])
            x2_3 = torch.cat([x2_pool2, x1_pool2], dim=1)  # torch.Size([1, 128, 46, 46])
            x2_3 = self.conv2_3(x2_3)  # torch.Size([1, 128, 46, 46])
            x2_pool3 = self.pool(x2_3)  # torch.Size([1, 128, 23, 23])
            x2_4 = torch.cat([x2_pool3, x1_pool3], dim=1)  # torch.Size([1, 256, 23, 23])
            x2_4 = self.conv2_4(x2_4)  # torch.Size([1, 256, 23, 23])

            merge2_1 = torch.cat((self.up2_1(x2_4), x2_3), dim=1)  # torch.Size([1, 256, 46, 46])
            x2_5 = self.conv2_5(merge2_1)  # torch.Size([1, 128, 46, 46])
            merge2_2 = torch.cat((self.up2_2(x2_5), x2_2), dim=1)
            x2_6 = self.conv2_6(merge2_2)
            merge2_3 = torch.cat((self.up2_3(x2_6), x2_1), dim=1)
            out2 = self.conv2_8(self.conv2_7(merge2_3))

            return out1, out2


class Multi_UNet3(nn.Module):
    '''
        inchannel1: for the whole gland segmentation
        outchannel1: for the whole gland segmentation

        inchannel2: for the multi-class segmentations
        outchannel2: for the multi-class segmentations
    '''
    def __init__(self, inchannels1, outchannels1, inchannels2, outchannels2, filters=32):
        super(Multi_UNet3, self).__init__()
        self.pool = nn.MaxPool2d(2)

        # encoding for the whole gland segmentation
        self.conv1_1 = DoubleConv(inchannels1, filters)
        self.conv1_2 = DoubleConv(filters*2, filters * 2)
        self.conv1_3 = DoubleConv(filters*2*2, filters*4)
        self.conv1_4 = DoubleConv(filters*4*2, filters*8)

        # encoding the multi-class segmentations
        self.conv2_1 = DoubleConv(inchannels2, filters)
        self.conv2_2 = DoubleConv(filters, filters * 2)
        self.conv2_3 = DoubleConv(filters*2, filters*4)
        self.conv2_4 = DoubleConv(filters*4, filters*8)

        self.spatialattention = SpatialAttention()

        # decoding for the whole gland segmentation
        self.up1_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1_5 = DoubleConv(filters * 8, filters * 4)
        self.up1_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv1_6 = DoubleConv(filters * 4, filters * 2)
        self.up1_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv1_7 = DoubleConv(filters * 2, filters)
        self.conv1_8 = nn.Conv2d(filters, outchannels1, 1)

        # decoding for multi-class segmentations
        self.up2_1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv2_5 = DoubleConv(filters * 8, filters * 4)
        self.up2_2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2_6 = DoubleConv(filters * 4, filters * 2)
        self.up2_3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv2_7 = DoubleConv(filters * 2, filters)
        self.conv2_8 = nn.Conv2d(filters, outchannels2, 1)

    def forward(self, x):
        ####################################   encoding  ##########################################################
        x2_1 = self.conv2_1(x)                                      #torch.Size([1, 32, 184, 184])
        x2_pool1 = self.pool(x2_1)                                  #torch.Size([1, 32, 92, 92])
        x2_2 = self.conv2_2(x2_pool1)                                   #torch.Size([1, 64, 92, 92])
        x2_pool2 = self.pool(x2_2)                                  #torch.Size([1, 64, 46, 46])
        x2_3 = self.conv2_3(x2_pool2)                                   #torch.Size([1, 128, 46, 46])
        x2_pool3 = self.pool(x2_3)                                  #torch.Size([1, 128, 23, 23])
        x2_4 = self.conv2_4(x2_pool3)                                   #torch.Size([1, 256, 23, 23])

        x1_1 = self.conv1_1(x)                                      #torch.Size([1, 32, 184, 184])
        x1_pool1 = self.pool(x1_1)                                  #torch.Size([1, 32, 92, 92])
        x1_2 = torch.cat([x2_pool1, x1_pool1], dim=1)
        x1_2 = self.conv1_2(x1_2)                                   #torch.Size([1, 64, 92, 92])
        x1_pool2 = self.pool(x1_2)                                  #torch.Size([1, 64, 46, 46])
        x1_3 = torch.cat([x2_pool2, x1_pool2], dim=1)               #torch.Size([1, 128, 46, 46])
        x1_3 = self.conv1_3(x1_3)                                   #torch.Size([1, 128, 46, 46])
        x1_pool3 = self.pool(x1_3)                                  #torch.Size([1, 128, 23, 23])
        x1_4 = torch.cat([x2_pool3, x1_pool3], dim=1)               #torch.Size([1, 256, 23, 23])
        x1_4 = self.conv1_4(x1_4)                                   #torch.Size([1, 256, 23, 23])

        ####################################   decoding  ##########################################################

        merge1_1 = torch.cat((self.up1_1(x1_4), x1_3), dim=1)       #torch.Size([1, 256, 46, 46])
        x1_5 = self.conv1_5(merge1_1)                               #torch.Size([1, 128, 46, 46])
        merge1_2 = torch.cat((self.up1_2(x1_5), x1_2), dim=1)       #torch.Size([1, 128, 92, 92])
        x1_6 = self.conv1_6(merge1_2)                               #torch.Size([1, 64, 92, 92])
        merge1_3 = torch.cat((self.up1_3(x1_6), x1_1), dim=1)       #torch.Size([1, 64, 184, 184])
        out1 = self.conv1_8(self.conv1_7(merge1_3))                 #torch.Size([1, 2, 184, 184])

        merge2_1 = torch.cat((self.up2_1(x2_4), x2_3), dim=1)       #torch.Size([1, 256, 46, 46])
        merge2_1 = self.spatialattention(merge1_1) * merge2_1
        x2_5 = self.conv2_5(merge2_1)                               #torch.Size([1, 128, 46, 46])
        merge2_2 = torch.cat((self.up2_2(x2_5), x2_2), dim=1)
        merge2_2 = self.spatialattention(merge1_2) * merge2_2
        x2_6 = self.conv2_6(merge2_2)
        merge2_3 = torch.cat((self.up2_3(x2_6), x2_1), dim=1)
        merge2_3 = self.spatialattention(merge1_3) * merge2_3
        out2 = self.conv2_8(self.conv2_7(merge2_3))

        return out1, out2


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Multi_UNet2(1, 2, 1, 5)
    # model = UNet25D(in_channels=1, out_channels=5)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 1, 184, 184).to(device)
    prediction = model(inputs, 15)
    print(prediction.shape)


if __name__ == '__main__':

    test()