import torch
import torch.nn as nn
import torch.nn.functional as F

from SegModel.Block import *


class Encoder(nn.Module):
    def __init__(self, in_channels, filters=32):
        super(Encoder, self).__init__()
        self.conv1 = DoubleConv(in_channels, filters)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(filters, filters*2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(filters*2, filters*4)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(filters*4, filters*8)
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        return [c4, c3, c2, c1]


class Decoder(nn.Module):
    def __init__(self, out_channels, filters=32):
        super(Decoder, self).__init__()

        self.up5 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv6 = DoubleConv(filters*8, filters*4)

        self.up6 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv7 = DoubleConv(filters*4, filters*2)

        self.up7 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv8 = DoubleConv(filters*2, filters)

        self.conv9 = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, encoding_list):
        c4 = encoding_list[0]
        c3 = encoding_list[1]
        c2 = encoding_list[2]
        c1 = encoding_list[3]

        up_5 = self.up5(c4)
        merge6 = torch.cat((up_5, c3), dim=1)
        c6 = self.conv6(merge6)
        up_6 = self.up6(c6)
        merge7 = torch.cat((up_6, c2), dim=1)
        c7 = self.conv7(merge7)
        up_7 = self.up7(c7)
        merge8 = torch.cat((up_7, c1), dim=1)
        c8 = self.conv8(merge8)
        c9 = self.conv9(c8)
        return torch.sigmoid(c9)


class AttenDecoder(nn.Module):
    def __init__(self, out_channels, filters=32):
        super(AttenDecoder, self).__init__()

        self.Up5 = nn.ConvTranspose2d(filters * 8, filters * 8, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=filters * 8, F_l=filters * 4, F_int=filters * 8)
        self.Conv5 = DoubleConv(filters * (8 + 4), filters * 4)

        self.Up6 = nn.ConvTranspose2d(filters * 4, filters * 4, kernel_size=2, stride=2)
        self.Att6 = AttentionBlock(F_g=filters * 4, F_l=filters * 2, F_int=filters * 4)
        self.Conv6 = DoubleConv(filters * (4 + 2), filters * 2)

        self.Up7 = nn.ConvTranspose2d(filters * 2, filters * 2, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=filters * 2, F_l=filters, F_int=filters * 2)
        self.Conv7 = DoubleConv(filters * (2 + 1), filters)

        self.Conv_1x1 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, encoding_list):
        conv_4 = encoding_list[0]
        conv_3 = encoding_list[1]
        conv_2 = encoding_list[2]
        conv_1 = encoding_list[3]

        up_5 = self.Up5(conv_4)
        atten_5 = self.Att5(g=up_5, x=conv_3)
        merge_5 = torch.cat((atten_5, up_5), dim=1)
        conv_5 = self.Conv5(merge_5)

        up_6 = self.Up6(conv_5)
        atten_6 = self.Att6(g=up_6, x=conv_2)
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        conv_6 = self.Conv6(merge_6)

        up_7 = self.Up7(conv_6)
        atten_7 = self.Att7(g=up_7, x=conv_1)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        conv_7 = self.Conv7(merge_7)

        out = self.Conv_1x1(conv_7)
        return (F.interpolate(atten_5, size=(200, 200), mode='bilinear', align_corners=True),
               F.interpolate(atten_6, size=(200, 200), mode='bilinear', align_corners=True),
               F.interpolate(atten_7, size=(200, 200), mode='bilinear', align_corners=True)),\
               torch.sigmoid(out)


class MultiSeg(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(MultiSeg, self).__init__()

        self.encoding = Encoder(in_channels, filters)


        self.decoding1 = Decoder(out_channels, filters)
        self.decoding2 = Decoder(out_channels, filters)
        self.decoding3 = Decoder(out_channels, filters)
        self.decoding4 = Decoder(out_channels, filters)

    def forward(self, x):
        encoding_list = self.encoding(x)

        out1 = self.decoding1(encoding_list)
        out2 = self.decoding2(encoding_list)
        out3 = self.decoding3(encoding_list)
        out4 = self.decoding4(encoding_list)

        return torch.cat([out1, out2, out3, out4], dim=1)


class MultiAttenSeg(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(MultiAttenSeg, self).__init__()

        self.encoding = Encoder(in_channels, filters)

        self.decoding1 = AttenDecoder(out_channels, filters)
        self.decoding2 = AttenDecoder(out_channels, filters)
        self.decoding3 = AttenDecoder(out_channels, filters)
        self.decoding4 = AttenDecoder(out_channels, filters)


    def forward(self, x):
        encoding_list = self.encoding(x)

        atten_pz, out1 = self.decoding1(encoding_list)
        atten_cg, out2 = self.decoding2(encoding_list)
        atten_u, out3 = self.decoding3(encoding_list)
        atten_as, out4 = self.decoding4(encoding_list)

        return atten_pz, atten_cg, atten_u, atten_as, torch.cat([out1, out2, out3, out4], dim=1)
        # return torch.sigmoid(out1)


class MultiSegPlus(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32, is_atten=False):
        super(MultiSegPlus, self).__init__()

        self.encoding = Encoder(in_channels, filters)

        if is_atten:
            self.decoding1 = AttenDecoder(out_channels, filters)
            self.decoding2 = AttenDecoder(out_channels, filters)
            self.decoding3 = AttenDecoder(out_channels, filters)
            self.decoding4 = AttenDecoder(out_channels, filters)
        else:
            self.decoding1 = Decoder(1, filters)
            self.decoding2 = Decoder(1, filters)
            self.decoding3 = Decoder(1, filters)
            self.decoding4 = Decoder(1, filters)

        self.Conv = DoubleConv(4, filters)
        self.out = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        encoding_list = self.encoding(x)

        out1 = self.decoding1(encoding_list)
        out2 = self.decoding2(encoding_list)
        out3 = self.decoding3(encoding_list)
        out4 = self.decoding4(encoding_list)

        seg = torch.cat([out1, out2, out3, out4], dim=1)
        seg = self.Conv(seg)
        seg = self.out(seg)
        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), torch.sigmoid(out4), torch.sigmoid(seg)
        # return torch.sigmoid(out1)


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MultiSeg(in_channels=1, out_channels=4)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 1, 200, 200).to(device)
    prediction = model(inputs)
    print(prediction.shape)


if __name__ == '__main__':

    test()













