import torch
import torch.nn as nn
import torch.nn.functional as F

from SegModel.Block import *


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


class AttenUNet2_5D(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(AttenUNet2_5D, self).__init__()

        self.Pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.a_Conv1 = DoubleConv(in_channels, filters)
        self.a_Conv2 = DoubleConv(filters, filters * 2)
        self.a_Conv3 = DoubleConv(filters * 2, filters * 4)
        self.a_Conv4 = DoubleConv(filters * 4, filters * 8)

        self.b_Conv1 = DoubleConv(in_channels, filters)
        self.b_Conv2 = DoubleConv(filters, filters * 2)
        self.b_Conv3 = DoubleConv(filters * 2, filters * 4)
        self.b_Conv4 = DoubleConv(filters * 4, filters * 8)

        self.c_Conv1 = DoubleConv(in_channels, filters)
        self.c_Conv2 = DoubleConv(filters, filters * 2)
        self.c_Conv3 = DoubleConv(filters * 2, filters * 4)
        self.c_Conv4 = DoubleConv(filters * 4, filters * 8)

        self.Up5 = nn.ConvTranspose2d(filters*8*3, filters*8, kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=filters*8, F_l=filters*4*3, F_int=filters*8)   #channels of output is the channel of F_l
        self.Conv5 = DoubleConv(filters*(8+4*3), filters*4)

        self.Up6 = nn.ConvTranspose2d(filters*4, filters*4, kernel_size=2, stride=2)
        self.Att6 = AttentionBlock(F_g=filters*4, F_l=filters*2*3, F_int=filters*4)
        self.Conv6 = DoubleConv(filters*(4+2*3), filters*2)

        self.Up7 = nn.ConvTranspose2d(filters*2, filters*2, kernel_size=2, stride=2)
        self.Att7 = AttentionBlock(F_g=filters*2, F_l=filters*3, F_int=filters*2)
        self.Conv7 = DoubleConv(filters*(2+1*3), filters)

        self.Conv_1x1 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # encoding path
        x1 = x[:, 0:1, ...]
        x2 = x[:, 1:2, ...]
        x3 = x[:, 2:3, ...]

        a_conv_1 = self.a_Conv1(x1)
        a_pool_1 = self.Pool(a_conv_1)
        a_conv_2 = self.a_Conv2(a_pool_1)
        a_pool_2 = self.Pool(a_conv_2)
        a_conv_3 = self.a_Conv3(a_pool_2)
        a_pool_3 = self.Pool(a_conv_3)
        a_conv_4 = self.a_Conv4(a_pool_3)

        b_conv_1 = self.b_Conv1(x2)
        b_pool_1 = self.Pool(b_conv_1)
        b_conv_2 = self.b_Conv2(b_pool_1)
        b_pool_2 = self.Pool(b_conv_2)
        b_conv_3 = self.b_Conv3(b_pool_2)
        b_pool_3 = self.Pool(b_conv_3)
        b_conv_4 = self.b_Conv4(b_pool_3)

        c_conv_1 = self.c_Conv1(x3)
        c_pool_1 = self.Pool(c_conv_1)
        c_conv_2 = self.c_Conv2(c_pool_1)
        c_pool_2 = self.Pool(c_conv_2)
        c_conv_3 = self.c_Conv3(c_pool_2)
        c_pool_3 = self.Pool(c_conv_3)
        c_conv_4 = self.c_Conv4(c_pool_3)

        # encoding path
        conv_4 = torch.cat([a_conv_4, b_conv_4, c_conv_4], dim=1)    # 1, 768,  25,  25
        conv_3 = torch.cat([a_conv_3, b_conv_3, c_conv_3], dim=1)    # 1, 384,  50,  50
        conv_2 = torch.cat([a_conv_2, b_conv_2, c_conv_2], dim=1)    # 1, 192, 100, 100
        conv_1 = torch.cat([a_conv_1, b_conv_1, c_conv_1], dim=1)    # 1,  96, 200, 200

        up_5 = self.Up5(conv_4)                                      # 1, 256,  50,  50
        atten_5 = self.Att5(g=up_5, x=conv_3)                        # 1, 384,  50,  50
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

        return F.interpolate(atten_5, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_6, size=(200, 200), mode='bilinear', align_corners=True), \
               F.interpolate(atten_7, size=(200, 200), mode='bilinear', align_corners=True), \
               out


def test():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = AttenUNet2_5D(in_channels=1, out_channels=5)
    model = model.to(device)
    print(model)

    inputs = torch.randn(1, 3, 200, 200).to(device)
    _, _, _, prediction = model(inputs)
    print(prediction.shape)


if __name__ == '__main__':
    test()