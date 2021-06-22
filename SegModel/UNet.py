import torch
import torch.nn as nn


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


# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, filters=32):
#         super(UNet, self).__init__()
#
#         self.pool = nn.MaxPool2d(2)
#
#         self.conv1 = DoubleConv(in_channels, filters)
#         self.conv2 = DoubleConv(filters, filters*2)
#         self.conv3 = DoubleConv(filters*2, filters*4)
#         self.conv4 = DoubleConv(filters*4, filters*8)
#         self.conv5 = DoubleConv(filters*8, filters*16)
#
#         self.up5 = nn.ConvTranspose2d(filters*16, filters*8, 2, stride=2)
#         self.conv6 = DoubleConv(filters*16, filters*4)
#         self.up6 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
#         self.conv7 = DoubleConv(filters*8, filters*4)
#         self.up7 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
#         self.conv8 = DoubleConv(filters*4, filters*2)
#         self.up8 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
#         self.conv9 = DoubleConv(filters*2, filters)
#         self.conv10 = nn.Conv2d(filters, out_channels, 1)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#
#         x1 = self.pool(x1)
#         x2 = self.conv2(x1)
#
#         x2 = self.pool(x2)
#         x3 = self.conv3(x2)
#
#         x3 = self.pool(x3)
#         x4 = self.conv4(x3)
#
#         x4 = self.pool(x4)
#         x5 = self.conv5(x4)
#
#         up_5 = self.up5(x5)
#         merge1 = torch.cat((up_5, x4), dim=1)
#         x6 = self.conv6(merge1)
#
#         up_6 = self.up6(x6)
#         merge2 = torch.cat((up_6, x3), dim=1)
#         x5 = self.conv7(merge2)
#
#         up_7 = self.up7(x5)
#         merge3 = torch.cat((up_7, x2), dim=1)
#         x7 = self.conv8(merge3)
#
#         up_8 = self.up8(x7)
#         merge4 = torch.cat((up_8, x1), dim=1)
#         x8 = self.conv9(merge4)
#         x9 = self.conv10(x8)
#         return torch.softmax(x9, dim=1)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = DoubleConv(in_channels, filters)
        self.conv2 = DoubleConv(filters, filters*2)
        self.conv3 = DoubleConv(filters*2, filters*4)
        self.conv4 = DoubleConv(filters*4, filters*8)

        self.up6 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv7 = DoubleConv(filters*8, filters*4)
        self.up7 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv8 = DoubleConv(filters*4, filters*2)
        self.up8 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv9 = DoubleConv(filters*2, filters)
        self.conv10 = nn.Conv2d(filters, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)

        x4 = self.pool(x3)
        x4 = self.conv4(x4)

        up_6 = self.up6(x4)
        merge2 = torch.cat((up_6, x3), dim=1)
        x5 = self.conv7(merge2)

        up_7 = self.up7(x5)
        merge3 = torch.cat((up_7, x2), dim=1)
        x7 = self.conv8(merge3)

        up_8 = self.up8(x7)
        merge4 = torch.cat((up_8, x1), dim=1)
        x8 = self.conv9(merge4)
        x9 = self.conv10(x8)
        return x9


class UNet25D(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNet25D, self).__init__()
        self.a_conv1 = DoubleConv(in_channels, filters)
        self.a_pool1 = nn.MaxPool2d(2)
        self.a_conv2 = DoubleConv(filters, filters*2)
        self.a_pool2 = nn.MaxPool2d(2)
        self.a_conv3 = DoubleConv(filters*2, filters*4)
        self.a_pool3 = nn.MaxPool2d(2)
        self.a_conv4 = DoubleConv(filters*4, filters*8)

        self.b_conv1 = DoubleConv(in_channels, filters)
        self.b_pool1 = nn.MaxPool2d(2)
        self.b_conv2 = DoubleConv(filters, filters*2)
        self.b_pool2 = nn.MaxPool2d(2)
        self.b_conv3 = DoubleConv(filters*2, filters*4)
        self.b_pool3 = nn.MaxPool2d(2)
        self.b_conv4 = DoubleConv(filters*4, filters*8)

        self.c_conv1 = DoubleConv(in_channels, filters)
        self.c_pool1 = nn.MaxPool2d(2)
        self.c_conv2 = DoubleConv(filters, filters*2)
        self.c_pool2 = nn.MaxPool2d(2)
        self.c_conv3 = DoubleConv(filters*2, filters*4)
        self.c_pool3 = nn.MaxPool2d(2)
        self.c_conv4 = DoubleConv(filters*4, filters*8)


        self.up5 = nn.ConvTranspose2d(filters*8*3, filters*4, 2, stride=2)       #768
        self.conv6 = DoubleConv(filters*4*4, filters*4)
        self.up6 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv7 = DoubleConv(filters*2*4, filters*2)
        self.up7 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv8 = DoubleConv(filters*4, filters)
        self.conv9 = nn.Conv2d(filters, out_channels, 1)

    def forward(self, x):
        x1 = x[:, 0:1, ...]
        x2 = x[:, 1:2, ...]
        x3 = x[:, 2:3, ...]

        a_c1 = self.a_conv1(x1)
        a_p1 = self.a_pool1(a_c1)
        a_c2 = self.a_conv2(a_p1)
        a_p2 = self.a_pool2(a_c2)
        a_c3 = self.a_conv3(a_p2)
        a_p3 = self.a_pool3(a_c3)
        a_c4 = self.a_conv4(a_p3)

        b_c1 = self.b_conv1(x2)
        b_p1 = self.b_pool1(b_c1)
        b_c2 = self.b_conv2(b_p1)
        b_p2 = self.b_pool2(b_c2)
        b_c3 = self.b_conv3(b_p2)
        b_p3 = self.b_pool3(b_c3)
        b_c4 = self.b_conv4(b_p3)

        c_c1 = self.c_conv1(x3)
        c_p1 = self.c_pool1(c_c1)
        c_c2 = self.c_conv2(c_p1)
        c_p2 = self.c_pool2(c_c2)
        c_c3 = self.c_conv3(c_p2)
        c_p3 = self.c_pool3(c_c3)
        c_c4 = self.c_conv4(c_p3)

        c4 = torch.cat((a_c4, b_c4, c_c4), dim=1)
        up_5 = self.up5(c4)
        merge6 = torch.cat((up_5, a_c3, b_c3, c_c3), dim=1)
        c5 = self.conv6(merge6)
        up_6 = self.up6(c5)
        merge7 = torch.cat((up_6, a_c2, b_c2, c_c2), dim=1)
        c7 = self.conv7(merge7)
        up_7 = self.up7(c7)
        merge8 = torch.cat((up_7, a_c1, b_c1, c_c1), dim=1)
        c8 = self.conv8(merge8)
        c9 = self.conv9(c8)
        return c9


class UNetSimple(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNetSimple, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = DoubleConv(in_channels, filters)
        self.conv2 = DoubleConv(filters, filters*2)
        self.conv3 = DoubleConv(filters*2, filters*4)

        self.up4 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv4 = DoubleConv(filters*4, filters*2)
        self.up5 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv5 = DoubleConv(filters*2, filters)
        self.conv6 = nn.Conv2d(filters, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool(x1)
        x2 = self.conv2(x2)

        x3 = self.pool(x2)
        x3 = self.conv3(x3)

        up_4 = self.up4(x3)
        merge1 = torch.cat((up_4, x2), dim=1)
        x4 = self.conv4(merge1)

        up_5 = self.up5(x4)
        merge2 = torch.cat((up_5, x1), dim=1)
        x5 = self.conv5(merge2)
        x6 = self.conv6(x5)
        return x6


def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNetSimple(in_channels=3, out_channels=2)
    # model = UNet25D(in_channels=1, out_channels=5)
    model = model.to(device)
    print(model)
    inputs = torch.randn(1, 3, 184, 184).to(device)
    prediction = model(inputs)
    print(prediction.shape)


if __name__ == '__main__':

    test()