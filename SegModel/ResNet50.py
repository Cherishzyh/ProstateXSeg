import torch
import torch.nn as nn
import torch.nn.functional as F
from SegModel.Block import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
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


class ResNet(nn.Module):
    def __init__(self, block, layers, inchannels, outchannels, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(inchannels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(512 * block.expansion, num_classes),
                                 nn.Dropout(0.5),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(num_classes, outchannels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class ResUNet4Cls(nn.Module):
    def __init__(self, layers, inchannels, outchannels, block=BasicBlock, filters=32, zero_init_residual=False, norm_layer=None):
        super(ResUNet4Cls, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = filters

        self.conv1 = DoubleConv(inchannels, filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters, layers[0])
        self.layer2 = self._make_layer(block, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filters * 8, outchannels - 1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1, width_per_group=64):
        norm_layer = self._norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)        # x  = [batch, 1, 192, 192]
        x2 = self.maxpool(x1)     # x1 = [batch, 32, 192, 192]
        x2 = self.layer1(x2)      # x2 = [batch, 32, 96, 96]
        x3 = self.layer2(x2)      # x3 = [batch, 64, 48, 48]
        x4 = self.layer3(x3)      # x4 = [batch, 128, 24, 24]
        x5 = self.layer4(x4)      # x5 = [batch, 256, 12, 12]

        cls = self.avgpool(x5)
        cls = torch.flatten(cls, 1)
        cls = self.fc(cls)
        cls = self.sigmoid(cls)

        return cls


class ResUNet(nn.Module):
    def __init__(self, layers, inchannels, outchannels, block=BasicBlock, filters=32, zero_init_residual=False, norm_layer=None):
        super(ResUNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = filters

        self.conv = DoubleConv(inchannels, filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters, layers[0])
        self.layer2 = self._make_layer(block, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(filters*8, outchannels-1)
        self.sigmoid = nn.Sigmoid()

        self.up1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1 = DoubleConv(filters*8, filters*4)
        self.up2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2 = DoubleConv(filters*4, filters*2)
        self.up3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv3 = DoubleConv(filters*2, filters)
        self.up4 = nn.ConvTranspose2d(filters, filters, 2, stride=2)
        self.conv4 = DoubleConv(filters*2, 32)

        self.conv5 = nn.Conv2d(32, outchannels, 1)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1, width_per_group=64):
        norm_layer = self._norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv(x)        # x  = [batch, 1, 192, 192]
        x2 = self.maxpool(x1)     # x1 = [batch, 32, 192, 192]
        x2 = self.layer1(x2)      # x2 = [batch, 32, 96, 96]
        x3 = self.layer2(x2)      # x3 = [batch, 64, 48, 48]
        x4 = self.layer3(x3)      # x4 = [batch, 128, 24, 24]
        x5 = self.layer4(x4)      # x5 = [batch, 256, 12, 12]

        cls = self.avgpool(x5)
        cls = torch.flatten(cls, 1)
        cls = self.fc1(cls)
        cls = self.sigmoid(cls)

        up_1 = self.up1(x5)
        merge1 = torch.cat((up_1, x4), dim=1)
        x6 = self.conv1(merge1)

        up_2 = self.up2(x6)
        merge2 = torch.cat((up_2, x3), dim=1)
        x7 = self.conv2(merge2)

        up_3 = self.up3(x7)
        merge3 = torch.cat((up_3, x2), dim=1)
        x8 = self.conv3(merge3)

        up_4 = self.up4(x8)
        merge4 = torch.cat((up_4, x1), dim=1)
        sg = self.conv5(self.conv4(merge4))
        sg = self.softmax(sg)

        return cls, sg


class ResUNetCA(nn.Module):
    def __init__(self, layers, inchannels, outchannels, block=BasicBlock, filters=32, zero_init_residual=False, norm_layer=None):
        super(ResUNetCA, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = filters

        self.conv = DoubleConv(inchannels, filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters, layers[0])
        self.layer2 = self._make_layer(block, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters*8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(filters*8, outchannels-1)
        self.sigmoid = nn.Sigmoid()

        self.up1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1 = DoubleConv(filters*8, filters*4)
        self.up2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2 = DoubleConv(filters*4, filters*2)
        self.up3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv3 = DoubleConv(filters*2, filters)
        self.up4 = nn.ConvTranspose2d(filters, filters, 2, stride=2)
        self.conv4 = DoubleConv(filters*2, 32)

        self.conv5 = nn.Conv2d(32, outchannels, 1)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1, width_per_group=64):
        norm_layer = self._norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv(x)        # x  = [batch, 1, 192, 192]
        x2 = self.maxpool(x1)     # x1 = [batch, 32, 192, 192]
        x2 = self.layer1(x2)      # x2 = [batch, 32, 96, 96]
        x3 = self.layer2(x2)      # x3 = [batch, 64, 48, 48]
        x4 = self.layer3(x3)      # x4 = [batch, 128, 24, 24]
        x5 = self.layer4(x4)      # x5 = [batch, 256, 12, 12]

        cls = self.avgpool(x5)
        cls = torch.flatten(cls, 1)
        cls = self.fc1(cls)
        cls = self.sigmoid(cls)

        up_1 = self.up1(x5)
        merge1 = torch.cat((up_1, x4), dim=1)
        x6 = self.conv1(merge1)

        up_2 = self.up2(x6)
        merge2 = torch.cat((up_2, x3), dim=1)
        x7 = self.conv2(merge2)

        up_3 = self.up3(x7)
        merge3 = torch.cat((up_3, x2), dim=1)
        x8 = self.conv3(merge3)

        up_4 = self.up4(x8)
        merge4 = torch.cat((up_4, x1), dim=1)
        sg = self.conv5(self.conv4(merge4))

        weight = F.pad(cls, pad=(1, 0, 0, 0), mode='constant', value=1)
        weight = weight.unsqueeze(dim=2).unsqueeze(dim=2)
        weight = weight.expand_as(sg)
        sg = sg * weight
        sg = self.softmax(sg)

        return cls, sg


class UNet(nn.Module):
    def __init__(self, layers, inchannels, outchannels, block=BasicBlock, filters=32, zero_init_residual=False, norm_layer=None):
        super(UNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = filters

        self.conv = DoubleConv(inchannels, filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, filters, layers[0])
        self.layer2 = self._make_layer(block, filters*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, filters*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, filters*8, layers[3], stride=2)

        self.up1 = nn.ConvTranspose2d(filters*8, filters*4, 2, stride=2)
        self.conv1 = DoubleConv(filters*8, filters*4)
        self.up2 = nn.ConvTranspose2d(filters*4, filters*2, 2, stride=2)
        self.conv2 = DoubleConv(filters*4, filters*2)
        self.up3 = nn.ConvTranspose2d(filters*2, filters, 2, stride=2)
        self.conv3 = DoubleConv(filters*2, filters)
        self.up4 = nn.ConvTranspose2d(filters, filters, 2, stride=2)
        self.conv4 = DoubleConv(filters*2, 32)

        self.conv5 = nn.Conv2d(32, outchannels, 1)
        self.softmax = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, groups=1, width_per_group=64):
        norm_layer = self._norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.dilation = 1

        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv(x)        # x  = [batch, 1, 192, 192]
        x2 = self.maxpool(x1)     # x1 = [batch, 32, 192, 192]
        x2 = self.layer1(x2)      # x2 = [batch, 32, 96, 96]
        x3 = self.layer2(x2)      # x3 = [batch, 64, 48, 48]
        x4 = self.layer3(x3)      # x4 = [batch, 128, 24, 24]
        x5 = self.layer4(x4)      # x5 = [batch, 256, 12, 12]

        up_1 = self.up1(x5)
        merge1 = torch.cat((up_1, x4), dim=1)
        x6 = self.conv1(merge1)

        up_2 = self.up2(x6)
        merge2 = torch.cat((up_2, x3), dim=1)
        x7 = self.conv2(merge2)

        up_3 = self.up3(x7)
        merge3 = torch.cat((up_3, x2), dim=1)
        x8 = self.conv3(merge3)

        up_4 = self.up4(x8)
        merge4 = torch.cat((up_4, x1), dim=1)
        sg = self.conv5(self.conv4(merge4))
        sg = self.softmax(sg)

        return sg


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model = ResNet(BasicBlock, [3, 4, 6, 3], 5, 2).to(device)
    model = ResUNetCA([2, 3, 3, 3], 1, 5).to(device)
    inputs = torch.randn(3, 1, 192, 192).to(device)
    prediction, _ = model(inputs)
    print(prediction.shape)