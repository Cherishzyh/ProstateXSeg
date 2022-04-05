# 3D-UNet model.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, activation, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding),
        nn.BatchNorm3d(out_dim),
        activation, )


def conv_trans_block_3d(in_dim, out_dim, activation,
                        kernel_size=(3, 3, 3),
                        stride=(2, 2, 2),
                        padding=(1, 1, 1),
                        output_padding=(1, 1, 1)):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size,
                           stride=stride, padding=padding, output_padding=output_padding),
        nn.BatchNorm3d(out_dim),
        activation, )


def max_pooling_3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)):
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation, kernel_size=(3, 3, 3), padding=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding),
        nn.BatchNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=kernel_size, stride=(1, 1, 1), padding=padding),
        nn.BatchNorm3d(out_dim),
        activation)


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation,
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool_1 = max_pooling_3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation,
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool_2 = max_pooling_3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation,
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.pool_3 = max_pooling_3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation, stride=(1, 2, 2),
                                           kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1))
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation,
                                    kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation, stride=(1, 2, 2),
                                           kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1))
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation,
                                    kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation, stride=(1, 2, 2),
                                           kernel_size=(1, 3, 3), padding=(0, 1, 1), output_padding=(0, 1, 1))
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation,
                                    kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, activation, kernel_size=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)       # -> torch.Size([1, 4, 20, 192, 192])
        pool_1 = self.pool_1(down_1)  # -> torch.Size([1, 4, 20, 96, 96])

        down_2 = self.down_2(pool_1)  # -> torch.Size([1, 8, 20, 96, 96])
        pool_2 = self.pool_2(down_2)  # -> torch.Size([1, 8, 20, 48, 48])

        down_3 = self.down_3(pool_2)  # -> torch.Size([1, 16, 20, 48, 48])
        pool_3 = self.pool_3(down_3)  # -> torch.Size([1, 16, 20, 24, 24])

        down_4 = self.down_4(pool_3)  # -> torch.Size([1, 32, 20, 24, 24])
        pool_4 = self.pool_4(down_4)  # -> torch.Size([1, 32, 10, 12, 12])

        down_5 = self.down_5(pool_4)  # -> torch.Size([1, 64, 10, 12, 12])
        pool_5 = self.pool_5(down_5)  # -> torch.Size([1, 64, 10, 6, 6])

        # Bridge
        bridge = self.bridge(pool_5)  # -> torch.Size([1, 128, 5, 6, 6])

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> torch.Size([1, 128, 10, 12, 12])
        concat_1 = torch.cat([trans_1, down_5], dim=1)  # -> torch.Size([1, 192, 10, 12, 12])
        up_1 = self.up_1(concat_1)  # -> torch.Size([1, 64, 10, 12, 12])

        trans_2 = self.trans_2(up_1)  # -> torch.Size([1, 64, 20, 24, 24])
        concat_2 = torch.cat([trans_2, down_4], dim=1)  # -> torch.Size([1, 96, 20, 24, 24])
        up_2 = self.up_2(concat_2)  # -> torch.Size([1, 32, 20, 24, 24])

        trans_3 = self.trans_3(up_2)  # -> torch.Size([1, 32, 20, 48, 48])
        concat_3 = torch.cat([trans_3, down_3], dim=1)  # -> torch.Size([1, 48, 20, 48, 48])
        up_3 = self.up_3(concat_3)  # -> torch.Size([1, 16, 20, 48, 48])

        trans_4 = self.trans_4(up_3)  # -> torch.Size([1, 16, 20, 96, 96])
        concat_4 = torch.cat([trans_4, down_2], dim=1)  # -> torch.Size([1, 24, 20, 96, 96])
        up_4 = self.up_4(concat_4)  # -> torch.Size([1, 8, 20, 96, 96])

        trans_5 = self.trans_5(up_4)  # -> torch.Size([1, 8, 20, 192, 192])
        concat_5 = torch.cat([trans_5, down_1], dim=1)  # -> torch.Size([1, 12, 20, 192, 192])
        up_5 = self.up_5(concat_5)  # -> torch.Size([1, 4, 20, 192, 192])

        # Output
        out = self.out(up_5)  # -> torch.Size([1, 4, 20, 192, 192])
        return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 128
    x = torch.randn(size=[1, 1, 20, 192, 192])
    x.to(device)
    print("x size: {}".format(x.size()))

    model = UNet(in_dim=1, out_dim=4, num_filters=4)
    print(model)
    out = model(x)
    print("out size: {}".format(out.size()))