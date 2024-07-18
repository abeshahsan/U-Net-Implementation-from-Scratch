import torch

import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()

        self.double_conv_out = None

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        self.double_conv_out = x
        x = self.max_pool(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampleBlock, self).__init__()

        self.up_block = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_con):
        x = self.up_block(x)
        if x.shape != skip_con.shape:
            x = nn.functional.interpolate(x, size=skip_con.shape[2:],
                                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_con], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, out_channels):
        super(UNet, self).__init__()

        self.down_block1 = DownSampleBlock(3, 64)
        self.down_block2 = DownSampleBlock(64, 128)
        self.down_block3 = DownSampleBlock(128, 256)
        self.down_block4 = DownSampleBlock(256, 512)

        self.extra = DoubleConv(512, 1024)

        self.up_block1 = UpSampleBlock(1024, 512)
        self.up_block2 = UpSampleBlock(512, 256)
        self.up_block3 = UpSampleBlock(256, 128)
        self.up_block4 = UpSampleBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down_block1(x)
        skip_con1 = self.down_block1.double_conv_out
        x2 = self.down_block2(x1)
        skip_con2 = self.down_block2.double_conv_out
        x3 = self.down_block3(x2)
        skip_con3 = self.down_block3.double_conv_out
        x4 = self.down_block4(x3)
        skip_con4 = self.down_block4.double_conv_out

        x = self.extra(x4)

        x = self.up_block1(x, skip_con4)
        x = self.up_block2(x, skip_con3)
        x = self.up_block3(x, skip_con2)
        x = self.up_block4(x, skip_con1)

        x = self.out(x)
        return x


if __name__ == '__main__':
    # Example usage
    model = UNet(in_channels=3, out_channels=1)
    input_tensor = torch.randn(16, 3, 256, 256)
    output_tensor = model(input_tensor)
    print(output_tensor)
