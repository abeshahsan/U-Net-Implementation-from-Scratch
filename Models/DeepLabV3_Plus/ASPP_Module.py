"""
This file contains the ASPP Module for the DeepLabV3+ model

The ASPP Module is the Atrous Spatial Pyramid Pooling Module
It is used to capture the context at multiple scales

The ASPP Module consists of the following components:

1. 1x1 Convolution
2. Atrous Separable Convolutions with different dilation rates
3. ASPP Pooling

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtrouSeparableConv2D(nn.Module):
    """
    Atrous Separable Convolution

    Args:
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_size: int
        Size of the kernel
    padding: int
        Padding to be applied
    dilation: int
        Dilation to be applied

    Returns:
    ----------
    output: torch.nn.Conv2d
        The Final output after applying the separable convolution

    """
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, dilation = 1, stride = 1):
        super(AtrouSeparableConv2D, self).__init__()
        self.m_separable_conv = nn.Conv2d(in_channels,
                                          in_channels, kernel_size,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=in_channels, stride=stride)
        self.m_pointwise_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.m_batch_norm = nn.BatchNorm2d(out_channels)
        self.m_relu = nn.ReLU()

    def forward(self, x):
        x = self.m_separable_conv(x)
        x = self.m_pointwise_conv(x)
        x = self.m_batch_norm(x)
        x = self.m_relu(x)

        return x


class ASPP_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        """
        ASPP Convolution

        Args:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        kernel_size: int
            Size of the kernel
        padding: int
            Padding to be applied
        dilation: int
            Dilation to be applied

        Returns:
        ----------
        output: torch.nn.Conv2d
            The Final output after applying the ASPP convolution

        """
        super(ASPP_Conv, self).__init__()
        self.m_conv = nn.Conv2d(in_channels,
                                out_channels, kernel_size,
                                padding=dilation, dilation=dilation)
        self.m_batch_norm = nn.BatchNorm2d(out_channels)
        self.m_relu = nn.ReLU()

    def forward(self, x):
        x = self.m_conv(x)
        x = self.m_batch_norm(x)
        x = self.m_relu(x)

        return x


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 0, dilation = 1):
        """
        ASPP Pooling

        Args:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        kernel_size: int
            Size of the kernel
        padding: int
            Padding to be applied
        dilation: int
            Dilation to be applied

        Returns:
        ----------
        output: torch.nn.Conv2d
            The Final output after applying the ASPP pooling

        """
        super(ASPPPooling, self).__init__()
        self.m_pool = nn.AvgPool2d(kernel_size=(2, 2))

        self.m_conv = nn.Conv2d(in_channels,
                                out_channels, kernel_size,
                                padding=padding,
                                dilation=dilation)
        self.m_batch_norm = nn.BatchNorm2d(out_channels)
        self.m_relu = nn.ReLU()
        self.m_interpolate = F.interpolate

    def forward(self, input):
        pooled = self.m_pool(input)
        conv = self.m_conv(pooled)
        normalized = self.m_batch_norm(conv)
        activated = self.m_relu(normalized)
        output = self.m_interpolate(activated,
                                    size=input.size()[2:], # H, W
                                    mode='bilinear',
                                    align_corners=True)

        return output

class ASPP(nn.Module):
    def __init__(self, in_channels, dilation_rates: list[int]):
        """
        ASPP Module

        Args:
        ----------
        in_channels: int
            Number of input channels
        dilation_rates: list[int]
            List of dilation rates to be applied
            Note: The 1x1 convolution is applied to the input first and then the dilated convolutions are applied

        Returns:
        ----------
        output: torch.nn.Conv2d
            The Final output after applying the ASPP module
        """
        super(ASPP, self).__init__()
        self.m_aspp_modules = nn.ModuleList()

        # 1x1 Convolution
        self.m_aspp_modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
            )
        )

        # ASPP Convolutions of different dilation rates
        for dilation in dilation_rates:
            self.m_aspp_modules.append(
                ASPP_Conv(in_channels, 256, 3, dilation=dilation)
            )

        # ASPP Pooling
        self.m_aspp_modules.append(ASPPPooling(in_channels, 256, 1))

        self.m_last_conv = nn.Sequential(
            nn.Conv2d(256 * (len(self.m_aspp_modules)), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        outputs = []
        for module in self.m_aspp_modules:
            o = module(x)
            outputs.append(o)

        x = torch.cat(outputs, dim=1)
        x = self.m_last_conv(x)
        return x
