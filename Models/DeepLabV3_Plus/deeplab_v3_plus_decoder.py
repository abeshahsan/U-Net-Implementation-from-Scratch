"""
DeepLabV3+ decoder module.

This module is responsible for the decoder part of the DeepLabV3+ model.
The decoder module is responsible for taking the outputfrom the
ASPP module and upsampling it to the original image size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config.config_deeplab_v3_plus import IMAGE_SIZE
from Config.config_deeplab_v3_plus import BACKBONE_CHANNEL_SIZE

class Decoder(nn.Module):
    def __init__(self, n_classes):
        """
        Initializes the DeepLabV3_Plus_Decoder module.

        Parameters:
        -----------
        - n_classes (int):
                The number of classes in the dataset.

        Returns:
        -----------
        - None

        """
        super(Decoder, self).__init__()

        self.point_conv_backbone = nn.Conv2d(BACKBONE_CHANNEL_SIZE, 48, 3, padding=1)
        self.m_conv2 = nn.Conv2d(304, 256, 3, padding=1)
        self.m_conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.m_conv4 = nn.Conv2d(256, n_classes, 1)


    def forward(self, x, low_level_features):
        """
        Forward pass of the DeepLabV3_Plus_Decoder module.

        Parameters:
        -----------
        - x (torch.Tensor):
                The output from the ASPP module.
        - low_level_features (torch.Tensor):
                The low level features from the backbone.

        Returns:
        -----------
        - x (torch.Tensor):
                The output from the decoder module.

        """

        # 1x1 Convolution for backbone features

        low_level_features = self.point_conv_backbone(low_level_features)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.m_conv2(x)
        x = F.interpolate(x, size=IMAGE_SIZE, mode='bilinear', align_corners=True)
        x = self.m_conv3(x)
        x = self.m_conv4(x)

        return x
