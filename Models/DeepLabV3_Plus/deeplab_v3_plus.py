import torch
import torch.nn as nn
import torch.nn.functional as F

from .ASPP_Module import ASPP
from .deeplab_v3_plus_decoder import Decoder
from .deeplab_v3_plus_encoder import DeepLabV3_Plus_Encoder as Encoder
from .deeplab_v3_plus_encoder import xception
from Config.config_deeplab_v3_plus import BACKBONE_CHANNEL_SIZE
from Config.config_deeplab_v3_plus import IMAGE_SIZE


class DeepLabV3_Plus(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3_Plus, self).__init__()
        self.m_backbone = xception()

        self.m_aspp = ASPP(BACKBONE_CHANNEL_SIZE, [6, 12, 18])

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(BACKBONE_CHANNEL_SIZE),
            nn.ReLU()
        )

        self.m_decoder = Decoder(n_classes)


    def forward(self, input):
        backbone_features = self.m_backbone(input)
        backbone_features = self.bn_relu(backbone_features)

        x = self.m_aspp(backbone_features)
        x = self.m_decoder(x, backbone_features)

        return x
