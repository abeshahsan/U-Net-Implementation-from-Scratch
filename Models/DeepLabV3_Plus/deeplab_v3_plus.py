import torch
import torch.nn as nn
import torch.nn.functional as F

from .ASPP_Module import ASPP
from .deeplab_v3_plus_decoder import Decoder
from .deeplab_v3_plus_encoder import xception as Encoder


class DeepLabV3_Plus(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3_Plus, self).__init__()
        self.m_backbone = Encoder()

        # Freeze the backbone
        for param in self.m_backbone.parameters():
            param.requires_grad = False

        self.m_decoder = Decoder(n_classes)

    def forward(self, input):
        device = input.device
        backbone_features = self.m_backbone(input)
        # backbone_features = backbone_features.view(backbone_features.shape[0], -1, 128, 128)
        x = ASPP(backbone_features.shape[1], [6, 12, 18]).to(device)(backbone_features)
        x = self.m_decoder(x, backbone_features)

        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)

        return x
