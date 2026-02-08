# File: encoder.py
# This file will contain the encoder model definition.
import torch
import torch.nn as nn
import torchvision.models as models

class PROFA_Encoder(nn.Module):
    """
    Hierarchical Visual Perception (PRO-FA)
    Extracts image features at three scales:
    pixel, region, organ
    """

    def __init__(self):
        super(PROFA_Encoder, self).__init__()

        backbone = models.densenet121(pretrained=True)

        # Shallow layers (pixel features)
        self.pixel = nn.Sequential(*list(backbone.features.children())[:4])

        # Middle layers (region features)
        self.region = nn.Sequential(*list(backbone.features.children())[4:8])

        # Deep layers (organ features)
        self.organ = nn.Sequential(*list(backbone.features.children())[8:])

        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):

        p = self.pool(self.pixel(x))
        r = self.pool(self.region(x))
        o = self.pool(self.organ(x))

        p = torch.flatten(p,1)
        r = torch.flatten(r,1)
        o = torch.flatten(o,1)

        return p, r, o
