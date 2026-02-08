import torch
import torch.nn as nn

class PROFA_Encoder(nn.Module):
    """
    Hierarchical Visual Perception (PRO-FA)
    Extracts image features at 3 levels:
    Pixel, Region, Organ
    """

    def __init__(self):
        super(PROFA_Encoder, self).__init__()

        # Pixel level (edges & textures)
        self.pixel_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Region level (structures)
        self.region_layer = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Organ level (global pattern)
        self.organ_layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

    def forward(self, x):

        pixel = self.pixel_layer(x)
        region = self.region_layer(pixel)
        organ = self.organ_layer(region)

        pixel = torch.flatten(pixel,1)
        region = torch.flatten(region,1)
        organ = torch.flatten(organ,1)

        return pixel, region, organ
