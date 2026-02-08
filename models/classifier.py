import torch
import torch.nn as nn

class MIX_MLP(nn.Module):
    """
    Knowledge-Enhanced Multi-label Classification
    Predicts 14 chest diseases from organ-level features
    """

    def __init__(self, input_dim=64, num_labels=14):
        super(MIX_MLP, self).__init__()

        # reasoning path
        self.reasoning = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # classification head
        self.classifier = nn.Linear(64, num_labels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.reasoning(x)
        x = self.classifier(x)
        return self.sigmoid(x)
