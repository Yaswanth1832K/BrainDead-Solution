import torch
import torch.nn as nn

class MIX_MLP(nn.Module):
    """
    Knowledge-Enhanced Multi-label Disease Classification
    """

    def __init__(self, input_dim=1024, num_labels=14):
        super(MIX_MLP, self).__init__()

        # Residual reasoning path
        self.residual = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

        # Expansion path (disease prediction)
        self.expansion = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_labels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, features):

        reasoning = features + self.residual(features)
        logits = self.expansion(reasoning)

        return self.sigmoid(logits)
