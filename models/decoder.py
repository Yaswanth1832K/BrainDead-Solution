# File: decoder.py
import torch
import torch.nn as nn

class RCTA_Decoder(nn.Module):
    """
    Triangular Cognitive Attention
    Image -> Context -> Hypothesis -> Verification
    """

    def __init__(self, feature_dim=1024, hidden_dim=512):
        super(RCTA_Decoder, self).__init__()

        self.context_layer = nn.Linear(feature_dim, hidden_dim)
        self.hypothesis_layer = nn.Linear(hidden_dim, hidden_dim)
        self.verify_layer = nn.Linear(hidden_dim, feature_dim)

        self.generator = nn.Linear(hidden_dim, 500)

    def forward(self, features):

        context = torch.relu(self.context_layer(features))
        hypothesis = torch.relu(self.hypothesis_layer(context))
        verification = self.verify_layer(hypothesis)

        output = self.generator(hypothesis)

        return output

