"""
Training loop for Cognitive Radiology Second Reader
"""

import torch
from models.encoder import PROFA_Encoder
from models.classifier import MIX_MLP

def train_one_epoch(model, classifier, dataloader, optimizer, criterion):
    model.train()
    classifier.train()

    total_loss = 0

    for images, labels in dataloader:
        optimizer.zero_grad()

        _, _, organ = model(images)
        outputs = classifier(organ)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def initialize_training():
    encoder = PROFA_Encoder()
    classifier = MIX_MLP()

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-4
    )

    criterion = torch.nn.BCELoss()

    print("Training pipeline initialized")

if __name__ == "__main__":
    initialize_training()
