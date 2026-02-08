"""
Clinical evaluation utilities
Simulated CheXpert-style evaluation
"""

import numpy as np
from sklearn.metrics import f1_score

def chexpert_f1(y_true, y_pred):
    """
    Multi-label F1 score for disease classification
    """
    y_pred = (y_pred > 0.5).astype(int)
    return f1_score(y_true, y_pred, average="macro")

def simple_radgraph(predicted_diseases, anatomical_map):
    """
    Simplified structural reasoning evaluation
    Checks if diseases are mapped to anatomy
    """
    score = 0
    for d in predicted_diseases:
        if d in anatomical_map:
            score += 1
    return score / len(predicted_diseases)

if __name__ == "__main__":
    print("Evaluation metrics ready")
