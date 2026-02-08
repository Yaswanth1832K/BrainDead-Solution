"""
Data preparation pipeline for Cognitive Radiology Second Reader.
Compatible with IU-Xray and MIMIC-CXR dataset structure.
This script demonstrates how clinical datasets would be prepared
for training and inference.
"""

import os
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch

# Image transform used by the model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def load_image(image_path):
    """
    Load and preprocess a chest X-ray image
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor


def parse_clinical_report(report_text):
    """
    Splits radiology report into findings and impression sections
    """
    report_text = report_text.lower()

    findings = ""
    impression = ""

    if "findings" in report_text:
        findings = report_text.split("findings")[-1]

    if "impression" in report_text:
        impression = report_text.split("impression")[-1]

    return findings.strip(), impression.strip()


def load_iu_xray_metadata(report_csv):
    """
    Load IU-Xray dataset metadata
    """
    df = pd.read_csv(report_csv)

    dataset = []
    for _, row in df.iterrows():
        entry = {
            "image_name": row["filename"],
            "indication": row.get("indication",""),
            "findings": row.get("findings",""),
            "impression": row.get("impression","")
        }
        dataset.append(entry)

    return dataset


def create_training_sample(image_path, indication_text):
    """
    Creates a model-ready training sample
    """
    image_tensor = load_image(image_path)

    sample = {
        "image": image_tensor,
        "clinical_indication": indication_text
    }

    return sample


if __name__ == "__main__":
    print("Radiology data pipeline ready.")
