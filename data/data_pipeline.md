# Data Pipeline

Datasets intended:
- MIMIC-CXR (training)
- IU-Xray (benchmark)

Due to restricted clinical access and dataset size, this repository demonstrates the full inference pipeline using a sample chest X-ray while maintaining the same preprocessing steps.

Preprocessing:
1. Load chest X-ray image
2. Resize to 224x224
3. Normalize intensity
4. Convert to tensor
5. Pass into hierarchical encoder

This pipeline matches clinical deployment workflow.
