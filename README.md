# 🧠 BrainDead 2K26 Hackathon Submission
### Revelation 2026 – IIEST Shibpur

This repository contains our solutions for **both problem statements** of the BrainDead Hackathon.

---

# 🩺 Problem Statement 2 : Cognitive Radiology Second Reader
## AI Radiology Report Generation from Chest X-Ray Images

### Objective
Radiologists often experience fatigue while reviewing large volumes of medical images.  
Our goal is to build an AI **Second Reader** that assists doctors by automatically analyzing a chest X-ray and drafting a structured clinical report.

The system does **not** behave like a simple image captioning model.  
Instead, it follows a reasoning-based workflow that mimics clinical decision making.

---

## Proposed Cognitive Framework

We designed an interpretable deep learning pipeline inspired by radiologist reasoning.

The system follows three mandatory modules:

---

### 1️⃣ PRO-FA — Hierarchical Visual Perception
Radiologists do not look at an X-ray in a single step.  
They observe:

• small textures  
• anatomical regions  
• whole organs  

Our encoder replicates this behavior by extracting features at three levels:

| Level | What it Represents |
|------|------|
Pixel Level | Edges and local textures |
Region Level | Lung zones and structures |
Organ Level | Overall chest anatomy |

This provides interpretable visual understanding instead of black-box feature extraction.

---

### 2️⃣ MIX-MLP — Knowledge-Enhanced Diagnosis
Before writing a report, a doctor first forms a **diagnostic hypothesis**.

Our model predicts multiple diseases from the organ-level features:

- Pneumonia
- Pleural Effusion
- Atelectasis
- Edema
- Cardiomegaly
- Consolidation
- Fibrosis
- and other thoracic conditions

This diagnosis-first approach prevents hallucinated reports and improves clinical reliability.

---

### 3️⃣ RCTA — Cognitive Report Generation
Instead of directly captioning the image, the system performs a reasoning loop:

**Image → Predicted Diseases → Verification → Report**

The final output is a structured clinical report containing:

**FINDINGS** – Observations detected in the image  
**IMPRESSION** – Final medical interpretation

This mirrors how radiologists write reports in practice.

---

## Cognitive Workflow
1. Input chest X-ray is analyzed at multiple visual scales  
2. Possible diseases are predicted  
3. Predictions are verified  
4. A structured clinical report is generated  

This demonstrates **cognitive simulation rather than black-box captioning**.

---

## Demo Instructions
Open the notebook: notebooks/radiology_demo.ipynb

Run the notebook in Google Colab and click **Runtime → Run All**.

The notebook will:

1. Load a chest X-ray image
2. Extract hierarchical features
3. Predict diseases
4. Generate a structured radiology report

---

## Repository Structure
BrainDead-Solution/
│
├── data/ # preprocessing documentation
├── models/
│ ├── encoder.py # PRO-FA implementation
│ ├── classifier.py # MIX-MLP implementation
│ └── decoder.py # RCTA report generator
│
├── training/ # training strategy
├── evaluation/ # clinical evaluation description
├── notebooks/
│ └── radiology_demo.ipynb
│
├── requirements.txt
└── README.md


---

## Key Contributions
- Interpretable medical AI system
- Diagnosis-first report generation
- Structured clinical report output
- Reproducible end-to-end pipeline
- Avoids black-box encoder-decoder captioning

## Reproducibility

The repository includes:
- data pipeline for clinical datasets
- training loop
- evaluation metrics
- inference notebook

The architecture is directly trainable on MIMIC-CXR and IU-Xray datasets.

---

# 🎬 Problem Statement 1: ReelSense – Explainable Movie Recommender

## Overview
ReelSense is an explainable and diversity-aware movie recommendation system built using the MovieLens dataset.

The system focuses on **trustworthy recommendations**, not just rating prediction.

---

## Implemented Recommendation Models
- Popularity-based recommender
- User-User collaborative filtering
- Item-Item collaborative filtering
- Matrix factorization (SVD)
- Hybrid recommendation model

---

## Explainability
Each recommendation is accompanied by a natural language explanation:

Example:
> “Recommended because you liked science-fiction movies such as Inception and The Matrix.”

---

## Evaluation Metrics
The recommender system is evaluated using:

Accuracy Metrics:
- RMSE
- MAE

Ranking Metrics:
- Precision@K
- Recall@K
- NDCG
- MAP

Diversity Metrics:
- Catalog Coverage
- Novelty
- Intra-List Diversity

Notebook: PS1_ReelSense/ReelSense.ipynb

---

## Final Note
This submission demonstrates interpretable AI systems in two domains:

**Healthcare** – clinical decision support (radiology assistant)  
**Recommender Systems** – trustworthy and explainable recommendations

The common goal of both solutions is improving **trust, transparency, and reliability in AI systems**.

