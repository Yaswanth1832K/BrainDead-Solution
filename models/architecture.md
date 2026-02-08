Cognitive Radiology Architecture

The system follows diagnosis-first reasoning rather than captioning.

PRO-FA:
Extracts hierarchical features from X-ray images at pixel, region and organ levels.

MIX-MLP:
Predicts disease labels before text generation using multi-label classification.

RCTA:
Implements a reasoning loop:
Image → Context → Hypothesis → Verification → Report

This mimics how radiologists interpret X-rays.
