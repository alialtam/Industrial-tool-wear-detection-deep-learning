# Multimodal Tool Wear Prediction using Deep Learning

## Overview
This project focuses on predicting industrial tool wear using a multimodal deep learning approach.

It combines multiple image modalities to:
- Classify tool condition (Sharp / Used / Dulled)
- Predict wear measurements (flank wear, gaps, overhang in μm)

---

## What I Built
- Multi-input deep learning model (9 modalities)
- Transfer learning using MobileNetV2
- Feature fusion architecture
- Dual-task learning:
  - Classification
  - Regression

---

## Dataset
- 512 samples
- 10 tools
- 9 modalities per sample

---

## Model Architecture
- Backbone: MobileNetV2 (pretrained)
- Shared encoder across all modalities
- Feature fusion using fully connected layers
- Two output heads:
  - Classification (3 classes)
  - Regression (3 values)

---

## Evaluation
- 10-Fold Leave-One-Tool-Out Cross Validation
- Ensures generalization to unseen tools

---

## Key Highlights
- Efficient batching of 9 modalities in one forward pass
- Reduced training time significantly
- Solved overfitting using dropout & augmentation
- Achieved up to **100% validation accuracy**
