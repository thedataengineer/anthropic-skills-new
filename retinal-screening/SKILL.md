---
name: retinal-screening
description: AI-powered retinal screening for detecting diabetic retinopathy from fundus images. Use this skill when analyzing retinal images for diabetic retinopathy classification.
---

# AI Retinal Screening Skill

This skill implements an AI system for detecting diabetic retinopathy (DR) from retinal fundus images using a Convolutional Neural Network (CNN).

## Usage

1. Train the model with `python train.py` (requires dataset in data/train and data/val)
2. Predict on images with `python predict.py image.jpg`

## Files
- `train.py`: Training script
- `predict.py`: Inference script
- `requirements.txt`: Dependencies
- `data_download.py`: Dataset helper
- `README.md`: Detailed instructions
