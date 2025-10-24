# AI Retinal Screening for Diabetic Retinopathy

This skill implements an AI system for detecting diabetic retinopathy (DR) from retinal fundus images using a Convolutional Neural Network (CNN).

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Prepare data:
   - Download a dataset like Kaggle Diabetic Retinopathy Detection
   - Organize into `data/train/` and `data/val/` folders with subfolders for each class (0-4)

3. Train the model:
   ```
   python train.py
   ```

4. Predict on new images:
   ```
   python predict.py path/to/retinal_image.jpg
   ```

## Model

The model uses a simple CNN architecture:
- 3 Convolutional layers with ReLU activation and MaxPooling
- Flatten layer
- Dense layers with Dropout
- Softmax output for 5 classes (No DR to Proliferative DR)

## Note

This is a basic implementation. For production use, consider using pre-trained models like ResNet or EfficientNet for better performance.
