# Detecting Melanoma with Deep Learning

An end-to-end deep learning project for automated skin lesion classification and melanoma detection using PyTorch, SE-ResNeXt-50 architecture, and an interactive Flask web interface for real-time inference.

---

## Overview

Melanoma is the most serious type of skin cancer, but early detection significantly increases survival rates. This repository provides a complete machine learning pipeline to detect malignant melanoma from dermoscopic skin lesion images:

- **Stratified K-Fold Cross-Validation**: Handles class imbalance through 10-fold stratified splitting.
- **Deep Convolutional Neural Network**: Implements an SE-ResNeXt-50 (32x4d) backbone with pre-trained ImageNet weights and a custom binary classification head.
- **Robust Training Pipeline**: Uses Albumentations for image transformations, ROC-AUC evaluation metric tracking, learning rate scheduling, and early stopping.
- **Interactive Web Interface**: A lightweight Flask web app that allows users to upload lesion images and receive predicted malignancy probabilities in real time.

---

## Project Structure

```text
Detecting-Melanoma/
├── api.py              # Flask web server and inference endpoint
├── create_folds.py     # Script to generate 10-fold stratified cross-validation splits
├── main.py             # Model definition, training loop, evaluation, and test prediction
├── pyproject.toml      # Project metadata, dependencies, and build configuration
├── README.md           # Project documentation
└── templates/
    └── index.html      # Frontend HTML template for image upload and prediction display
```

---

## Model Architecture & Pipeline

1. **Backbone**: Squeeze-and-Excitation ResNeXt-50 (`se_resnext50_32x4d`) pre-trained on ImageNet.
2. **Feature Aggregation**: Adaptive average pooling (`F.adaptive_avg_pool2d`) reducing feature maps to a 2048-dimensional feature vector.
3. **Classification Head**: Fully-connected layer mapping features to a single logit output with binary cross-entropy loss (`BCEWithLogitsLoss`).
4. **Augmentations**: Image normalization using standard ImageNet statistics and random flip augmentations via `albumentations`.
5. **Optimization**: Adam optimizer with `ReduceLROnPlateau` scheduler and early stopping based on ROC-AUC validation score.

---

## Installation

### Prerequisites
- Python 3.8+
- PyTorch (CUDA-enabled GPU recommended for training, CPU supported for inference)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/westonkl/Detecting-Melanoma.git
   cd Detecting-Melanoma
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   For development tools (linters, formatters, pytest):
   ```bash
   pip install -e ".[dev]"
   ```

---

## Usage

### 1. Prepare Cross-Validation Folds
Split the dataset into 10 stratified folds:
```bash
python create_folds.py
```
This generates `train_folds.csv` with fold assignments.

### 2. Train the Model
Train the SE-ResNeXt-50 model on a chosen fold:
```bash
python main.py
```
Trained weights will be saved as `model{fold}.bin` upon achieving improved validation ROC-AUC scores.

### 3. Run the Web Application
Launch the Flask inference server:
```bash
python api.py
```
Navigate to `http://localhost:12000` in your web browser to upload lesion images and view malignancy predictions.

---

## License

This project is open source and available under the standard MIT License.
