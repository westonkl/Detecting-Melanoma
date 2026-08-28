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

## Installation & Setup with `uv`

### Prerequisites
- [uv](https://docs.astral.sh/uv/) installed on your system:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Or via Homebrew
  brew install uv
  ```
- PyTorch (CUDA-enabled GPU recommended for training, CPU supported for inference)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/westonkl/Detecting-Melanoma.git
   cd Detecting-Melanoma
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # Install project dependencies in editable mode
   uv pip install -e .

   # Or include development tools (ruff, pytest, black, flake8)
   uv pip install -e ".[dev]"
   ```

   > **Tip**: You can also use `uv sync` to automatically sync the project environment and lockfile:
   > ```bash
   > uv sync --extra dev
   > ```

---

## Usage

You can execute any script directly using `uv run` (which automatically uses the project environment) or run with `python` inside an activated virtual environment.

### 1. Prepare Cross-Validation Folds
Split the dataset into 10 stratified folds:
```bash
uv run create_folds.py
```
This generates `train_folds.csv` with fold assignments.

### 2. Train the Model
Train the SE-ResNeXt-50 model on a chosen fold:
```bash
uv run main.py
```
Trained weights will be saved as `model{fold}.bin` upon achieving improved validation ROC-AUC scores.

### 3. Run the Web Application
Launch the Flask inference server:
```bash
uv run api.py
```
Navigate to `http://localhost:12000` in your web browser to upload lesion images and view malignancy predictions.

### 4. Code Quality & Linting
Run Ruff for linting and formatting:
```bash
uv run ruff check .
uv run ruff format .
```

---

## License

This project is open source and available under the standard MIT License.
