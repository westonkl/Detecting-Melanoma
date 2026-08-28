import os

import albumentations
import numpy as np
import torch
from sklearn import metrics

from engine import EarlyStopping


def test_albumentations_pipeline():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    aug = albumentations.Compose(
        [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            albumentations.HorizontalFlip(p=1.0),
        ]
    )

    # Synthetic RGB image of shape (224, 224, 3) in [0, 255]
    sample_image = np.uint8(np.random.rand(224, 224, 3) * 255)
    transformed = aug(image=sample_image)["image"]

    assert transformed.shape == (224, 224, 3)
    assert transformed.dtype == np.float32


def test_roc_auc_metric_calculation():
    # Targets: 10 binary labels
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Perfect predictions
    y_pred_perfect = np.array([0.1, 0.2, 0.1, 0.3, 0.2, 0.8, 0.9, 0.7, 0.95, 0.85])
    auc_perfect = metrics.roc_auc_score(y_true, y_pred_perfect)
    assert auc_perfect == 1.0

    # Random / inverted predictions
    y_pred_bad = np.array([0.9, 0.8, 0.7, 0.9, 0.8, 0.1, 0.2, 0.1, 0.3, 0.2])
    auc_bad = metrics.roc_auc_score(y_true, y_pred_bad)
    assert auc_bad == 0.0


def test_early_stopping_saves_and_triggers(tmp_path):
    checkpoint_file = str(tmp_path / "model_best.bin")
    dummy_model = torch.nn.Linear(10, 1)

    # Initialize early stopping with patience of 2 epochs
    es = EarlyStopping(patience=2, mode="max")

    # Epoch 1: score 0.80 (improvement, should save)
    es(0.80, dummy_model, model_path=checkpoint_file)
    assert os.path.exists(checkpoint_file)
    assert not es.early_stop
    assert es.best_score == 0.80

    # Epoch 2: score 0.75 (no improvement, counter = 1)
    es(0.75, dummy_model, model_path=checkpoint_file)
    assert not es.early_stop

    # Epoch 3: score 0.70 (no improvement, counter = 2 -> triggers early stop)
    es(0.70, dummy_model, model_path=checkpoint_file)
    assert es.early_stop
