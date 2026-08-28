import os
from pathlib import Path

import albumentations
import numpy as np
import pandas as pd
import pretrainedmodels
import torch
from sklearn import metrics
from torch import nn
from torch.nn import functional as F

from dataset import ClassificationDataset
from engine import EarlyStopping, Engine

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = Path(os.getenv("MELANOMA_INPUT_PATH", BASE_DIR / "input"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super().__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](
            pretrained=pretrained
        )
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        loss = nn.BCEWithLogitsLoss()(out, targets.reshape(-1, 1).type_as(out))
        return out, loss


def train(fold):
    training_data_path = INPUT_PATH / "train224"
    df = pd.read_csv(INPUT_PATH / "train_folds.csv")

    device = DEVICE
    epochs = 50
    train_bs = 32
    valid_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            albumentations.HorizontalFlip(p=0.5),
        ]
    )

    valid_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            albumentations.HorizontalFlip(p=0.5),
        ]
    )

    train_images = [str(training_data_path / f"{img_id}.png") for img_id in df_train.image_name.values]
    train_targets = df_train.target.values

    valid_images = [str(training_data_path / f"{img_id}.png") for img_id in df_valid.image_name.values]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationDataset(
        image_paths=train_images,
        targets=train_targets,
        resize=None,
        augmentations=train_aug,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True, num_workers=4
    )

    valid_dataset = ClassificationDataset(
        image_paths=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, mode="max"
    )

    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        _training_loss = Engine.train(
            train_loader,
            model,
            optimizer,
            device,
            fp16=False,  # set to true if using amp
        )
        predictions, _valid_loss = Engine.evaluate(
            valid_loader,
            model,
            device,
        )
        predictions = np.vstack(predictions).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, str(MODEL_DIR / f"model{fold}.bin"))
        if es.early_stop:
            print("early stopping")
            break


def predict(fold):
    test_data_path = INPUT_PATH / "test224"
    df_test = pd.read_csv(INPUT_PATH / "test.csv")
    df_test.loc[:, "target"] = 0

    device = DEVICE
    test_bs = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            albumentations.HorizontalFlip(p=0.5),
        ]
    )

    test_images = [str(test_data_path / f"{img_id}.png") for img_id in df_test.image_name.values]
    test_targets = df_test.target.values

    test_dataset = ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False, num_workers=4
    )

    model = SEResNext50_32x4d(pretrained="imagenet")
    model.load_state_dict(torch.load(MODEL_DIR / f"model{fold}.bin", map_location=torch.device(device)))
    model.to(device)

    predictions = Engine.predict(test_loader, model, device)
    return np.vstack(predictions).ravel()


if __name__ == "__main__":
    train(fold=0)
    predict(fold=0)
