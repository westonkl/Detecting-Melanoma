import os
from pathlib import Path

import albumentations
import numpy as np
import pretrainedmodels
import torch
from flask import Flask, render_template, request
from torch import nn
from torch.nn import functional as F

from dataset import ClassificationDataset
from engine import Engine

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "static"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model0.bin"))
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" for local/docker or "cuda"
MODEL = None


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
        out = torch.sigmoid(self.out(x))
        loss = 0
        return out, loss


def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            albumentations.HorizontalFlip(p=0.5),
        ]
    )

    test_images = [str(image_path)]
    test_targets = [0]

    test_dataset = ClassificationDataset(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    predictions = Engine.predict(test_loader, model, DEVICE)
    return np.vstack(predictions).ravel()


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
            image_location = UPLOAD_FOLDER / image_file.filename
            image_file.save(image_location)
            pred = predict(str(image_location), MODEL)[0]
            return render_template(
                "index.html", prediction=pred, image_loc=image_file.filename
            )
    return render_template("index.html", prediction=0, image_loc=None)


# model0.bin was the name of my model, to create your own train yours using main.py
if __name__ == "__main__":
    UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL = SEResNext50_32x4d(pretrained=None)
    if MODEL_PATH.exists():
        MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    app.run(host="0.0.0.0", port=12000, debug=True)
