import os
import torch

import albumentations
import pretrainedmodels

import numpy as np
import torch.nn as nn

from flask import Flask
from flask import request
from flask import render_template
from torch.nn import functional as F

from wtfml.data_loaders.image import ClassificationLoader
from wtfml.engine import Engine


app = Flask(__name__)
UPLOAD_FOLDER = "E:/Users/Weston/workspace/Detecting-Melanoma/static"
DEVICE = "cpu" #cpu with docker
MODEL = None


class SEResNext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNext50_32x4d, self).__init__()
        #self.base_model = pretrainedmodels.__dict__[
        self.model = pretrainedmodels.__dict__[
            "se_resnext50_32x4d"
        ](pretrained=pretrained)
        #self.l0 = nn.Linear(2048, 1)
        self.out = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape
        #x = self.base_model.features(image)
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        #out = torch.sigmoid(self.l0(x))
        out = torch.sigmoid(self.out(x))
        loss = 0
        return out, loss


def predict(image_path, model):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.augmentations.transforms.Flip(),
        ]
    )

    test_images = [image_path]
    test_targets = [0]

    test_dataset = ClassificationLoader(
        image_paths=test_images,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    predictions = Engine.predict(
        test_loader,
        model,
        DEVICE
    )
    return np.vstack((predictions)).ravel()


@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location, MODEL)[0]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)

# model loading error- fix convert model.layer0... to base_model.layer0... - was a naming convention issue - full fix when ensembling fold trainings
# to do:
# dockerize
# add title to html, make prediction clearer
# add more data augmentations
# fix apex issues

if __name__ == "__main__":
    MODEL = SEResNext50_32x4d(pretrained=None)
    MODEL.load_state_dict(torch.load("model0.bin", map_location=torch.device(DEVICE)))
    MODEL.to(DEVICE)
    app.run(host="0.0.0.0", port=12000, debug=True)