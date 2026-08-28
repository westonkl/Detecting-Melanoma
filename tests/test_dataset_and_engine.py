import albumentations
import pytest
import torch
from torch.utils.data import DataLoader

from dataset import ClassificationDataset
from engine import Engine


@pytest.fixture
def dummy_dataset_files(tmp_path, dummy_pil_image):
    # Save 4 dummy images
    img_paths = []
    for i in range(4):
        p = tmp_path / f"img_{i}.png"
        dummy_pil_image.save(str(p))
        img_paths.append(str(p))

    targets = [0, 1, 0, 1]
    return img_paths, targets


def test_classification_dataset(dummy_dataset_files):
    img_paths, targets = dummy_dataset_files
    aug = albumentations.Compose(
        [albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    )

    dataset = ClassificationDataset(
        image_paths=img_paths, targets=targets, augmentations=aug
    )
    assert len(dataset) == 4

    sample = dataset[0]
    assert "image" in sample
    assert "targets" in sample
    assert sample["image"].shape == (3, 224, 224)
    assert sample["targets"].item() == 0.0


def test_engine_train_step(dummy_dataset_files):
    img_paths, targets = dummy_dataset_files
    dataset = ClassificationDataset(image_paths=img_paths, targets=targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, 1)

        def forward(self, x, targets):
            out = self.fc(x.view(x.size(0), -1))
            loss = torch.nn.BCEWithLogitsLoss()(out, targets.view(-1, 1))
            return out, loss

    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = Engine.train(loader, model, optimizer, device="cpu")

    assert isinstance(loss, float)
    assert loss > 0


def test_engine_evaluate_and_predict(dummy_dataset_files):
    img_paths, targets = dummy_dataset_files
    dataset = ClassificationDataset(image_paths=img_paths, targets=targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3 * 224 * 224, 1)

        def forward(self, x, targets):
            out = self.fc(x.view(x.size(0), -1))
            loss = torch.nn.BCEWithLogitsLoss()(out, targets.view(-1, 1))
            return out, loss

    model = SimpleModel()

    preds, val_loss = Engine.evaluate(loader, model, device="cpu")
    assert preds.shape[0] == 4
    assert isinstance(val_loss, float)

    pred_only = Engine.predict(loader, model, device="cpu")
    assert pred_only.shape[0] == 4
