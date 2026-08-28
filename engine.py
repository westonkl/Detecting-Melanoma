import os

import numpy as np
import torch


class Engine:
    """Native PyTorch training, evaluation, and prediction engine replacing wtfml.engine.Engine."""

    @staticmethod
    def train(data_loader, model, optimizer, device, fp16=False):
        model.train()
        total_loss = 0.0
        for data in data_loader:
            images = data["image"].to(device)
            targets = data["targets"].to(device)

            optimizer.zero_grad()
            _, loss = model(images, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    @staticmethod
    def evaluate(data_loader, model, device):
        model.eval()
        predictions = []
        total_loss = 0.0
        with torch.no_grad():
            for data in data_loader:
                images = data["image"].to(device)
                targets = data["targets"].to(device)

                out, loss = model(images, targets)
                total_loss += loss.item()

                preds = (
                    torch.sigmoid(out)
                    if not ((out >= 0).all() and (out <= 1).all())
                    else out
                )
                predictions.extend(preds.cpu().numpy().tolist())

        return np.array(predictions), (
            total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
        )

    @staticmethod
    def predict(data_loader, model, device):
        model.eval()
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                images = data["image"].to(device)
                targets = data.get("targets", torch.zeros(images.size(0))).to(device)

                out, _ = model(images, targets)
                preds = (
                    torch.sigmoid(out)
                    if not ((out >= 0).all() and (out <= 1).all())
                    else out
                )
                predictions.extend(preds.cpu().numpy().tolist())

        return np.array(predictions)


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, patience=5, mode="max", delta=0.0001):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = -np.inf if mode == "max" else np.inf

    def __call__(self, epoch_score, model, model_path):
        score = epoch_score if self.mode == "max" else -epoch_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score
