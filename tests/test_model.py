import torch

from api import SEResNext50_32x4d as APISEResNext50
from main import SEResNext50_32x4d as MainSEResNext50


def test_main_model_initialization():
    model = MainSEResNext50(pretrained=None)
    assert model is not None
    assert hasattr(model, "model")
    assert hasattr(model, "out")
    assert model.out.in_features == 2048
    assert model.out.out_features == 1


def test_main_model_forward(dummy_image_tensor, dummy_targets_tensor):
    model = MainSEResNext50(pretrained=None)
    model.eval()
    with torch.no_grad():
        out, loss = model(dummy_image_tensor, dummy_targets_tensor)

    assert out.shape == (2, 1)
    assert isinstance(loss.item(), float)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_main_model_backward(dummy_image_tensor, dummy_targets_tensor):
    model = MainSEResNext50(pretrained=None)
    model.train()
    _out, loss = model(dummy_image_tensor, dummy_targets_tensor)
    loss.backward()

    # Check that gradients are computed for classification layer
    assert model.out.weight.grad is not None
    assert model.out.bias.grad is not None
    assert not torch.isnan(model.out.weight.grad).any()


def test_api_model_initialization():
    model = APISEResNext50(pretrained=None)
    assert model is not None
    assert hasattr(model, "model")
    assert hasattr(model, "out")
    assert model.out.in_features == 2048
    assert model.out.out_features == 1


def test_api_model_forward(dummy_image_tensor, dummy_targets_tensor):
    model = APISEResNext50(pretrained=None)
    model.eval()
    with torch.no_grad():
        out, loss = model(dummy_image_tensor, dummy_targets_tensor)

    assert out.shape == (2, 1)
    assert loss == 0
    # Output should be sigmoid probabilities between 0 and 1
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()
