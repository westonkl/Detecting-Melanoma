import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from api import app, predict


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_get_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Please upload image" in response.data
    # When no prediction made yet, image preview shouldn't be rendered
    assert b"Prediction:" not in response.data


def test_api_post_without_file(client):
    response = client.post("/", data={})
    # Should handle missing file gracefully or with 400 Bad Request
    assert response.status_code in [200, 400]


def test_api_post_with_image(client, tmp_path):
    # Create fake image bytes
    data = {"image": (io.BytesIO(b"fake_image_data"), "test_lesion.png")}

    with (
        patch("api.UPLOAD_FOLDER", str(tmp_path)),
        patch("api.predict", return_value=np.array([0.75])),
    ):
        response = client.post("/", data=data, content_type="multipart/form-data")

        assert response.status_code == 200
        # 0.75 * 100 = 75.0%
        assert b"75.0% chance of malignancy" in response.data
        assert b"test_lesion.png" in response.data


def test_api_predict_function(tmp_path):
    # Create dummy image file
    fake_img = tmp_path / "sample.png"
    fake_img.write_bytes(b"dummy image bytes")

    mock_model = MagicMock()
    mock_predictions = [np.array([[0.82]])]

    with (
        patch("api.ClassificationDataset") as mock_loader,
        patch("api.Engine.predict", return_value=mock_predictions),
    ):
        result = predict(str(fake_img), mock_model)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert np.isclose(result[0], 0.82)
        mock_loader.assert_called_once()
