import pandas as pd
from fastapi.testclient import TestClient
from py._path.local import LocalPath

from heart_disease_app.app import app, load_model

client = TestClient(
    app
)

load_model()


def test_predict_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "it is entry  point of our predictor"


def test_predict_validation_error_400(fake_dataset_path: LocalPath):
    df = pd.read_csv(fake_dataset_path)

    sample = df.sample(100).head(1)
    items = [el for el in sample.values[0]]
    request = {
        "data": [items],
        "features": [col for col in sample.columns[::-1]]
    }

    response = client.get(
        "/predict",
        json=request
    )

    assert response.status_code == 400
    assert "trace" in response.json()
    assert "body" in response.json()


def test_predict(fake_dataset_path: LocalPath):
    df = pd.read_csv(fake_dataset_path)

    for _ in range(100):
        sample = df.sample(100).head(1)
        items = [el for el in sample.values[0]]
        request = {
            "data": [items],
            "features": [col for col in sample.columns]
        }

        response = client.get(
            "/predict",
            json=request
        )

        assert response.status_code == 200
        assert "id" in response.json()[0]
        assert "target" in response.json()[0]
