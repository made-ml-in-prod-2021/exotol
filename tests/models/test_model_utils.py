from sklearn.linear_model import LogisticRegression

from src.entities import ModelParameters
from src.entities.model_params import HyperParameters
from src.models.model_utils import create_model


def test_create_model():
    model_params = ModelParameters(
        name="LogisticRegression",
        hyper_params=HyperParameters(
            random_state=11,
            n_estimators=None,
            n_jobs=None,
            max_iter=None
        )
    )
    model = create_model(model_params=model_params)
    assert isinstance(model, LogisticRegression)
    assert model.random_state == model_params.hyper_params.random_state
