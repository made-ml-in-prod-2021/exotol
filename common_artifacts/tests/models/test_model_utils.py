from sklearn.linear_model import LogisticRegression

from heart_disease_common.entities import ModelParameters
from heart_disease_common.entities.model_params import HyperParameters
from heart_disease_common.models.model_utils import create_model


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
