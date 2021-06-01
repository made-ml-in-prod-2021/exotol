import pandas as pd
import numpy as np
from py._path.local import LocalPath
from src.entities.feature_params import FeatureParams
from src.features.transformers import (
    StandardScaler,
    read_map_features_transformers
)
from sklearn.base import TransformerMixin


def test_read_map_features_transformers(
        simple_feature_map_transformers: LocalPath
):
    feature_params = FeatureParams(
        target=["none"],
        features_and_transformers_map=simple_feature_map_transformers
    )
    mapper_feature_to_transformer = read_map_features_transformers(
        feature_params
    )
    assert len(mapper_feature_to_transformer[0][1]) == 4
    assert isinstance(mapper_feature_to_transformer[0][1], list)
    assert isinstance(mapper_feature_to_transformer[0][0], TransformerMixin)


def test_transform(fake_dataset: pd.DataFrame):
    scaler = StandardScaler()
    transformed_data = scaler.fit_transform(
        fake_dataset[["trestbps", "chol"]].values
    )
    assert np.isclose(np.mean(transformed_data[:, 0]), 0.0)
    assert np.isclose(np.std(transformed_data[:, 0]), 1.0)
    assert np.isclose(np.mean(transformed_data[:, 1]), 0.0)
    assert np.isclose(np.std(transformed_data[:, 1]), 1.0)

