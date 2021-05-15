from py._path.local import LocalPath
from src.entities.feature_params import FeatureParams
from src.features.transformers import read_map_features_transformers
from typing import List, Tuple
from sklearn.base import TransformerMixin


def test_read_map_features_transformers(
        simple_feature_map_transformers: LocalPath
):
    feature_params = FeatureParams(
        features_and_transformers_map=simple_feature_map_transformers
    )
    mapper_feature_to_transformer = read_map_features_transformers(
        feature_params
    )
    assert len(mapper_feature_to_transformer[0][1]) == 4
    assert isinstance(mapper_feature_to_transformer[0][1], list)
    assert isinstance(mapper_feature_to_transformer[0][0], TransformerMixin)
