import numpy as np
import pandas as pd
from py._path.local import LocalPath
from src.entities.feature_params import FeatureParams
from src.features.build_features import (
    build_transformer,
    create_features,
    create_target
)


def test_build_transformer(
        simple_feature_map_transformers: LocalPath
):
    feature_params = FeatureParams(
        features_and_transformers_map=simple_feature_map_transformers
    )
    transformer = build_transformer(feature_params)
    assert len(transformer.transformers) == 1


def test_create_features(
        simple_df: LocalPath,
        simple_feature_map_transformers: LocalPath
):
    feature_params = FeatureParams(
        target="none",
        features_and_transformers_map=simple_feature_map_transformers
    )
    df = pd.read_csv(simple_df)
    transformer = build_transformer(feature_params)

    features = create_features(transformer, df)

    assert features.shape == df.shape
    assert np.all(features[0, :] == df.values[0, :])


def test_create_target(
        simple_df: LocalPath,
):
    feature_params = FeatureParams(
        target=["h4"],
        features_and_transformers_map=""
    )
    df = pd.read_csv(simple_df)
    targets = create_target(df, feature_params)
    assert len(targets) == 3
    assert targets[0] == [4]

