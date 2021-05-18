import logging

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import TransformerMixin, BaseEstimator
from src.entities.feature_params import FeatureParams
from src.features.transformers import read_map_features_transformers

logger = logging.getLogger("features")


def build_transformer(
        feature_params: FeatureParams
) -> ColumnTransformer:
    logger.debug("start build_transformers")
    tuples_feat_transformers = read_map_features_transformers(
        feature_params
    )
    logger.debug(f"Transformers: {tuples_feat_transformers}")
    transformer = make_column_transformer(
        *tuples_feat_transformers
    )
    logger.debug("stop build_transformers")
    return transformer


def create_features(
        transformer: ColumnTransformer,
        train_df: pd.DataFrame) -> np.array:
    transformer.fit(train_df)
    logger.debug("start create features")
    features = transformer.transform(train_df)
    if isinstance(features, pd.DataFrame):
        features = features.values
    logger.debug("end create features")
    return features


def create_target(
        train_df: pd.DataFrame,
        feature_params: FeatureParams) -> List[int]:
    logger.debug("start create_target")
    targets = train_df[feature_params.target].values
    logger.debug("end create_target")
    return targets