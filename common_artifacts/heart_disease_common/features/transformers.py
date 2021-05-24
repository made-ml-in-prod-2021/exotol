import logging
from typing import List, Tuple

import numpy as np
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from heart_disease_common.entities.feature_params import FeatureParams

logger = logging.getLogger("features")


class NoneTransformer(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        return X


# class StandardScaler(TransformerMixin, BaseEstimator):
#
#     def fit(self, X, y=None, **fit_params):
#         self.means = np.mean(X, axis=0)
#         self.stds = np.std(X, axis=0)
#         return self
#
#     def transform(self, X, y=None):
#         return (X - self.means) / self.stds


def factory_method(transformer_name: str) -> TransformerMixin:
    logger.debug("start factory_method")
    if transformer_name == "NoneTransformer":
        transformer = NoneTransformer()
    elif transformer_name == "StandardScaler":
        transformer = StandardScaler()
    else:
        transformer = None
    logger.debug("end factory_method")
    if transformer is None:
        raise NotImplemented(f"{transformer_name} transformer not implemented!")
    return transformer


def read_map_features_transformers(
        feature_params: FeatureParams
) -> List[Tuple[TransformerMixin, List[str]]]:
    logger.debug("start read_map_features_transformers")
    mapper_feature_to_transformers = []
    with open(feature_params.features_and_transformers_map, "r") as in_stream:
        dict_data = yaml.safe_load(in_stream)
        for key in dict_data:
            mapper_feature_to_transformers.append(
                (factory_method(key), dict_data[key])
            )
    logger.debug("end read_map_features_transformers")
    return mapper_feature_to_transformers
