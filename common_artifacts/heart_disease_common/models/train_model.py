import numpy as np
import logging
from sklearn.base import BaseEstimator


logger = logging.getLogger("models")


def train_model(
        model: BaseEstimator,
        train_features: np.array,
        train_target: np.array
):
    logger.debug("start train_model")
    model.fit(
        train_features,
        train_target
    )
    logger.debug("stop train_model")