import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger("models")


def create_inference_pipeline(
        model: BaseEstimator,
        transformer: ColumnTransformer
):
    logger.debug("start create_inference_pipeline")
    pipeline = Pipeline([
        ("transformer", transformer),
        ("model", model)
    ])
    logger.debug("stop create_inference_pipeline")
    return pipeline


def predict_model(model: Pipeline, data: pd.DataFrame) -> np.array:
    logger.debug("start predict_model")
    predictions = model.predict(data)
    logger.debug("stop predict_model")
    return predictions


def evaluate_model(predicts: np.ndarray, target: np.array) -> Dict[str, float]:
    logger.debug("start evaluate_model")
    scores = {
        "roc_auc_score": metrics.roc_auc_score(
            y_true=target,
            y_score=predicts
        ),
        "precision_recall": metrics.classification_report(
            y_true=target,
            y_pred=predicts
        )
    }
    logger.debug("stop evaluate_model")
    return scores
