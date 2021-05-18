import logging
import pickle as pkl
from typing import Any, Dict

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestClassifier
)
from src.entities import ModelParameters, TrainPipelineParams
from src.entities.predict_pipeline_params import PredictPipelineParams

logger = logging.getLogger("models")


def model_factory_method(name_model: str) -> BaseEstimator:
    logger.debug("start model_factory_method")
    if name_model == "LogisticRegression":
        model = LogisticRegression()
    elif name_model == "RandomForestClassifier":
        model = RandomForestClassifier()
    else:
        model = None
    if model is None:
        raise NotImplemented(f"{name_model} not implemented yet!")
    logger.debug("stop model_factory_method")
    return model


def create_model(model_params: ModelParameters):
    logger.debug("start create_model")
    model = model_factory_method(model_params.name)
    model.set_params(**model_params.hyper_params.to_dict())
    logger.info(f"Created model: {model}")
    logger.debug("end create_model")
    return model


def serialize(data: Dict[str, Any], settings: TrainPipelineParams):
    logger.debug("start serialize")
    with open(settings.output_model_path, "wb") as in_stream:
        pkl.dump(data, in_stream, protocol=pkl.HIGHEST_PROTOCOL)
    logger.debug("stop serialize")


def deserialize(settings: PredictPipelineParams) -> Dict[str, Any]:
    logger.debug("start deserialize")
    with open(settings.model_path, "rb") as in_stream:
        result = pkl.load(in_stream)
    logger.debug("stop deserialize")
    return result
