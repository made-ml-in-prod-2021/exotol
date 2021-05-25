import logging
import logging.config
import os
import pickle as pkl
from typing import (Any, Dict, List, Optional, Union)
import pandas as pd
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    conlist,
    validator
)
from sklearn.pipeline import Pipeline


logger = logging.getLogger("inference")


PATH_TO_MODEL = "PATH_TO_MODEL"
PORT = "PORT"
PATH_TO_LOGGING_CONF = "PATH_TO_LOGGING_CONF"
STATUS_CODE = 400

ID_COL = "Id"

FEATURES_MIN_MAX = [
    ("age", None, None),
    ("sex", 0, 1),
    ("cp", 0, 3),
    ("trestbps", None, None),
    ("chol", None, None),
    ("fbs", 0, 1),
    ("restecg", 0, 2),
    ("thalach", None, None),
    ("exang", 0, 1),
    ("oldpeak", None, None),
    ("slope", 0, 2),
    ("ca", 0, 4),
    ("thal", 0, 3)
]

FEATURE_NAMES = [f for f, _, _ in FEATURES_MIN_MAX]
N_FEATURES = len(FEATURE_NAMES)
OUT_OF_RANGE_ERROR = "Out of range feature value"
INCORRECT_FEATURE_ORDER = "Incorrect feature order"

load_dotenv()

logger = logging.getLogger("inference")

app = FastAPI()


class PatientRequest(BaseModel):
    data: conlist(
        conlist(
            Union[float, int],
            min_items=N_FEATURES,
            max_items=N_FEATURES
        ),
        min_items=1,
    )
    features: conlist(str, min_items=N_FEATURES, max_items=N_FEATURES)

    @validator("data")
    def data_consistency(cls, data):
        for sample in data:
            for val, (_, v_min, v_max) in zip(sample, FEATURES_MIN_MAX):
                if v_min is None and v_max is None:
                    continue

                if not (v_min <= val <= v_max):
                    raise ValueError(OUT_OF_RANGE_ERROR)

        return data

    @validator("features")
    def feature_consistency(cls, features):
        if features != FEATURE_NAMES:
            raise ValueError(INCORRECT_FEATURE_ORDER)

        return features


class HeartDiseaseResponse(BaseModel):
    id: int
    target: int


model: Optional[Pipeline] = None


def deserialize(path: str) -> Dict[str, Any]:
    logger.debug("start deserialize")
    with open(path, "rb") as in_stream:
        result = pkl.load(in_stream)
    logger.debug("stop deserialize")
    return result


@app.on_event("startup")
def load_model():
    logger.debug("start load_model")
    global model
    model_path = os.getenv(PATH_TO_MODEL)
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    loaded_dict = deserialize(model_path)
    model = loaded_dict["model"]
    logger.debug("stop load_model")


@app.get("/")
def main():
    logger.debug("start main")
    logger.info("request on /")
    string = "it is entry  point of our predictor"
    logger.debug("stop main")
    return string


@app.get("/health")
def health() -> bool:
    logger.debug("start health")
    logger.info("request on /health")
    is_model_loaded = not (model is None)
    logger.debug("stop health")
    return is_model_loaded


def make_predict(
        data: List[List[Union[float, int]]],
        features: List[str],
        model: Pipeline
) -> List[HeartDiseaseResponse]:
    logger.debug("start make_predict")
    df = pd.DataFrame(data, columns=features)
    logger.info("Data shape: {}".format(df.shape))
    predicts = model.predict(df)

    response = [
        HeartDiseaseResponse(id=index, target=target)
        for index, target in zip(df.index.values, predicts)
    ]
    logger.debug("stop make_predict")
    return response


@app.get("/predict", response_model=List[HeartDiseaseResponse])
def predict(request: PatientRequest):
    logger.info("request on /predict")
    return make_predict(
        request.data,
        request.features,
        model
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exception):
    return JSONResponse(
        status_code=STATUS_CODE,
        content=jsonable_encoder({
            "trace": exception.errors(),
            "body": exception.body}
        ),
    )


def set_logging_config(path_to_config: str):
    with open(path_to_config, "r") as input_stream:
        config = yaml.safe_load(input_stream)
        logging.config.dictConfig(config)


if __name__ == "__main__":
    set_logging_config(os.getenv(PATH_TO_LOGGING_CONF, "logging.yaml"))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv(PORT, 8_000))
    )
