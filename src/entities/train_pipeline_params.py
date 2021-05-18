import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema

from src.entities.logging_params import LoggingParameters
from src.entities.split_params import SplitParameters
from src.entities.feature_params import FeatureParams
from src.entities.model_params import ModelParameters


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str
    log_params: LoggingParameters
    split_params: SplitParameters
    feature_params: FeatureParams
    model_params: ModelParameters


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_parameters(config_path: str) -> TrainPipelineParams:
    schema = TrainPipelineParamsSchema()
    with open(config_path, "r") as input_stream:
        return schema.load(yaml.safe_load(input_stream))
