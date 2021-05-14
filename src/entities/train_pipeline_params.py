import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema


@dataclass()
class TrainPipelineParams:
    input_data_path: str
    output_model_path: str


TrainPipelineParamsSchema = class_schema(TrainPipelineParams)


def read_train_parameters(config_path: str) -> TrainPipelineParams:
    schema = TrainPipelineParamsSchema()
    with open(config_path, "r") as input_stream:
        return schema.load(yaml.safe_load(input_stream))
