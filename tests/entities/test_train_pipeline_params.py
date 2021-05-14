from py._path.local import LocalPath

from src.entities.train_pipeline_params import (
    read_train_parameters,
    TrainPipelineParams
)


def test_read_train_parameters(simple_config_path: LocalPath):
    simple_config_path: LocalPath
    settings = read_train_parameters(simple_config_path)
    assert isinstance(settings, TrainPipelineParams)
    assert settings.input_data_path == "./data/tmp/data.csv"
    assert settings.output_model_path == "./models/tmp/model.pkl"

