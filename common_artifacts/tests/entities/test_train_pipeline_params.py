from py._path.local import LocalPath

from heart_disease_common.entities.train_pipeline_params import (
    TrainPipelineParams,
    read_train_parameters
)


def test_read_train_parameters(simple_config_path: LocalPath):
    settings = read_train_parameters(simple_config_path)
    assert isinstance(settings, TrainPipelineParams)
    assert settings.input_data_path == "./data/tmp/data.csv"
    assert settings.output_model_path == "./models/tmp/model.pkl"


def test_read_train_parameters_check_log_params(
    simple_config_path: LocalPath
):
    settings = read_train_parameters(simple_config_path)
    assert settings.log_params.path_to_config == "./configs/logging.yaml"


def test_read_train_parameters_check_split_parameters(
        simple_config_path: LocalPath
):
    settings = read_train_parameters(simple_config_path)
    assert settings.split_params.random_seed == 101
    assert abs(settings.split_params.val_size - 0.15) < 1e-6
