import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn import metrics
from heart_disease_common.entities import LoggingParameters
from heart_disease_common.entities.predict_pipeline_params import (
    PredictPipelineParams
)
from heart_disease_train.predict_pipeline import predict_pipeline


@pytest.mark.depends(on=['tests/test_train_pipeline.py::test_train_pipeline'])
def test_predict_pipeline_command(fake_dataset_path: LocalPath):
    settings = PredictPipelineParams(
        model_path="./tests/hub/model.pkl",
        input_data_path=fake_dataset_path,
        output_data_path="./tests/hub/test_features.csv",
        output_target_path="./tests/hub/test_target.csv",
        log_params=LoggingParameters(
            path_to_config="./tests/logging.yaml"
        )
    )
    df = pd.read_csv(fake_dataset_path)
    predictions = predict_pipeline(settings)
    assert len(predictions) == 10_000
    assert metrics.roc_auc_score(
            y_score=predictions,
            y_true=df['target'].values
        ) > 0




