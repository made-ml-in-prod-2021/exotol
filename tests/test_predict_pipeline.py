import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn import metrics
from src.entities import LoggingParameters
from src.entities.predict_pipeline_params import PredictPipelineParams
from src.predict_pipeline import predict_pipeline


@pytest.mark.depends(on=['test_train_pipeline'])
def test_predict_pipeline_command(fake_dataset_path: LocalPath):
    settings = PredictPipelineParams(
        model_path="./hub/model1.pkl",
        input_data_path=fake_dataset_path,
        output_data_path="./hub/test_features.csv",
        output_target_path="./hub/test_target.csv",
        log_params = LoggingParameters(
            path_to_config="logging.yaml"
        )
    )
    df = pd.read_csv(fake_dataset_path)
    predictions = predict_pipeline(settings)
    assert len(predictions) == 10_000
    assert metrics.roc_auc_score(
            y_score=predictions,
            y_true=df['target'].values
        ) > 0




