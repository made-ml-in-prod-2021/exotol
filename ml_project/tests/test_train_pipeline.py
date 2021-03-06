import os
import pytest
from py._path.local import LocalPath
from heart_disease_common.entities import (
    FeatureParams,
    LoggingParameters,
    ModelParameters,
    SplitParameters,
    TrainPipelineParams
)
from heart_disease_common.entities.model_params import HyperParameters
from heart_disease_train.train_pipeline import train_pipeline


def test_train_pipeline(fake_dataset_path: LocalPath):
    if not os.path.exists("./tests/hub"):
        os.mkdir("./tests/hub")
    settings = TrainPipelineParams(
        input_data_path=fake_dataset_path,
        output_model_path="./tests/hub/model.pkl",
        log_params=LoggingParameters(
            path_to_config="./tests/logging.yaml"
        ),
        split_params=SplitParameters(
            random_seed=1001,
            val_size=0.1
        ),
        feature_params=FeatureParams(
            target=["target"],
            features_and_transformers_map="./tests/features_lr.yaml"
        ),
        model_params=ModelParameters(
            name="LogisticRegression",
            hyper_params=HyperParameters(
                random_state=1001,
                max_iter=1001,
                n_jobs=None,
                n_estimators=None
            )
        )
    )
    model_output_path, metrics = train_pipeline(settings)
    assert metrics["roc_auc_score"] > 0
    assert os.path.exists(model_output_path)
    assert os.path.exists(settings.output_model_path)