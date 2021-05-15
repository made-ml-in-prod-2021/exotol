from textwrap import dedent

import pytest
from py._path.local import LocalPath

from src.entities.split_params import SplitParameters


@pytest.fixture()
def simple_config_path(tmpdir: LocalPath) -> LocalPath:
    text_config = dedent("""
    input_data_path: "./data/tmp/data.csv"
    output_model_path: "./models/tmp/model.pkl"
    log_params:
        path_to_config: "./configs/logging.yaml"
    split_params:
        random_seed: 10
        val_size: 0.15
    feature_params:
    """)
    config = tmpdir.join("simple_config.yaml")
    config.write(text_config)
    return config


@pytest.fixture()
def simple_df(tmpdir: LocalPath) -> LocalPath:
    test_df = dedent("""
        h1,h2,h3,h4
        1,2,3,4
        5,6,7,8
        9,10,11,12
    """)
    sample = tmpdir.join("sample.csv")
    sample.write(test_df)
    return sample


@pytest.fixture()
def simple_split_params() -> SplitParameters:
    return SplitParameters(
        val_size=0.1,
        random_seed=1001,
        shuffle=True
    )


@pytest.fixture()
def simple_feature_map_transformers(tmpdir: LocalPath) -> LocalPath:
    feature_to_transformer = dedent("""
    NoneTransformer:
      - h1
      - h2
      - h3
      - h4
    """)
    feat2trmers_test = tmpdir.join("feature_test.yaml")
    feat2trmers_test.write(feature_to_transformer)
    return feat2trmers_test
