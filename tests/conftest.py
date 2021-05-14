from textwrap import dedent

import pytest
from py._path.local import LocalPath


@pytest.fixture()
def simple_config_path(tmpdir: LocalPath) -> LocalPath:
    text_config = dedent("""
    input_data_path: ./data/tmp/data.csv
    output_model_path: ./models/tmp/model.pkl
    """)
    config = tmpdir.join("simple_config.yaml")
    config.write(text_config)
    return config
