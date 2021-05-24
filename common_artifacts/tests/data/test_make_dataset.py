import pytest
import pandas as pd
from py._path.local import LocalPath
from heart_disease_common.data.make_dataset import (
    read_data,
    train_val_split
)
from heart_disease_common.entities.split_params import SplitParameters
from textwrap import dedent


def test_read_data(simple_df):
    data = read_data(simple_df)
    assert data.shape == (3, 4)
    assert list(data.columns) == ['h1', 'h2', 'h3', 'h4']


@pytest.fixture()
def train_df(tmpdir: LocalPath) -> pd.DataFrame:
    sample = dedent("""
        h1,h2,h3,h4
        1,2,3,4
        5,6,7,8
        9,10,11,12
        13,14,15,16
    """)
    csv = tmpdir.join("sample.csv")
    csv.write(sample)
    return pd.read_csv(csv)


def test_train_val_split(
        train_df: LocalPath,
        simple_split_params: SplitParameters
):
    train, valid = train_val_split(train_df, simple_split_params)
    assert train.shape == (3, 4)
    assert valid.shape == (1, 4)
