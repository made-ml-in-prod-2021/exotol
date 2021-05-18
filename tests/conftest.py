import sys
sys.path.append("src/")
import pytest
import pandas as pd
import numpy as np
from faker import Faker
from typing import Tuple
from textwrap import dedent
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
        random_seed: 101
        val_size: 0.15
    feature_params:
        features_and_transformers_map: "configs/features_lr.yaml"
        target:
            - target
    model_params:
        name: "LogisticRegression"
        hyper_params:
            random_state: 1
            max_iter: 1000
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


@pytest.fixture()
def fake_dataset(
        seed: int = 100,
        size: int = 10_000
) -> pd.DataFrame:
    return make_dataset(seed=seed, size=size)


@pytest.fixture()
def fake_dataset_path(
        tmpdir: LocalPath,
        seed: int = 100,
        size: int = 10_000
) -> LocalPath:
    df = make_dataset(seed=seed, size=size)
    fake = tmpdir.join("sample_fake.csv")
    df.to_csv(fake, index=False)
    return fake



@pytest.fixture()
def map_features_to_transformer_fake(tmpdir: LocalPath) -> LocalPath:
    feature_to_transformer = dedent("""
    NoneTransformer:
      - age
      - sex
      - cp
      - fbs
      - restecg
      - thalach
      - exang
      - oldpeak
      - slope
      - ca
      - thal
    StandardScaler:
      - trestbps
      - chol
    """)
    feat2trmers_test = tmpdir.join("feature_test_fake.yaml")
    feat2trmers_test.write(feature_to_transformer)
    return feat2trmers_test


def make_dataset(seed: int, size: int) -> pd.DataFrame:
    """
    age: The person's age in years
    sex: The person's sex (1 = male, 0 = female)
    cp: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
    trestbps: The person's resting blood pressure (mm Hg on admission to the hospital)
    chol: The person's cholesterol measurement in mg/dl
    fbs: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
    restecg: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
    thalach: The person's maximum heart rate achieved
    exang: Exercise induced angina (1 = yes; 0 = no)
    oldpeak: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
    slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
    ca: The number of major vessels (0-3)
    thal: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
    target: Heart disease (0 = no, 1 = yes)
    """
    """ Generates random tests data. """
    np.random.seed(seed)

    df = pd.DataFrame()

    df['age'] = np.random.randint(29, 77, size=size)
    df['cp'] = np.random.randint(4, size=size)
    df['sex'] = np.random.randint(2, size=size)
    df['trestbps'] = np.random.randint(94, 200, size=size)
    df['chol'] = np.random.randint(126, 564, size=size)
    df['fbs'] = np.random.randint(2, size=size)
    df['restecg'] = np.random.randint(2, size=size)
    df['thalach'] = np.random.randint(71, 202, size=size)
    df['exang'] = np.random.randint(2, size=size)
    df['oldpeak'] = 4 * np.random.rand()
    df['slope'] = np.random.randint(3, size=size)
    df['ca'] = np.random.randint(5, size=size)
    df['thal'] = np.random.randint(4, size=size)
    df['target'] = np.random.randint(2, size=size)
    return df
