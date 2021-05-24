import pandas as pd
from py._path.local import LocalPath
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from heart_disease_common.models.train_model import train_model


def test_train_model(simple_df: LocalPath):
    df = pd.read_csv(simple_df)
    model = LogisticRegression(random_state=100500)
    train_model(
        model,
        df[['h1', 'h2', 'h3']].values,
        df[['h4']].values
    )

    try:
        is_error = not (check_is_fitted(model) is None)
    except:
        is_error = True
    assert not is_error


def test_train_model_on_fake_data(fake_dataset: pd.DataFrame):
    model = LogisticRegression(random_state=100500)
    train_model(
        model,
        fake_dataset[[col for col in fake_dataset.columns if col != 'target']].values,
        fake_dataset[['target']].values
    )

    try:
        is_error = not (check_is_fitted(model) is None)
    except:
        is_error = True
    assert not is_error