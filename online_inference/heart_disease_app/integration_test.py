import click
import numpy as np
import pandas as pd
import requests


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
    df['sex'] = np.random.randint(2, size=size)
    df['cp'] = np.random.randint(4, size=size)
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


@click.command("integration_test")
@click.argument("host")
@click.argument("port")
def integration_test(host: str, port: int):
    df = make_dataset(seed=100, size=100)
    df.drop('target', axis=1, inplace=True)
    for i in range(df.shape[0]):
        sample = [el for el in df.iloc[i].tolist()]

        request = {
            "data": [sample],
            "features": list(df.columns)
        }

        response = requests.get(
            f"http://{host}:{port}/predict",
            json=request,
        )

        assert response.status_code == 200
        assert "id" in response.json()[0]
        assert "target" in response.json()[0]


if __name__ == "__main__":
    integration_test()