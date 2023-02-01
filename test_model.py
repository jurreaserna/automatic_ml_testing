import pickle
from sklearn.datasets import make_regression
from pathlib import Path
import pytest


@pytest.fixture()
def dataset():
    return make_regression(1000, n_features=11)


def test_model_exists():
    path = Path("models/model.pkl")
    assert path.is_file()


def test_prediction(dataset):
    X_test, y = dataset
    path = Path("models/model.pkl")
    with path.open("rb") as file:
        model = pickle.load(file)
    y_hat = model.predict(X_test)
    assert y_hat.any()
