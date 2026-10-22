import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_data


@pytest.fixture(scope="session")
def orders():
    return make_orders()


@pytest.fixture(scope="session")
def split(orders):
    return split_data(orders)


@pytest.fixture(scope="session")
def fitted_pipeline(split):
    X_train, _, y_train, _ = split
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    return pipe.fit(X_train, y_train)
