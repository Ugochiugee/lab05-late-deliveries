import numpy as np
import pytest

from latedeliveries.pipeline import (
    CATEGORY_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    split_data,
)


@pytest.mark.parametrize("column", ["order_id", "customer_rating", TARGET])
def test_feature_list_excludes_column(column):
    assert column not in FEATURES
    assert column not in NUMERIC_FEATURES
    assert column not in CATEGORY_FEATURES


def test_split_is_75_25_and_stratified(orders, split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 2250
    assert len(X_test) == 750
    overall = orders[TARGET].mean()
    assert abs(y_train.mean() - overall) < 0.005
    assert abs(y_test.mean() - overall) < 0.005


def test_split_is_repeatable(orders, split):
    _, X_test_again, _, _ = split_data(orders)
    assert list(split[1].index) == list(X_test_again.index)


def test_train_and_test_do_not_overlap(split):
    X_train, X_test, _, _ = split
    assert set(X_train.index).isdisjoint(X_test.index)


def test_predicted_probabilities_are_valid(fitted_pipeline, split):
    X_test = split[1]
    proba = fitted_pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)


def test_pipeline_handles_missing_value_and_unseen_category(fitted_pipeline, split):
    row = split[1].iloc[[0]].copy()
    row["prep_minutes"] = np.nan
    row["weather"] = "hail"
    proba = fitted_pipeline.predict_proba(row)[:, 1]
    assert proba.shape == (1,)
    assert 0 <= proba[0] <= 1
