import numpy as np
import pytest

from latedeliveries.evaluation import (
    choose_threshold,
    cost_by_threshold,
    order_cost,
    policy_costs,
    threshold_metrics,
)


def test_cost_of_a_small_example():
    # Six orders, three late.
    y_true = [1, 1, 1, 0, 0, 0]
    flagged = [True, False, False, True, True, False]
    # Three vouchers (3 x 4) and two missed late orders (2 x 15) = 42
    assert order_cost(y_true, flagged) == 42


@pytest.mark.parametrize(
    ("threshold", "expected_cost"),
    [
        (0.05, 16),  # flags all four: 4 x 4
        (0.35, 23),  # flags the 0.4 and 0.6 orders: 2 x 4 + one late missed 15
        (0.5, 19),  # flags only the 0.6 order: 4 + 15
        (0.95, 30),  # flags nothing: two late missed, 2 x 15
    ],
)
def test_cost_by_threshold_hand_example(threshold, expected_cost):
    y_true = [1, 0, 1, 0]
    proba = [0.6, 0.4, 0.1, 0.2]
    table = cost_by_threshold(y_true, proba, thresholds=[threshold])
    assert table["cost"].iloc[0] == expected_cost


def test_choose_threshold_can_flag_nobody():
    # Only on-time orders get high scores; the one late order scores 0.1.
    # Flag all = 16, flag nobody = 15, anything in between is worse.
    y_true = [0, 0, 0, 1]
    proba = [0.9, 0.8, 0.7, 0.1]
    threshold = choose_threshold(y_true, proba)
    assert threshold == 0.95
    assert not np.any(np.asarray(proba) >= threshold)


def test_choose_threshold_can_flag_everybody():
    # Three late orders score low, one on-time order scores high.
    # Flag all = 16, flag nobody = 45, flag only the 0.9 order = 49.
    y_true = [1, 1, 1, 0]
    proba = [0.1, 0.1, 0.1, 0.9]
    threshold = choose_threshold(y_true, proba)
    assert threshold == 0.05
    assert np.all(np.asarray(proba) >= threshold)


def test_choose_threshold_in_the_middle():
    # Flagging the two orders at 0.5 and above catches both late ones for $8.
    y_true = [1, 1, 0, 0]
    proba = [0.8, 0.5, 0.3, 0.2]
    assert choose_threshold(y_true, proba) == 0.35


def test_policy_costs():
    y_true = [1, 0, 0, 1, 0]
    flagged = [True, False, False, False, False]
    assert policy_costs(y_true, flagged) == {
        "flag_nobody": 30,
        "flag_everybody": 20,
        "model": 19,
    }


def test_threshold_metrics_confusion_matrix():
    y_true = [1, 1, 0, 0]
    proba = [0.9, 0.2, 0.6, 0.1]
    result = threshold_metrics(y_true, proba, 0.5)
    assert result["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
