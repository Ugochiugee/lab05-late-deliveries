"""Metrics, the cost rule and the threshold choice."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

VOUCHER_COST = 4
MISSED_LATE_COST = 15
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def order_cost(y_true, flagged) -> int:
    """$4 for every flagged order plus $15 for every late order not flagged."""
    y_true = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    vouchers = int(flagged.sum())
    missed = int(((y_true == 1) & ~flagged).sum())
    return vouchers * VOUCHER_COST + missed * MISSED_LATE_COST


def cost_by_threshold(y_true, proba, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Total cost when every order with proba >= threshold is flagged."""
    proba = np.asarray(proba)
    rows = []
    for threshold in thresholds:
        flagged = proba >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "flagged": int(flagged.sum()),
                "cost": order_cost(y_true, flagged),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(y_true, proba, thresholds=THRESHOLDS) -> float:
    """The threshold with the lowest total cost. Ties go to the lower one."""
    table = cost_by_threshold(y_true, proba, thresholds)
    return float(table.loc[table["cost"].idxmin(), "threshold"])


def policy_costs(y_true, flagged) -> dict[str, int]:
    """Cost of flagging nobody, flagging everybody and the model's flags."""
    y_true = np.asarray(y_true)
    return {
        "flag_nobody": order_cost(y_true, np.zeros(len(y_true), dtype=bool)),
        "flag_everybody": order_cost(y_true, np.ones(len(y_true), dtype=bool)),
        "model": order_cost(y_true, flagged),
    }


def cv_splitter(seed: int) -> StratifiedKFold:
    """Five-fold stratified CV, shuffled with the project seed."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def compare_models(pipelines: dict, X, y, seed: int) -> pd.DataFrame:
    """Mean and standard deviation of each metric over five CV folds."""
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=cv_splitter(seed), scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_metrics(y_true, proba, threshold: float) -> dict:
    """Confusion matrix and metrics when flagging proba >= threshold."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "precision": round(float(precision_score(y_true, flagged)), 4),
        "recall": round(float(recall_score(y_true, flagged)), 4),
        "f1": round(float(f1_score(y_true, flagged)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
    }
