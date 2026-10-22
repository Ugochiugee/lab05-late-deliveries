"""Run the whole lab: python -m latedeliveries.train"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    compare_models,
    cost_by_threshold,
    cv_splitter,
    policy_costs,
    threshold_metrics,
)
from latedeliveries.pipeline import (
    CATEGORY_FEATURES,
    EXCLUDED_COLUMNS,
    NUMERIC_FEATURES,
    SEED,
    TARGET,
    build_pipeline,
    split_data,
)

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
CHOSEN = "logistic_regression"
PARAM_GRID = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__class_weight": [None, "balanced"],
}
GRID_SCORING = "roc_auc"


def candidate_models(seed: int = SEED) -> dict:
    """The baseline and the real candidates, each inside the shared pipeline."""
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }
    return {name: build_pipeline(model) for name, model in models.items()}


def oof_proba(pipe, X, y, seed: int = SEED) -> np.ndarray:
    """Out-of-fold probability of late for every training order."""
    return cross_val_predict(pipe, X, y, cv=cv_splitter(seed), method="predict_proba")[
        :, 1
    ]


def lowest_cv_cost(pipelines: dict, X, y, seed: int = SEED) -> dict:
    """Best achievable out-of-fold cost per model, over the threshold grid."""
    result = {}
    for name, pipe in pipelines.items():
        table = cost_by_threshold(y, oof_proba(pipe, X, y, seed))
        best = table.loc[table["cost"].idxmin()]
        result[name] = {
            "lowest_cost": int(best["cost"]),
            "at_threshold": float(best["threshold"]),
        }
    return result


def leakage_check(train_frame, seed: int = SEED):
    """Re-run the comparison with customer_rating added back in."""
    X = train_frame[NUMERIC_FEATURES + ["customer_rating"] + CATEGORY_FEATURES]
    y = train_frame[TARGET]
    pipe = build_pipeline(
        LogisticRegression(max_iter=1000),
        numeric=NUMERIC_FEATURES + ["customer_rating"],
    )
    return compare_models({"logistic_regression_with_rating": pipe}, X, y, seed)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # 1 Split
    orders = make_orders()
    X_train, X_test, y_train, y_test = split_data(orders)
    print(f"Late rate: {orders[TARGET].mean():.3f} over {len(orders)} orders")

    # 2 Compare, training data only
    pipelines = candidate_models()
    cv_results = compare_models(pipelines, X_train, y_train, SEED)
    costs = lowest_cv_cost(pipelines, X_train, y_train)
    cv_results["lowest_cv_cost"] = cv_results["model"].map(
        lambda m: costs[m]["lowest_cost"]
    )
    cv_results["at_threshold"] = cv_results["model"].map(
        lambda m: costs[m]["at_threshold"]
    )
    cv_results.to_csv(REPORTS / "cv_results.csv", index=False)
    print(cv_results.to_string(index=False))

    # Leakage demonstration, training data only
    rating_check = leakage_check(orders.loc[X_train.index])
    rating_check.to_csv(REPORTS / "leakage_check.csv", index=False)

    # 3 Tune the chosen model
    search = GridSearchCV(
        build_pipeline(LogisticRegression(max_iter=1000)),
        PARAM_GRID,
        scoring=GRID_SCORING,
        cv=cv_splitter(SEED),
    )
    search.fit(X_train, y_train)
    untuned = cv_results.set_index("model").loc[CHOSEN]
    gain = float(search.best_score_ - untuned["roc_auc_mean"])
    tuning = {
        "scoring": GRID_SCORING,
        "best_params": {k: v for k, v in search.best_params_.items()},
        "best_cv_roc_auc": round(float(search.best_score_), 4),
        "untuned_cv_roc_auc": float(untuned["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": float(untuned["roc_auc_std"]),
        "gain": round(gain, 4),
        "gain_beats_fold_spread": bool(gain > untuned["roc_auc_std"]),
    }
    print("Tuning:", tuning)

    # 4 Threshold from out-of-fold probabilities, training data only
    final = search.best_estimator_
    train_proba = oof_proba(final, X_train, y_train)
    threshold_table = cost_by_threshold(y_train, train_proba)
    threshold_table.to_csv(REPORTS / "threshold_costs.csv", index=False)
    threshold = float(
        threshold_table.loc[threshold_table["cost"].idxmin(), "threshold"]
    )
    print(f"Chosen threshold: {threshold}")

    # 5 Test once
    final.fit(X_train, y_train)
    test_proba = final.predict_proba(X_test)[:, 1]
    flagged = test_proba >= threshold
    test_result = threshold_metrics(y_test, test_proba, threshold)
    test_result["n_test"] = int(len(y_test))
    test_result["late_in_test"] = int(y_test.sum())
    test_result["flagged"] = int(flagged.sum())
    test_result["policy_costs"] = policy_costs(y_test, flagged)
    (REPORTS / "metrics.json").write_text(json.dumps(test_result, indent=2) + "\n")
    print("Test:", json.dumps(test_result, indent=2))

    # 6 Save
    joblib.dump(final, MODELS / "late_orders.joblib")
    info = {
        "sklearn_version": sklearn.__version__,
        "seed": SEED,
        "model": CHOSEN,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "category_features": CATEGORY_FEATURES,
        "excluded_columns": EXCLUDED_COLUMNS,
        "best_params": tuning["best_params"],
        "tuning": tuning,
        "test_results": test_result,
    }
    (MODELS / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Saved model and reports.")


if __name__ == "__main__":
    main()
