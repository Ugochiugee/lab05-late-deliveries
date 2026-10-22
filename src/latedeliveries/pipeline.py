"""Feature lists, the train/test split and the shared preprocessing pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TEST_SIZE = 0.25
TARGET = "late"

# Only columns whose value is known when the order is placed.
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "prep_minutes",
    "courier_months",
]
CATEGORY_FEATURES = ["weather", "day_type", "vehicle"]
FEATURES = NUMERIC_FEATURES + CATEGORY_FEATURES

# Left out on purpose. The README explains each one.
EXCLUDED_COLUMNS = ["order_id", "customer_rating"]


def split_data(
    frame: pd.DataFrame, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 75/25 split. Returns X_train, X_test, y_train, y_test."""
    X = frame[FEATURES]
    y = frame[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)


def build_preprocessor(
    numeric: list[str] | None = None, categories: list[str] | None = None
) -> ColumnTransformer:
    """Median fill and scale numbers, one-hot encode categories."""
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categories = CATEGORY_FEATURES if categories is None else categories
    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    category_steps = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        [
            ("num", numeric_steps, numeric),
            ("cat", category_steps, categories),
        ]
    )


def build_pipeline(
    model,
    numeric: list[str] | None = None,
    categories: list[str] | None = None,
) -> Pipeline:
    """Wrap any scikit-learn model in the shared preprocessing pipeline."""
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categories)),
            ("model", model),
        ]
    )
