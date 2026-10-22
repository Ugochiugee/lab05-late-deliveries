import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from latedeliveries.pipeline import CATEGORY_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]


def test_saved_model_predicts_the_same_after_reload(fitted_pipeline, split, tmp_path):
    X_test = split[1]
    before = fitted_pipeline.predict_proba(X_test)
    path = tmp_path / "model.joblib"
    joblib.dump(fitted_pipeline, path)
    after = joblib.load(path).predict_proba(X_test)
    np.testing.assert_array_equal(before, after)


def test_committed_model_info_matches_code():
    info = json.loads((ROOT / "models" / "model_info.json").read_text())
    assert info["numeric_features"] == NUMERIC_FEATURES
    assert info["category_features"] == CATEGORY_FEATURES
    assert info["sklearn_version"] == sklearn.__version__
    assert 0.05 <= info["threshold"] <= 0.95
