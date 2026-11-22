"""Tests for the MP140 cleaning and training pipeline.

Uses small dummy data built inline instead of reading train.csv, since the
raw CSVs aren't guaranteed to be on GitHub.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.mp140.preprocess import clean_data
from src.mp140.train import FEATURE_COLS, build_pipeline

MODEL_PATH = "models/model.joblib"


def test_clean_data_fixes_messy_values():
    messy = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03"],
            "age_years": ["4.5", "4.5", "12", "-2"],
            "km_driven": ["30000", "30000", "50", "15000"],
            "engine_cc": ["150 cc", "150 cc", "119", "250cc"],
            "owners": ["-1", "-1", "1", "2"],
            "service_records": ["5", "5", "-2", "12"],
            "accidents": ["0", "0", "1", "-1"],
            "brand_tier": [" Premium ", " Premium ", "bud.", "MID"],
            "city_zone": ["Central", "Central", "North", "South"],
            "resale_thousand_rs": [75.0, 75.0, 0.0, 95.5],
        }
    )

    cleaned = clean_data(messy)

    # Duplicate rows removed
    assert cleaned.shape[0] < messy.shape[0]
    assert not cleaned.drop_duplicates().shape[0] < cleaned.shape[0]

    # Sentinel owner values (-1) no longer present
    assert (cleaned["owners"] > 0).all()

    # Unit strings like "150 cc" converted to pure numbers
    assert pd.api.types.is_numeric_dtype(cleaned["engine_cc"])

    # Spelling/abbreviation variants normalised to the real categories
    assert set(cleaned["brand_tier"].unique()).issubset({"budget", "mid", "premium"})

    # Zero/invalid resale prices dropped from the target
    assert (cleaned["resale_thousand_rs"] > 0).all()

    # Original input untouched
    assert messy["owners"].tolist() == ["-1", "-1", "1", "2"]


def test_build_pipeline_predicts_valid_values():
    sample = pd.DataFrame(
        {
            "age_years": [2.0, 5.0, 8.0, 12.0, 15.0, 20.0],
            "km_driven": [5000, 20000, 35000, 50000, 70000, 90000],
            "engine_cc": [110, 150, 200, 250, 350, 500],
            "owners": [1, 1, 2, 2, 3, 3],
            "service_records": [2, 4, 6, 8, 10, 12],
            "accidents": [0, 0, 1, 0, 1, 2],
            "brand_tier": ["budget", "mid", "mid", "premium", "premium", "budget"],
            "city_zone": ["North", "South", "Central", "North", "South", "Central"],
        }
    )
    target = pd.Series([80.0, 65.0, 55.0, 45.0, 35.0, 25.0])

    run_cfg = {"type": "random_forest", "n_estimators": 20, "max_depth": 5}
    pipeline = build_pipeline(run_cfg, random_state=42)
    pipeline.fit(sample[FEATURE_COLS], target)

    preds = pipeline.predict(sample[FEATURE_COLS])

    assert len(preds) == len(sample)
    assert not np.isnan(preds).any()
    assert (preds > 0).all()


def test_saved_model_predicts_for_sample_bike():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        pytest.skip(f"{MODEL_PATH} not present in this checkout")

    sample_bike = pd.DataFrame(
        {
            "age_years": [5.0],
            "km_driven": [35000],
            "engine_cc": [200],
            "owners": [2],
            "service_records": [6],
            "accidents": [0],
            "brand_tier": ["mid"],
            "city_zone": ["North"],
        }
    )

    prediction = model.predict(sample_bike[FEATURE_COLS])

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0
