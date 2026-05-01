# tests/test_pipeline.py

import pytest
import pandas as pd
from pathlib import Path


def test_processed_data_exists():
    """Processed data file should exist after running the pipeline."""
    path = Path("data/processed/unified_mental_health.csv")
    assert path.exists(), "Run: python -m src.processing.clean"


def test_processed_data_shape():
    """Dataset should have expected number of rows and columns."""
    path = Path("data/processed/unified_mental_health.csv")
    if not path.exists():
        pytest.skip("Data not generated yet")

    df = pd.read_csv(path)
    assert len(df) > 50000, f"Expected >50k rows, got {len(df)}"
    assert "mental_health_risk_score" in df.columns
    assert "high_risk" in df.columns
    assert "age_group" in df.columns


def test_no_nulls_in_key_columns():
    """Key columns should not have null values."""
    path = Path("data/processed/unified_mental_health.csv")
    if not path.exists():
        pytest.skip("Data not generated yet")

    df = pd.read_csv(path)
    key_cols = ["mental_health_risk_score", "high_risk", "year", "country"]
    for col in key_cols:
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"Column '{col}' has {null_count} null values"


def test_risk_score_range():
    """Risk scores should be between 0 and 100."""
    path = Path("data/processed/unified_mental_health.csv")
    if not path.exists():
        pytest.skip("Data not generated yet")

    df = pd.read_csv(path)
    assert df["mental_health_risk_score"].min() >= 0
    assert df["mental_health_risk_score"].max() <= 100


def test_model_exists():
    """Trained model file should exist."""
    path = Path("data/models/risk_model.joblib")
    if not path.exists():
        pytest.skip("Model not trained yet")
    assert path.exists()


def test_feature_engineering():
    """Feature engineering should produce expected columns."""
    path = Path("data/processed/unified_mental_health.csv")
    if not path.exists():
        pytest.skip("Data not generated yet")

    df = pd.read_csv(path)
    from src.ml.features import prepare_features
    X, y = prepare_features(df)

    assert X.shape[1] == 17, f"Expected 17 features, got {X.shape[1]}"
    assert len(y) == len(df)
    assert y.isin([0, 1]).all(), "Target should be binary"


def test_model_prediction():
    """Model should produce valid predictions."""
    import joblib
    model_path = Path("data/models/risk_model.joblib")
    if not model_path.exists():
        pytest.skip("Model not trained yet")

    model = joblib.load(model_path)
    from src.ml.evaluate import predict_single

    result = predict_single(model, {
        "age_group": "18-24", "gender": "Female",
        "sleep_hours": 5.0, "activity_days_per_week": 1,
        "anxiety_diagnosis": 1, "depression_diagnosis": 0,
        "sought_help": 0, "social_support": "Rarely",
        "country": "USA", "year": 2023, "source": "CDC_BRFSS",
    })

    assert "high_risk" in result
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
    assert result["risk_level"] in ["Low", "Moderate", "High", "Critical"]
