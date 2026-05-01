# src/ml/features.py
#
# Feature engineering pipeline.
# Transforms raw dataset columns into ML-ready features.
#
# WHY FEATURE ENGINEERING MATTERS:
# Raw data rarely goes straight into an ML model.
# We need to:
# 1. Encode categorical variables (strings → numbers)
# 2. Create interaction features (e.g. sleep * activity)
# 3. Handle missing values
# 4. Scale numerical features

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns as numbers."""
    df = df.copy()

    # Gender encoding
    gender_map = {"Male": 0, "Female": 1, "Non-binary": 2, "Non-binary/Other": 2}
    df["gender_encoded"] = df["gender"].map(gender_map).fillna(2)

    # Social support encoding (ordinal — order matters)
    support_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Usually": 3, "Always": 4}
    df["social_support_encoded"] = df["social_support"].map(support_map).fillna(2)

    # Country encoding
    df["country_encoded"] = (df["country"] == "Canada").astype(int)

    # Age group encoding — convert to numeric midpoint
    age_midpoints = {
        "12-13": 12.5, "13-14": 13.5, "14-15": 14.5,
        "15-16": 15.5, "16-17": 16.5, "17-18": 17.5,
        "18-24": 21.0, "25-34": 29.5, "35-44": 39.5,
        "45-54": 49.5, "55-64": 59.5, "65+": 70.0,
    }
    df["age_numeric"] = df["age_group"].map(age_midpoints).fillna(30.0)

    # Youth flag (under 25)
    df["is_youth"] = (df["age_numeric"] < 25).astype(int)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features that capture combined effects.

    Research shows that combinations of risk factors are more
    predictive than individual factors alone.
    """
    df = df.copy()

    # Sleep deprivation severity
    df["sleep_deprived"] = (df["sleep_hours"] < 6).astype(int)
    df["severely_sleep_deprived"] = (df["sleep_hours"] < 5).astype(int)

    # Low activity flag
    df["low_activity"] = (df["activity_days_per_week"] <= 1).astype(int)

    # Combined risk: sleep deprived AND low activity
    df["lifestyle_risk"] = df["sleep_deprived"] * df["low_activity"]

    # Social isolation proxy
    df["socially_isolated"] = (df["social_support_encoded"] <= 1).astype(int)

    # Multiple diagnoses
    df["comorbid"] = (
        (df["anxiety_diagnosis"] == 1) & (df["depression_diagnosis"] == 1)
    ).astype(int)

    # Youth with poor sleep — particularly high risk
    df["youth_sleep_deprived"] = df["is_youth"] * df["sleep_deprived"]

    return df


def get_feature_columns() -> list[str]:
    """Returns the list of feature columns used for training."""
    return [
        # Demographics
        "age_numeric",
        "gender_encoded",
        "country_encoded",
        "is_youth",

        # Lifestyle
        "sleep_hours",
        "sleep_deprived",
        "severely_sleep_deprived",
        "activity_days_per_week",
        "low_activity",

        # Mental health history
        "anxiety_diagnosis",
        "depression_diagnosis",
        "sought_help",
        "comorbid",

        # Social
        "social_support_encoded",
        "socially_isolated",

        # Interaction features
        "lifestyle_risk",
        "youth_sleep_deprived",
    ]


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature preparation pipeline.
    Returns (X, y) ready for model training.
    """
    logger.info("Preparing features for ML model...")

    # Encode categoricals
    df = encode_categoricals(df)

    # Create interaction features
    df = create_interaction_features(df)

    # Get feature columns
    feature_cols = get_feature_columns()

    # Select features and target
    X = df[feature_cols].copy()
    y = df["high_risk"].copy()

    # Handle any remaining NaNs
    X = X.fillna(X.median())

    logger.success(f"Features prepared: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Class balance: {y.mean():.1%} high risk")

    return X, y


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")
    X, y = prepare_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeatures:\n{X.dtypes}")
