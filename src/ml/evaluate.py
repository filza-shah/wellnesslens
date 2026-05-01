# src/ml/evaluate.py
#
# Model evaluation and inference.
# Run after training to get full evaluation report.
# Also provides predict() function used by the dashboard.
#
# HOW TO RUN:
#   python -m src.ml.evaluate

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

from src.ml.features import prepare_features, encode_categoricals, create_interaction_features, get_feature_columns

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
MODEL_PATH    = MODELS_DIR / "risk_model.joblib"


def load_model():
    """Load the trained model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python -m src.ml.train")
    return joblib.load(MODEL_PATH)


def full_evaluation():
    """Complete evaluation report on the trained model."""
    logger.info("Loading model and data...")
    model = load_model()
    df = pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")
    X, y = prepare_features(df)

    print("=" * 60)
    print("WellnessLens Risk Model — Evaluation Report")
    print("=" * 60)

    # Hold-out evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n── Hold-out Test Set ({len(X_test):,} samples) ──")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n── Confusion Matrix ──")
    print(f"{'':15} {'Pred Low':>10} {'Pred High':>10}")
    print(f"{'Actual Low':15} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"{'Actual High':15} {cm[1][0]:>10} {cm[1][1]:>10}")

    # 5-fold CV
    print(f"\n── 5-Fold Cross-Validation ──")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    print(f"F1 scores: {[round(s, 4) for s in cv_scores]}")
    print(f"Mean F1:   {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

    # Example predictions
    print(f"\n── Example Predictions ──")
    examples = [
        {"age_group": "18-24", "gender": "Female", "sleep_hours": 4.5,
         "activity_days_per_week": 0, "anxiety_diagnosis": 1,
         "depression_diagnosis": 1, "sought_help": 0,
         "social_support": "Never", "country": "USA",
         "year": 2023, "source": "CDC_BRFSS"},
        {"age_group": "25-34", "gender": "Male", "sleep_hours": 8.0,
         "activity_days_per_week": 5, "anxiety_diagnosis": 0,
         "depression_diagnosis": 0, "sought_help": 0,
         "social_support": "Always", "country": "Canada",
         "year": 2023, "source": "CAMH_OSDUHS"},
        {"age_group": "16-17", "gender": "Female", "sleep_hours": 5.5,
         "activity_days_per_week": 2, "anxiety_diagnosis": 1,
         "depression_diagnosis": 0, "sought_help": 1,
         "social_support": "Sometimes", "country": "Canada",
         "year": 2023, "source": "CAMH_OSDUHS"},
    ]

    for ex in examples:
        result = predict_single(model, ex)
        print(f"\n  Profile: {ex['age_group']} {ex['gender']}, "
              f"sleep={ex['sleep_hours']}h, anxiety={ex['anxiety_diagnosis']}")
        print(f"  → {'HIGH RISK' if result['high_risk'] else 'LOW RISK'} "
              f"(confidence: {result['probability']:.1%})")


def predict_single(model, profile: dict) -> dict:
    """
    Predict risk for a single individual profile.
    Used by the dashboard for the prediction interface.

    Args:
        profile: dict with same columns as the dataset

    Returns:
        dict with high_risk (bool) and probability (float)
    """
    df = pd.DataFrame([profile])
    df = encode_categoricals(df)
    df = create_interaction_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].fillna(0)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "high_risk": bool(prediction),
        "probability": round(float(probability), 4),
        "risk_level": (
            "Critical" if probability >= 0.75 else
            "High" if probability >= 0.50 else
            "Moderate" if probability >= 0.30 else
            "Low"
        ),
    }


if __name__ == "__main__":
    full_evaluation()
