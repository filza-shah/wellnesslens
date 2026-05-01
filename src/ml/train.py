# src/ml/train.py
#
# Trains a mental health risk prediction model.
#
# ARCHITECTURE:
# We train 3 models and pick the best one:
# 1. Logistic Regression — fast, interpretable, good baseline
# 2. Random Forest — handles non-linear patterns, robust
# 3. Gradient Boosting — usually best performance
#
# Final model is saved to data/models/risk_model.joblib
#
# HOW TO RUN:
#   python -m src.ml.train

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from loguru import logger

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.features import prepare_features

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_models() -> dict:
    """Define the candidate models to evaluate."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ))
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            ))
        ]),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":    round(accuracy_score(y_test, y_pred), 4),
        "f1_macro":    round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "roc_auc":     round(roc_auc_score(y_test, y_proba), 4),
        "report":      classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """Extract feature importances from the trained model."""
    try:
        # Random Forest / Gradient Boosting
        clf = model.named_steps.get("clf")
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            # Logistic Regression
            importances = np.abs(clf.coef_[0])
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def train(save_path: str = "data/models/risk_model.joblib") -> dict:
    """
    Full training pipeline:
    1. Load and prepare features
    2. Train/test split
    3. Train 3 candidate models
    4. 5-fold cross-validation
    5. Select best model
    6. Save to disk
    7. Return metadata
    """
    logger.info("=" * 60)
    logger.info("WellnessLens Risk Prediction Model Training")
    logger.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    df = pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")
    X, y = prepare_features(df)
    feature_names = X.columns.tolist()

    logger.info(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"High risk rate: {y.mean():.1%}")

    # ── 2. Train/test split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── 3. Train all models ───────────────────────────────────────
    models = build_models()
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)

        # Hold-out test metrics
        metrics = evaluate_model(model, X_test, y_test)

        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"]  = round(cv_scores.std(), 4)

        results[name] = {"model": model, "metrics": metrics}

        logger.info(f"  {name}: F1={metrics['f1_macro']:.4f}, AUC={metrics['roc_auc']:.4f}, CV={metrics['cv_f1_mean']:.4f}±{metrics['cv_f1_std']:.4f}")

    # ── 4. Select best model ──────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    logger.success(f"Best model: {best_name} (AUC={best_metrics['roc_auc']:.4f})")

    # ── 5. Print results ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"{'='*60}")
    print(f"\nAccuracy:    {best_metrics['accuracy']:.4f}")
    print(f"F1 (macro):  {best_metrics['f1_macro']:.4f}")
    print(f"ROC-AUC:     {best_metrics['roc_auc']:.4f}")
    print(f"CV F1:       {best_metrics['cv_f1_mean']:.4f} ± {best_metrics['cv_f1_std']:.4f}")
    print(f"\n{best_metrics['report']}")

    # Feature importance
    importance_df = get_feature_importance(best_model, feature_names)
    if not importance_df.empty:
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    # ── 6. Save model ─────────────────────────────────────────────
    joblib.dump(best_model, save_path)
    logger.success(f"Model saved: {save_path} ({os.path.getsize(save_path)/1024:.1f} KB)")

    # ── 7. Save metadata ──────────────────────────────────────────
    all_metrics = {
        name: {k: v for k, v in res["metrics"].items() if k not in ["report", "confusion_matrix"]}
        for name, res in results.items()
    }

    metadata = {
        "model_name": "WellnessLens Risk Prediction Model v1.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_model": best_name,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": feature_names,
        "n_features": len(feature_names),
        "target": "high_risk (binary: 0=low risk, 1=high risk)",
        "class_balance": {"high_risk_rate": round(y.mean(), 4)},
        "best_model_metrics": {
            k: v for k, v in best_metrics.items()
            if k not in ["report", "confusion_matrix"]
        },
        "all_model_metrics": all_metrics,
        "feature_importance": importance_df.to_dict("records") if not importance_df.empty else [],
    }

    metadata_path = save_path.replace(".joblib", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.success(f"Metadata saved: {metadata_path}")

    return metadata


if __name__ == "__main__":
    metadata = train()
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best model:  {metadata['best_model']}")
    print(f"Accuracy:    {metadata['best_model_metrics']['accuracy']*100:.1f}%")
    print(f"F1 (macro):  {metadata['best_model_metrics']['f1_macro']:.4f}")
    print(f"ROC-AUC:     {metadata['best_model_metrics']['roc_auc']:.4f}")
