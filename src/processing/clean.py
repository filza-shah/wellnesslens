# src/processing/clean.py
#
# Data cleaning and merging pipeline.
# Handles CDC (USA), CAMH (Canada), and Pakistan datasets.
#
# HOW TO RUN:
#   python -m src.processing.clean

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_cdc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the CDC dataset."""
    logger.info("Cleaning CDC dataset...")
    df = df.copy()

    df["mental_health_risk_score"] = (
        (df["poor_mental_health_days"] / 30 * 40) +
        (df["anxiety_diagnosis"] * 25) +
        (df["depression_diagnosis"] * 25) +
        ((8 - df["sleep_hours"].clip(4, 8)) / 4 * 10)
    ).clip(0, 100).round(2)

    df["high_risk"] = (df["mental_health_risk_score"] >= 50).astype(int)

    result = df.rename(columns={
        "age_group": "age_group",
        "physical_activity_days": "activity_days_per_week",
        "sought_treatment": "sought_help",
    })[[
        "year", "country", "age_group", "gender",
        "sleep_hours", "activity_days_per_week",
        "anxiety_diagnosis", "depression_diagnosis",
        "sought_help", "mental_health_risk_score", "high_risk",
        "social_support",
    ]]
    result["source"] = "CDC_BRFSS"

    logger.success(f"CDC cleaned: {len(result):,} rows")
    return result


def clean_canada(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Canada dataset."""
    logger.info("Cleaning Canada dataset...")
    df = df.copy()

    mh_map = {"Excellent": 0, "Very Good": 10, "Good": 25, "Fair": 50, "Poor": 80}
    df["mh_base_score"] = df["self_rated_mental_health"].map(mh_map).fillna(25)

    df["mental_health_risk_score"] = (
        (df["mh_base_score"] * 0.4) +
        (df["high_psychological_distress"] * 25) +
        (df["anxiety_symptoms"] * 20) +
        (df["depression_symptoms"] * 15)
    ).clip(0, 100).round(2)

    df["high_risk"] = (df["mental_health_risk_score"] >= 50).astype(int)

    friend_map = {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Usually", 4: "Always", 5: "Always"}
    df["social_support"] = df["close_friends"].map(friend_map).fillna("Sometimes")

    result = df.rename(columns={
        "sleep_hours_school_night": "sleep_hours",
        "physical_activity_days_per_week": "activity_days_per_week",
        "anxiety_symptoms": "anxiety_diagnosis",
        "depression_symptoms": "depression_diagnosis",
    })[[
        "year", "country", "age_group", "gender",
        "sleep_hours", "activity_days_per_week",
        "anxiety_diagnosis", "depression_diagnosis",
        "sought_help", "mental_health_risk_score", "high_risk",
        "social_support",
    ]]
    result["source"] = "CAMH_OSDUHS"

    logger.success(f"Canada cleaned: {len(result):,} rows")
    return result


def clean_pakistan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the Pakistan dataset.

    Pakistan-specific adjustments:
    - Higher anxiety baseline from published research
    - Lower help-seeking rates due to cultural stigma
    - Risk score calibrated to Pakistan-specific distributions
    """
    logger.info("Cleaning Pakistan dataset...")
    df = df.copy()

    mh_map = {"Excellent": 0, "Very Good": 10, "Good": 25, "Fair": 50, "Poor": 80}
    df["mh_base_score"] = df["self_rated_mental_health"].map(mh_map).fillna(30)

    df["mental_health_risk_score"] = (
        (df["mh_base_score"] * 0.35) +
        (df["anxiety_symptoms"] * 30) +
        (df["depression_symptoms"] * 25) +
        ((7 - df["sleep_hours_school_night"].clip(4, 7)) / 3 * 10)
    ).clip(0, 100).round(2)

    df["high_risk"] = (df["mental_health_risk_score"] >= 50).astype(int)

    df["activity_days_per_week"] = df["physical_activity_days_per_week"]

    result = df.rename(columns={
        "sleep_hours_school_night": "sleep_hours",
        "anxiety_symptoms": "anxiety_diagnosis",
        "depression_symptoms": "depression_diagnosis",
    })[[
        "year", "country", "age_group", "gender",
        "sleep_hours", "activity_days_per_week",
        "anxiety_diagnosis", "depression_diagnosis",
        "sought_help", "mental_health_risk_score", "high_risk",
        "social_support",
    ]]
    result["source"] = "Pakistan_NPMS_GSHS"

    logger.success(f"Pakistan cleaned: {len(result):,} rows")
    return result


def merge_datasets(*dfs) -> pd.DataFrame:
    """Merge all datasets into unified schema."""
    logger.info("Merging datasets...")

    merged = pd.concat(list(dfs), ignore_index=True)
    merged = merged.dropna(subset=["mental_health_risk_score", "age_group"])
    merged["year"] = merged["year"].astype(int)

    logger.success(f"Merged: {len(merged):,} rows from {merged['country'].nunique()} countries")
    logger.info(f"  Countries: {sorted(merged['country'].unique().tolist())}")
    logger.info(f"  Year range: {merged['year'].min()} – {merged['year'].max()}")
    return merged


def run_pipeline() -> pd.DataFrame:
    """Full data processing pipeline."""
    logger.info("=" * 50)
    logger.info("WellnessLens Data Processing Pipeline")
    logger.info("=" * 50)

    cdc_raw      = pd.read_csv(RAW_DIR / "cdc_mental_health.csv")
    canada_raw   = pd.read_csv(RAW_DIR / "canada_youth_mental_health.csv")
    pakistan_raw = pd.read_csv(RAW_DIR / "pakistan_youth_mental_health.csv")

    logger.info(f"Loaded CDC: {len(cdc_raw):,} rows")
    logger.info(f"Loaded Canada: {len(canada_raw):,} rows")
    logger.info(f"Loaded Pakistan: {len(pakistan_raw):,} rows")

    cdc_clean      = clean_cdc(cdc_raw)
    canada_clean   = clean_canada(canada_raw)
    pakistan_clean = clean_pakistan(pakistan_raw)

    unified = merge_datasets(cdc_clean, canada_clean, pakistan_clean)

    output_path = PROCESSED_DIR / "unified_mental_health.csv"
    unified.to_csv(output_path, index=False)
    unified.describe().round(3).to_csv(PROCESSED_DIR / "summary_stats.csv")

    logger.success(f"Saved: {output_path}")
    return unified


if __name__ == "__main__":
    df = run_pipeline()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Countries: {df['country'].unique().tolist()}")
    print(f"High risk rate overall: {df['high_risk'].mean():.1%}")
    print(f"\nHigh risk by country:")
    print(df.groupby("country")["high_risk"].mean().round(3))
