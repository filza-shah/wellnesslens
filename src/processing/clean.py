# src/processing/clean.py
#
# Data cleaning and merging pipeline.
# Takes raw CDC + Canada datasets and produces a single
# unified, analysis-ready dataset.
#
# HOW TO RUN:
#   python -m src.processing.clean

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

RAW_DIR    = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_cdc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the CDC dataset."""
    logger.info("Cleaning CDC dataset...")

    # Standardize column names
    df = df.copy()

    # Create mental health risk score (0-100)
    df["mental_health_risk_score"] = (
        (df["poor_mental_health_days"] / 30 * 40) +
        (df["anxiety_diagnosis"] * 25) +
        (df["depression_diagnosis"] * 25) +
        ((8 - df["sleep_hours"].clip(4, 8)) / 4 * 10)
    ).clip(0, 100).round(2)

    # Binary risk label for ML
    df["high_risk"] = (df["mental_health_risk_score"] >= 50).astype(int)

    # Standardize age groups
    df["age_group_standard"] = df["age_group"]

    # Add dataset source
    df["source"] = "CDC_BRFSS"

    # Select and rename columns for unified schema
    result = df[[
        "year", "country", "age_group_standard", "gender",
        "sleep_hours", "physical_activity_days",
        "anxiety_diagnosis", "depression_diagnosis",
        "sought_treatment", "mental_health_risk_score", "high_risk",
        "social_support", "source"
    ]].rename(columns={
        "age_group_standard": "age_group",
        "physical_activity_days": "activity_days_per_week",
        "sought_treatment": "sought_help",
    })

    logger.success(f"CDC cleaned: {len(result):,} rows")
    return result


def clean_canada(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the Canada/CAMH dataset."""
    logger.info("Cleaning Canada dataset...")

    df = df.copy()

    # Map self-rated mental health to numeric
    mh_map = {"Excellent": 0, "Very Good": 10, "Good": 25, "Fair": 50, "Poor": 80}
    df["mh_base_score"] = df["self_rated_mental_health"].map(mh_map).fillna(25)

    # Create mental health risk score
    df["mental_health_risk_score"] = (
        (df["mh_base_score"] * 0.4) +
        (df["high_psychological_distress"] * 25) +
        (df["anxiety_symptoms"] * 20) +
        (df["depression_symptoms"] * 15)
    ).clip(0, 100).round(2)

    df["high_risk"] = (df["mental_health_risk_score"] >= 50).astype(int)

    # Map sleep hours to match CDC column
    df["sleep_hours"] = df["sleep_hours_school_night"]

    # Map activity
    df["activity_days_per_week"] = df["physical_activity_days_per_week"]

    # Map anxiety/depression to binary
    df["anxiety_diagnosis"] = df["anxiety_symptoms"]
    df["depression_diagnosis"] = df["depression_symptoms"]

    # Social support proxy from close friends
    friend_map = {0: "Never", 1: "Rarely", 2: "Sometimes", 3: "Usually", 4: "Always", 5: "Always"}
    df["social_support"] = df["close_friends"].map(friend_map).fillna("Sometimes")

    df["source"] = "CAMH_OSDUHS"

    result = df[[
        "year", "country", "age_group", "gender",
        "sleep_hours", "activity_days_per_week",
        "anxiety_diagnosis", "depression_diagnosis",
        "sought_help", "mental_health_risk_score", "high_risk",
        "social_support", "source"
    ]]

    logger.success(f"Canada cleaned: {len(result):,} rows")
    return result


def merge_datasets(cdc_df: pd.DataFrame, canada_df: pd.DataFrame) -> pd.DataFrame:
    """Merge CDC and Canada datasets into unified schema."""
    logger.info("Merging datasets...")

    # Ensure same columns
    assert set(cdc_df.columns) == set(canada_df.columns), \
        f"Column mismatch: {set(cdc_df.columns) ^ set(canada_df.columns)}"

    merged = pd.concat([cdc_df, canada_df], ignore_index=True)

    # Final cleaning
    merged = merged.dropna(subset=["mental_health_risk_score", "age_group"])
    merged["year"] = merged["year"].astype(int)

    logger.success(f"Merged dataset: {len(merged):,} rows from {merged['source'].nunique()} sources")
    logger.info(f"  Year range: {merged['year'].min()} - {merged['year'].max()}")
    logger.info(f"  Countries: {merged['country'].unique().tolist()}")
    logger.info(f"  Age groups: {sorted(merged['age_group'].unique().tolist())}")

    return merged


def run_pipeline() -> pd.DataFrame:
    """Full data processing pipeline."""
    logger.info("=" * 50)
    logger.info("WellnessLens Data Processing Pipeline")
    logger.info("=" * 50)

    # Load raw data
    cdc_raw    = pd.read_csv(RAW_DIR / "cdc_mental_health.csv")
    canada_raw = pd.read_csv(RAW_DIR / "canada_youth_mental_health.csv")

    logger.info(f"Loaded CDC: {len(cdc_raw):,} rows")
    logger.info(f"Loaded Canada: {len(canada_raw):,} rows")

    # Clean each dataset
    cdc_clean    = clean_cdc(cdc_raw)
    canada_clean = clean_canada(canada_raw)

    # Merge
    unified = merge_datasets(cdc_clean, canada_clean)

    # Save
    output_path = PROCESSED_DIR / "unified_mental_health.csv"
    unified.to_csv(output_path, index=False)
    logger.success(f"Saved unified dataset: {output_path}")

    # Save summary stats
    summary = unified.describe().round(3)
    summary.to_csv(PROCESSED_DIR / "summary_stats.csv")
    logger.success("Saved summary statistics")

    return unified


if __name__ == "__main__":
    df = run_pipeline()
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nHigh risk rate: {df['high_risk'].mean():.1%}")
    print(f"\nSample:\n{df.head()}")
