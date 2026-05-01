# src/analysis/trends.py
#
# Year-over-year mental health trend analysis.
# Answers: How has youth mental health changed from 2015-2023?

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def load_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "unified_mental_health.csv"
    if not path.exists():
        raise FileNotFoundError("Run the data pipeline first: python -m src.processing.clean")
    return pd.read_csv(path)


def yearly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate year-over-year averages for key mental health metrics.
    Returns one row per year with aggregated statistics.
    """
    logger.info("Calculating yearly trends...")

    trends = df.groupby("year").agg(
        total_records=("year", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
        help_seeking_rate=("sought_help", "mean"),
        avg_sleep_hours=("sleep_hours", "mean"),
        avg_activity_days=("activity_days_per_week", "mean"),
    ).round(4).reset_index()

    # Calculate year-over-year change in risk score
    trends["risk_score_yoy_change"] = trends["avg_risk_score"].diff().round(4)

    logger.success(f"Yearly trends calculated: {len(trends)} years")
    return trends


def trends_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year trends split by country (USA vs Canada)."""
    logger.info("Calculating trends by country...")

    trends = df.groupby(["year", "country"]).agg(
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        help_seeking_rate=("sought_help", "mean"),
    ).round(4).reset_index()

    logger.success(f"Country trends calculated: {len(trends)} rows")
    return trends


def trends_by_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year trends split by age group."""
    logger.info("Calculating trends by age group...")

    # Focus on youth age groups
    youth_groups = ["12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-24"]
    youth_df = df[df["age_group"].isin(youth_groups)]

    trends = youth_df.groupby(["year", "age_group"]).agg(
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
    ).round(4).reset_index()

    logger.success(f"Age group trends calculated: {len(trends)} rows")
    return trends


def age_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall summary statistics by age group."""
    logger.info("Calculating age group summary...")

    summary = df.groupby("age_group").agg(
        count=("year", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
        help_seeking_rate=("sought_help", "mean"),
        avg_sleep=("sleep_hours", "mean"),
    ).round(4).reset_index()

    # Sort by age group logically
    age_order = ["12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
                 "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    summary["age_order"] = summary["age_group"].map(
        {age: i for i, age in enumerate(age_order)}
    )
    summary = summary.sort_values("age_order").drop("age_order", axis=1)

    logger.success(f"Age group summary: {len(summary)} groups")
    return summary


def gender_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Mental health trends by gender over time."""
    logger.info("Calculating gender trends...")

    trends = df.groupby(["year", "gender"]).agg(
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
    ).round(4).reset_index()

    logger.success(f"Gender trends calculated: {len(trends)} rows")
    return trends


def run_all_trends(df: pd.DataFrame) -> dict:
    """Run all trend analyses and return as a dict of DataFrames."""
    return {
        "yearly": yearly_trends(df),
        "by_country": trends_by_country(df),
        "by_age_group": trends_by_age_group(df),
        "age_summary": age_group_summary(df),
        "by_gender": gender_trends(df),
    }


if __name__ == "__main__":
    df = load_data()
    results = run_all_trends(df)

    print("\n── Yearly Trends ──")
    print(results["yearly"].to_string(index=False))

    print("\n── Age Group Summary ──")
    print(results["age_summary"].to_string(index=False))

    print("\n── Gender Breakdown ──")
    print(results["by_gender"].groupby("gender").agg(
        avg_risk=("avg_risk_score", "mean")
    ).round(3))
