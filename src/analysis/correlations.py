# src/analysis/correlations.py
#
# Correlation analysis between lifestyle factors and mental health outcomes.
# Answers: Does sleep, exercise, and social support actually matter?

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")


def sleep_correlation(df: pd.DataFrame) -> dict:
    """
    Analyse the relationship between sleep hours and mental health risk.

    Uses Pearson correlation + sleep bucket analysis to show
    at what point sleep deprivation becomes critical.
    """
    logger.info("Analysing sleep correlation...")

    # Pearson correlation
    corr, p_value = stats.pearsonr(
        df["sleep_hours"].dropna(),
        df["mental_health_risk_score"].dropna()
    )

    # Sleep buckets — show risk at each sleep level
    df = df.copy()
    df["sleep_bucket"] = pd.cut(
        df["sleep_hours"],
        bins=[0, 5, 6, 7, 8, 9, 12],
        labels=["<5h", "5-6h", "6-7h", "7-8h", "8-9h", "9h+"]
    )

    sleep_analysis = df.groupby("sleep_bucket", observed=True).agg(
        count=("sleep_hours", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
    ).round(4).reset_index()

    result = {
        "pearson_correlation": round(corr, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "interpretation": (
            f"Sleep hours and mental health risk have a "
            f"{'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'} "
            f"{'negative' if corr < 0 else 'positive'} correlation (r={corr:.3f}, p={p_value:.4f})"
        ),
        "by_bucket": sleep_analysis,
    }

    logger.success(f"Sleep correlation: r={corr:.3f}, p={p_value:.4f}")
    return result


def activity_correlation(df: pd.DataFrame) -> dict:
    """
    Analyse the relationship between physical activity and mental health risk.
    """
    logger.info("Analysing physical activity correlation...")

    corr, p_value = stats.pearsonr(
        df["activity_days_per_week"],
        df["mental_health_risk_score"]
    )

    # Activity buckets
    df = df.copy()
    df["activity_bucket"] = pd.cut(
        df["activity_days_per_week"],
        bins=[-1, 0, 1, 3, 5, 7],
        labels=["None", "1 day", "2-3 days", "4-5 days", "6-7 days"]
    )

    activity_analysis = df.groupby("activity_bucket", observed=True).agg(
        count=("activity_days_per_week", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
    ).round(4).reset_index()

    result = {
        "pearson_correlation": round(corr, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "interpretation": (
            f"Physical activity and mental health risk: "
            f"r={corr:.3f}, p={p_value:.4f}"
        ),
        "by_bucket": activity_analysis,
    }

    logger.success(f"Activity correlation: r={corr:.3f}, p={p_value:.4f}")
    return result


def social_support_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse how social support levels relate to mental health outcomes.
    """
    logger.info("Analysing social support impact...")

    support_order = ["Never", "Rarely", "Sometimes", "Usually", "Always"]

    analysis = df.groupby("social_support").agg(
        count=("social_support", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
    ).round(4).reset_index()

    analysis["support_order"] = analysis["social_support"].map(
        {s: i for i, s in enumerate(support_order)}
    )
    analysis = analysis.sort_values("support_order").drop("support_order", axis=1)

    logger.success(f"Social support analysis: {len(analysis)} groups")
    return analysis


def country_comparison(df: pd.DataFrame) -> dict:
    """
    Statistical comparison between USA and Canada mental health outcomes.
    Uses independent t-test to check if difference is significant.
    """
    logger.info("Comparing USA vs Canada...")

    usa = df[df["country"] == "USA"]["mental_health_risk_score"]
    canada = df[df["country"] == "Canada"]["mental_health_risk_score"]

    t_stat, p_value = stats.ttest_ind(usa, canada)

    summary = df.groupby("country").agg(
        count=("mental_health_risk_score", "count"),
        avg_risk_score=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
        help_seeking_rate=("sought_help", "mean"),
        avg_sleep=("sleep_hours", "mean"),
    ).round(4).reset_index()

    result = {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "usa_mean": round(usa.mean(), 3),
        "canada_mean": round(canada.mean(), 3),
        "summary": summary,
    }

    logger.success(f"Country comparison: USA={usa.mean():.2f}, Canada={canada.mean():.2f}, p={p_value:.4f}")
    return result


def run_all_correlations(df: pd.DataFrame) -> dict:
    """Run all correlation analyses."""
    return {
        "sleep": sleep_correlation(df),
        "activity": activity_correlation(df),
        "social_support": social_support_analysis(df),
        "country": country_comparison(df),
    }


if __name__ == "__main__":
    df = load_data()
    results = run_all_correlations(df)

    print("\n── Sleep Correlation ──")
    print(results["sleep"]["interpretation"])
    print(results["sleep"]["by_bucket"].to_string(index=False))

    print("\n── Activity Correlation ──")
    print(results["activity"]["interpretation"])

    print("\n── Country Comparison ──")
    print(f"USA mean risk: {results['country']['usa_mean']}")
    print(f"Canada mean risk: {results['country']['canada_mean']}")
    print(f"Significant difference: {results['country']['significant']}")
    print(results["country"]["summary"].to_string(index=False))
