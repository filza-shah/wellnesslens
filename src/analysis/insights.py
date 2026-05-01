# src/analysis/insights.py
#
# Combines all analyses into a clean set of key findings.
# This is what gets displayed on the dashboard as the "headline" insights.

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.analysis.trends import run_all_trends
from src.analysis.correlations import run_all_correlations
from src.analysis.significance import run_all_tests

PROCESSED_DIR = Path("data/processed")


def generate_key_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generate a list of key findings from all analyses.
    Each insight has a title, value, description, and significance level.
    """
    logger.info("Generating key insights...")

    trends = run_all_trends(df)
    correlations = run_all_correlations(df)
    significance = run_all_tests(df)

    insights = []

    # ── Insight 1: Overall risk trend ────────────────────────────────────────
    yearly = trends["yearly"]
    first_year_risk = yearly.iloc[0]["avg_risk_score"]
    last_year_risk  = yearly.iloc[-1]["avg_risk_score"]
    risk_change = ((last_year_risk - first_year_risk) / first_year_risk * 100)

    insights.append({
        "title": "Mental Health Risk Trend",
        "value": f"{risk_change:+.1f}%",
        "description": f"Average mental health risk score changed by {risk_change:+.1f}% from {yearly.iloc[0]['year']} to {yearly.iloc[-1]['year']}",
        "severity": "high" if risk_change > 5 else "medium" if risk_change > 0 else "low",
        "category": "trend",
    })

    # ── Insight 2: Youth vs adults ────────────────────────────────────────────
    age_summary = trends["age_summary"]
    youth_mask = age_summary["age_group"].isin(["12-13", "13-14", "14-15", "15-16", "16-17", "17-18", "18-24"])
    youth_risk  = age_summary[youth_mask]["avg_risk_score"].mean()
    adult_risk  = age_summary[~youth_mask]["avg_risk_score"].mean()

    insights.append({
        "title": "Youth vs Adult Risk",
        "value": f"{youth_risk:.1f} vs {adult_risk:.1f}",
        "description": f"Youth (12-24) average risk score is {youth_risk:.1f} compared to {adult_risk:.1f} for adults 25+",
        "severity": "high" if youth_risk > adult_risk else "low",
        "category": "age",
    })

    # ── Insight 3: Sleep impact ───────────────────────────────────────────────
    sleep = correlations["sleep"]
    low_sleep_risk  = sleep["by_bucket"][sleep["by_bucket"]["sleep_bucket"] == "<5h"]["avg_risk_score"].values
    good_sleep_risk = sleep["by_bucket"][sleep["by_bucket"]["sleep_bucket"] == "7-8h"]["avg_risk_score"].values

    if len(low_sleep_risk) > 0 and len(good_sleep_risk) > 0:
        sleep_diff = low_sleep_risk[0] - good_sleep_risk[0]
        insights.append({
            "title": "Sleep Deprivation Impact",
            "value": f"+{sleep_diff:.1f} risk points",
            "description": f"People sleeping less than 5 hours score {sleep_diff:.1f} points higher on the risk scale than those sleeping 7-8 hours",
            "severity": "high" if sleep_diff > 15 else "medium",
            "category": "lifestyle",
        })

    # ── Insight 4: Gender gap ─────────────────────────────────────────────────
    gender = significance["gender"]
    gender_gap = abs(gender["female_mean"] - gender["male_mean"])
    insights.append({
        "title": "Gender Mental Health Gap",
        "value": f"{gender_gap:.1f} points",
        "description": f"{'Statistically significant' if gender['significant'] else 'No significant'} difference between female (M={gender['female_mean']}) and male (M={gender['male_mean']}) risk scores",
        "severity": "medium" if gender["significant"] else "low",
        "category": "gender",
    })

    # ── Insight 5: Help-seeking gap ───────────────────────────────────────────
    help = significance["help_seeking"]
    help_gap = (help["high_risk_help_rate"] - help["low_risk_help_rate"]) * 100
    insights.append({
        "title": "Help-Seeking Gap",
        "value": f"{help['high_risk_help_rate']:.0%} vs {help['low_risk_help_rate']:.0%}",
        "description": f"High-risk individuals seek professional help at {help['high_risk_help_rate']:.0%} vs {help['low_risk_help_rate']:.0%} for low-risk — a {abs(help_gap):.1f}pp gap",
        "severity": "high",
        "category": "intervention",
    })

    # ── Insight 6: Country comparison ─────────────────────────────────────────
    country = correlations["country"]
    insights.append({
        "title": "USA vs Canada",
        "value": f"{country['usa_mean']} vs {country['canada_mean']}",
        "description": f"USA avg risk score: {country['usa_mean']} | Canada: {country['canada_mean']}. Difference is {'statistically significant' if country['significant'] else 'not statistically significant'}",
        "severity": "low",
        "category": "geography",
    })

    logger.success(f"Generated {len(insights)} key insights")
    return insights


def run_full_analysis(df: pd.DataFrame) -> dict:
    """Run complete analysis pipeline and return all results."""
    logger.info("=" * 50)
    logger.info("WellnessLens Full Analysis Pipeline")
    logger.info("=" * 50)

    results = {
        "trends": run_all_trends(df),
        "correlations": run_all_correlations(df),
        "significance": run_all_tests(df),
        "insights": generate_key_insights(df),
    }

    logger.success("Full analysis complete")
    return results


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")
    results = run_full_analysis(df)

    print("\n── KEY INSIGHTS ──\n")
    for i, insight in enumerate(results["insights"], 1):
        print(f"{i}. {insight['title']}: {insight['value']}")
        print(f"   {insight['description']}\n")
