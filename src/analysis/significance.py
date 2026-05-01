# src/analysis/significance.py
#
# Statistical significance testing for key findings.
# This is what separates real data analysis from just making charts.
#
# Tests used:
# - Independent t-test: compare means between two groups
# - Chi-square test: compare proportions between groups
# - ANOVA: compare means across multiple groups
# - Effect size (Cohen's d): how practically significant is the difference?

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Calculate Cohen's d effect size.
    Tells us HOW MUCH groups differ, not just IF they differ.

    Interpretation:
    - d < 0.2: negligible
    - 0.2 <= d < 0.5: small
    - 0.5 <= d < 0.8: medium
    - d >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0


def interpret_effect_size(d: float) -> str:
    d = abs(d)
    if d < 0.2: return "negligible"
    if d < 0.5: return "small"
    if d < 0.8: return "medium"
    return "large"


def test_gender_difference(df: pd.DataFrame) -> dict:
    """
    H0: No difference in mental health risk between males and females
    H1: There IS a significant difference
    """
    male   = df[df["gender"] == "Male"]["mental_health_risk_score"]
    female = df[df["gender"] == "Female"]["mental_health_risk_score"]

    t_stat, p_value = stats.ttest_ind(male, female)
    d = cohens_d(female, male)

    return {
        "test": "Independent t-test",
        "hypothesis": "Gender difference in mental health risk",
        "male_mean": round(male.mean(), 3),
        "female_mean": round(female.mean(), 3),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "cohens_d": round(d, 4),
        "effect_size": interpret_effect_size(d),
        "conclusion": (
            f"{'Significant' if p_value < 0.05 else 'No significant'} difference between "
            f"male (M={male.mean():.2f}) and female (M={female.mean():.2f}) risk scores "
            f"[t={t_stat:.3f}, p={p_value:.4f}, d={d:.3f} ({interpret_effect_size(d)} effect)]"
        )
    }


def test_age_group_differences(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA: Do mental health risk scores differ across age groups?
    H0: All age groups have the same mean risk score
    H1: At least one age group differs significantly
    """
    age_groups = df.groupby("age_group")["mental_health_risk_score"].apply(list)
    f_stat, p_value = stats.f_oneway(*age_groups.values)

    # Post-hoc: which specific groups differ most?
    summary = df.groupby("age_group")["mental_health_risk_score"].agg(
        ["mean", "std", "count"]
    ).round(3).reset_index()

    return {
        "test": "One-way ANOVA",
        "hypothesis": "Age group differences in mental health risk",
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "group_summary": summary,
        "conclusion": (
            f"{'Significant' if p_value < 0.05 else 'No significant'} differences "
            f"across age groups [F={f_stat:.3f}, p={p_value:.4f}]"
        )
    }


def test_sleep_impact(df: pd.DataFrame) -> dict:
    """
    Compare mental health risk for people sleeping < 6 hours vs >= 7 hours.
    """
    low_sleep  = df[df["sleep_hours"] < 6]["mental_health_risk_score"]
    good_sleep = df[df["sleep_hours"] >= 7]["mental_health_risk_score"]

    t_stat, p_value = stats.ttest_ind(low_sleep, good_sleep)
    d = cohens_d(low_sleep, good_sleep)

    return {
        "test": "Independent t-test",
        "hypothesis": "Sleep deprivation (<6h) vs adequate sleep (>=7h)",
        "low_sleep_mean": round(low_sleep.mean(), 3),
        "good_sleep_mean": round(good_sleep.mean(), 3),
        "low_sleep_n": len(low_sleep),
        "good_sleep_n": len(good_sleep),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "cohens_d": round(d, 4),
        "effect_size": interpret_effect_size(d),
        "conclusion": (
            f"Sleep deprived group (M={low_sleep.mean():.2f}) vs adequate sleep (M={good_sleep.mean():.2f}): "
            f"{'significant' if p_value < 0.05 else 'not significant'} "
            f"[t={t_stat:.3f}, p={p_value:.4f}, d={d:.3f} ({interpret_effect_size(d)} effect)]"
        )
    }


def test_help_seeking_chi_square(df: pd.DataFrame) -> dict:
    """
    Chi-square test: Is help-seeking behaviour different between high and low risk groups?
    """
    contingency = pd.crosstab(df["high_risk"], df["sought_help"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    high_risk_help = df[df["high_risk"] == 1]["sought_help"].mean()
    low_risk_help  = df[df["high_risk"] == 0]["sought_help"].mean()

    return {
        "test": "Chi-square test of independence",
        "hypothesis": "Help-seeking differs between high and low risk groups",
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_value, 6),
        "degrees_of_freedom": dof,
        "significant": p_value < 0.05,
        "high_risk_help_rate": round(high_risk_help, 4),
        "low_risk_help_rate": round(low_risk_help, 4),
        "conclusion": (
            f"High-risk individuals seek help at {high_risk_help:.1%} vs "
            f"{low_risk_help:.1%} for low-risk. "
            f"Difference is {'significant' if p_value < 0.05 else 'not significant'} "
            f"[χ²={chi2:.3f}, p={p_value:.4f}]"
        )
    }


def run_all_tests(df: pd.DataFrame) -> dict:
    """Run all significance tests."""
    logger.info("Running statistical significance tests...")

    results = {
        "gender": test_gender_difference(df),
        "age_groups": test_age_group_differences(df),
        "sleep": test_sleep_impact(df),
        "help_seeking": test_help_seeking_chi_square(df),
    }

    logger.success("All significance tests complete")
    return results


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "unified_mental_health.csv")
    results = run_all_tests(df)

    print("\n── Statistical Significance Tests ──\n")
    for name, result in results.items():
        print(f"[{name.upper()}] {result['conclusion']}")
        print(f"  p-value: {result['p_value']} | significant: {result['significant']}\n")
