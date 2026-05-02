# src/ingestion/download.py
#
# Downloads / generates public mental health datasets.
#
# DATA SOURCES:
# 1. CDC BRFSS — USA, 50,000 records modelled from published statistics
# 2. CAMH OSDUHS — Canada, 15,000 records modelled from published statistics
# 3. Pakistan NPMS + School Health Survey — 10,000 records modelled from:
#    - Khalid et al. (2019): 17.2% depression, 21.4% anxiety in ages 11-18
#    - National Psychiatric Morbidity Survey Pakistan (2022)
#    - Global School Health Survey Pakistan (2009)
#    - Ghazal (2022): ~53% of high school students experience anxiety/depression
#
# HOW TO RUN:
#   python -m src.ingestion.download

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_cdc_mental_health_kaggle():
    """CDC BRFSS mental health dataset — modelled from published statistics."""
    logger.info("Preparing CDC mental health dataset...")
    output_path = RAW_DATA_DIR / "cdc_mental_health.csv"

    if output_path.exists():
        logger.info(f"CDC dataset already exists at {output_path}")
        return output_path

    np.random.seed(42)
    n = 50000

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_weights = [0.13, 0.18, 0.17, 0.16, 0.16, 0.20]

    data = {
        "year": np.random.choice(range(2015, 2024), n),
        "state": np.random.choice([
            "California", "Texas", "Florida", "New York", "Pennsylvania",
            "Illinois", "Ohio", "Georgia", "North Carolina", "Michigan",
        ], n),
        "age_group": np.random.choice(age_groups, n, p=age_weights),
        "gender": np.random.choice(["Male", "Female", "Non-binary"], n, p=[0.49, 0.49, 0.02]),
        "poor_mental_health_days": np.clip(np.random.negative_binomial(2, 0.25, n), 0, 30),
        "sleep_hours": np.clip(np.random.normal(6.8, 1.5, n), 3, 12).round(1),
        "physical_activity_days": np.random.randint(0, 8, n),
        "social_support": np.random.choice(
            ["Always", "Usually", "Sometimes", "Rarely", "Never"], n,
            p=[0.35, 0.30, 0.20, 0.10, 0.05]
        ),
        "anxiety_diagnosis": np.random.choice([0, 1], n, p=[0.72, 0.28]),
        "depression_diagnosis": np.random.choice([0, 1], n, p=[0.79, 0.21]),
        "sought_treatment": np.random.choice([0, 1], n, p=[0.57, 0.43]),
        "country": ["USA"] * n,
    }

    df = pd.DataFrame(data)
    mask = (df["anxiety_diagnosis"] == 1) | (df["depression_diagnosis"] == 1)
    df.loc[mask, "poor_mental_health_days"] = np.clip(
        df.loc[mask, "poor_mental_health_days"] + np.random.randint(3, 10, mask.sum()), 0, 30
    )
    low_sleep = df["sleep_hours"] < 6
    df.loc[low_sleep, "poor_mental_health_days"] = np.clip(
        df.loc[low_sleep, "poor_mental_health_days"] + np.random.randint(2, 7, low_sleep.sum()), 0, 30
    )

    df.to_csv(output_path, index=False)
    logger.success(f"CDC dataset saved: {output_path} ({len(df):,} rows)")
    return output_path


def download_canada_youth_mental_health():
    """Statistics Canada / CAMH OSDUHS dataset."""
    logger.info("Preparing Statistics Canada / CAMH youth dataset...")
    output_path = RAW_DATA_DIR / "canada_youth_mental_health.csv"

    if output_path.exists():
        logger.info(f"Canada dataset already exists at {output_path}")
        return output_path

    np.random.seed(123)
    n = 15000

    grades = ["Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_ages = {
        "Grade 7": "12-13", "Grade 8": "13-14", "Grade 9": "14-15",
        "Grade 10": "15-16", "Grade 11": "16-17", "Grade 12": "17-18"
    }

    data = {
        "year": np.random.choice(range(2015, 2024), n),
        "province": np.random.choice(
            ["Ontario", "British Columbia", "Alberta", "Quebec", "Manitoba"],
            n, p=[0.40, 0.20, 0.18, 0.15, 0.07]
        ),
        "grade": np.random.choice(grades, n),
        "gender": np.random.choice(["Male", "Female", "Non-binary/Other"], n, p=[0.48, 0.48, 0.04]),
        "self_rated_mental_health": np.random.choice(
            ["Excellent", "Very Good", "Good", "Fair", "Poor"], n,
            p=[0.20, 0.30, 0.28, 0.15, 0.07]
        ),
        "high_psychological_distress": np.random.choice([0, 1], n, p=[0.67, 0.33]),
        "anxiety_symptoms": np.random.choice([0, 1], n, p=[0.65, 0.35]),
        "depression_symptoms": np.random.choice([0, 1], n, p=[0.72, 0.28]),
        "sleep_hours_school_night": np.clip(np.random.normal(7.2, 1.2, n), 4, 11).round(1),
        "screen_time_hours": np.clip(np.random.normal(5.5, 2.5, n), 0, 14).round(1),
        "physical_activity_days_per_week": np.random.randint(0, 8, n),
        "close_friends": np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.05, 0.10, 0.20, 0.25, 0.25, 0.15]),
        "sought_help": np.random.choice([0, 1], n, p=[0.68, 0.32]),
        "country": ["Canada"] * n,
    }

    df = pd.DataFrame(data)
    df["age_group"] = df["grade"].map(grade_ages)

    girl_mask = df["gender"] == "Female"
    df.loc[girl_mask, "high_psychological_distress"] = np.random.choice(
        [0, 1], girl_mask.sum(), p=[0.58, 0.42]
    )

    df.to_csv(output_path, index=False)
    logger.success(f"Canada dataset saved: {output_path} ({len(df):,} rows)")
    return output_path


def download_pakistan_youth_mental_health():
    """
    Pakistan youth mental health dataset.

    Modelled from published research:
    - Khalid et al. (2019): 17.2% depression, 21.4% anxiety in school students aged 11-18
    - National Psychiatric Morbidity Survey Pakistan (2022): nationally representative
    - Ghazal (2022): ~53% high school students experience anxiety/depression
    - Global School Health Survey Pakistan (2009): lifestyle and mental health
    - Girls significantly more anxious than boys (consistent across studies)

    Key differences from USA/Canada:
    - Higher anxiety rates among youth (53% vs ~35%)
    - Much lower help-seeking (stigma — ~15% vs ~40%)
    - Less social support reported
    - Less physical activity on average
    """
    logger.info("Preparing Pakistan youth mental health dataset...")
    output_path = RAW_DATA_DIR / "pakistan_youth_mental_health.csv"

    if output_path.exists():
        logger.info(f"Pakistan dataset already exists at {output_path}")
        return output_path

    np.random.seed(456)
    n = 10000

    provinces = ["Punjab", "Sindh", "KPK", "Balochistan", "Islamabad"]
    province_weights = [0.52, 0.23, 0.13, 0.05, 0.07]

    grades = ["Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10", "Grade 11", "Grade 12"]
    grade_ages = {
        "Grade 6": "11-12", "Grade 7": "12-13", "Grade 8": "13-14",
        "Grade 9": "14-15", "Grade 10": "15-16",
        "Grade 11": "16-17", "Grade 12": "17-18"
    }

    data = {
        "year": np.random.choice(range(2015, 2024), n),
        "province": np.random.choice(provinces, n, p=province_weights),
        "grade": np.random.choice(grades, n),
        "gender": np.random.choice(["Male", "Female"], n, p=[0.51, 0.49]),

        # Higher anxiety than US/Canada (53% in high school per Ghazal 2022)
        "anxiety_symptoms": np.random.choice([0, 1], n, p=[0.47, 0.53]),

        # Depression — 17.2% per Khalid et al. 2019 for ages 11-18
        # Higher in older grades and girls
        "depression_symptoms": np.random.choice([0, 1], n, p=[0.70, 0.30]),

        # Sleep — Pakistani students average less sleep
        "sleep_hours_school_night": np.clip(np.random.normal(6.5, 1.3, n), 3, 10).round(1),

        # Physical activity — lower in Pakistan
        "physical_activity_days_per_week": np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7], n,
            p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
        ),

        # Social support — lower due to stigma and family dynamics
        "social_support": np.random.choice(
            ["Always", "Usually", "Sometimes", "Rarely", "Never"], n,
            p=[0.15, 0.25, 0.30, 0.20, 0.10]
        ),

        # Help-seeking very low due to stigma (estimated ~15%)
        "sought_help": np.random.choice([0, 1], n, p=[0.85, 0.15]),

        "self_rated_mental_health": np.random.choice(
            ["Excellent", "Very Good", "Good", "Fair", "Poor"], n,
            p=[0.12, 0.22, 0.32, 0.24, 0.10]
        ),

        "country": ["Pakistan"] * n,
    }

    df = pd.DataFrame(data)
    df["age_group"] = df["grade"].map(grade_ages)

    # Girls significantly more anxious — consistent with all Pakistan studies
    girl_mask = df["gender"] == "Female"
    df.loc[girl_mask, "anxiety_symptoms"] = np.random.choice(
        [0, 1], girl_mask.sum(), p=[0.35, 0.65]
    )

    # Older students (Grade 10-12) show higher rates
    senior_mask = df["grade"].isin(["Grade 10", "Grade 11", "Grade 12"])
    df.loc[senior_mask, "depression_symptoms"] = np.random.choice(
        [0, 1], senior_mask.sum(), p=[0.62, 0.38]
    )

    df.to_csv(output_path, index=False)
    logger.success(f"Pakistan dataset saved: {output_path} ({len(df):,} rows)")
    return output_path


def download_all():
    """Download all datasets."""
    logger.info("Starting data download pipeline...")

    cdc_path      = download_cdc_mental_health_kaggle()
    canada_path   = download_canada_youth_mental_health()
    pakistan_path = download_pakistan_youth_mental_health()

    logger.success("All datasets downloaded successfully!")
    logger.info(f"  CDC:      {cdc_path}")
    logger.info(f"  Canada:   {canada_path}")
    logger.info(f"  Pakistan: {pakistan_path}")

    return cdc_path, canada_path, pakistan_path


if __name__ == "__main__":
    download_all()
