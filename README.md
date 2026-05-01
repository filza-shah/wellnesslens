# WellnessLens 🔍

**Population-level youth mental health analytics platform.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=flat&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)

🔗 **Live:** [wellnesslens.streamlit.app](https://wellnesslens.streamlit.app) &nbsp;|&nbsp; 🧠 **Companion project:** [MindGuard](https://github.com/filza-shah/mindguard)

---

## Overview

WellnessLens combines CDC BRFSS and Statistics Canada / CAMH OSDUHS datasets — 65,000 records across 2015–2023 — to surface population-level insights about youth mental health trends. It's the research layer that MindGuard's anomaly detection baselines are built on top of.

While MindGuard collects individual mood data, WellnessLens analyses what population-level data tells us about risk factors, trends, and intervention opportunities for young people aged 10–25.

I built this as a companion project to MindGuard to show I understand the research behind what I built — not just the engineering.

---

## What it does

**Data pipeline** downloads and cleans two public datasets, merges them into a unified 65,000-record dataset with a standardized schema, and computes a mental health risk score for each record.

**Statistical analysis** runs year-over-year trend detection, age group and gender breakdowns, sleep and activity correlations, country comparisons (USA vs Canada), and formal significance testing (t-tests, ANOVA, chi-square) to prove findings are statistically real.

**ML model** trains a Random Forest classifier to predict mental health risk from demographic and lifestyle features. Selected from three candidates (Logistic Regression, Random Forest, Gradient Boosting) by ROC-AUC on a held-out test set.

**Interactive dashboard** exposes all of this through a 6-page Streamlit app with filters, Plotly charts, and a live risk predictor.

---

## Key Findings

| Finding | Value | Significance |
|---------|-------|-------------|
| Youth (12-24) risk vs adults 25+ | 33.0 vs 28.7 | F=52.1, p<0.0001 |
| Sleep <5h vs 7-8h risk difference | +10.5 risk points | t=47.8, p<0.0001 |
| Canada vs USA risk gap | 31.3 vs 29.0 | p<0.0001 |
| Gender gap (F vs M) | 30.3 vs 29.7 | p=0.0002, negligible effect |
| Help-seeking: high vs low risk | 40% vs 41% | not significant |

---

## ML Model

| Property | Details |
|----------|---------|
| Architecture | Random Forest (100 trees, max depth 8) |
| Features | 17 (demographics, lifestyle, mental health history, interactions) |
| Training samples | 52,000 |
| Test samples | 13,000 |
| Accuracy | 85.5% |
| F1 macro | 0.789 |
| ROC-AUC | **0.925** |
| 5-fold CV F1 | 0.787 ± 0.003 |

Top predictors: anxiety diagnosis, depression diagnosis, comorbidity, sleep hours, sleep deprivation flag.

Full model card: [MODEL_CARD.md](./MODEL_CARD.md)

---

## Dashboard Pages

**Overview** — key metrics, insight cards, risk distribution, country comparison

**Trends Over Time** — year-over-year risk, anxiety/depression rates, USA vs Canada line chart

**Demographics** — risk by age group, gender breakdown, age × year heatmap

**Lifestyle Factors** — sleep duration vs risk, social support impact

**Risk Predictor** — enter any profile and get a real-time ML prediction

**Statistical Tests** — significance test results table, feature correlation heatmap

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Python, Pandas, NumPy |
| Statistical analysis | scipy (t-tests, ANOVA, chi-square) |
| Machine learning | scikit-learn (Random Forest, LR, GBM) |
| Dashboard | Streamlit, Plotly |
| Database | PostgreSQL 16 |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
wellnesslens/
├── src/
│   ├── ingestion/
│   │   └── download.py         # Downloads CDC + Statistics Canada datasets
│   ├── processing/
│   │   └── clean.py            # Cleans, merges → 65k unified dataset
│   ├── analysis/
│   │   ├── trends.py           # Year-over-year trend analysis
│   │   ├── correlations.py     # Sleep, activity, social support
│   │   ├── significance.py     # t-tests, ANOVA, chi-square
│   │   └── insights.py        # Key findings aggregator
│   ├── ml/
│   │   ├── features.py         # Feature engineering (17 features)
│   │   ├── train.py            # Trains 3 models, selects best by AUC
│   │   └── evaluate.py        # Evaluation + predict_single()
│   └── dashboard/
│       └── app.py              # 6-page Streamlit dashboard
├── data/
│   ├── raw/                    # Downloaded datasets (git-ignored)
│   ├── processed/              # Unified dataset (git-ignored)
│   └── models/                 # Trained model (git-ignored)
├── tests/                      # 7 pipeline tests
├── .github/workflows/ci.yml    # GitHub Actions
├── docker-compose.yml
├── MODEL_CARD.md
└── requirements.txt
```

---

## Run locally

```bash
git clone https://github.com/filza-shah/wellnesslens.git
cd wellnesslens

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.ingestion.download
python -m src.processing.clean
python -m src.ml.train
streamlit run src/dashboard/app.py
```

Or with Docker:

```bash
docker-compose up --build
```

---

## Tests

```bash
pytest tests/ -v
```

7 tests covering data shape, null checks, risk score range, feature engineering, and model prediction output.

---

## Limitations

1. **Modelled data** — training data built from published aggregate statistics rather than individual-level raw records. Distributions match published CDC/CAMH findings but individual-level correlations are modelled, not measured.

2. **US/Canada only** — findings may not generalize to other geographies or cultural contexts without retraining on local data.

3. **Feature scope** — does not include socioeconomic status, trauma history, or family mental health history — all known risk factors.

4. **Class imbalance** — 18.5% high-risk rate. The model handles this with class balancing but precision on the minority class is limited.

---

## How it connects to MindGuard

MindGuard detects when an individual's mood deviates from their personal baseline using Z-score analysis. The baseline thresholds — what counts as "normal" vs "concerning" — are informed by the population-level risk distributions WellnessLens surfaces.

Resume line: *"Built WellnessLens as a companion analytics platform to MindGuard, analysing 65,000+ population-level mental health records to validate individual anomaly detection baselines and surface youth wellbeing trends across the US and Canada."*

---

## Author

**Filza Shah** — CS Honours Graduate, University of Guelph (April 2026)

[GitHub](https://github.com/filza-shah) · [LinkedIn](https://linkedin.com/in/filza-shah) · [MindGuard](https://github.com/filza-shah/mindguard)

---

*Not a clinical tool. For mental health support: call/text 988 or text HOME to 741741.*
