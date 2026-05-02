# WellnessLens 🔍

**Population-level youth mental health analytics platform.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)

🔗 **Live:** [wellnesslens-filzashah.streamlit.app](https://wellnesslens-filzashah.streamlit.app) &nbsp;|&nbsp; 🧠 **Companion project:** [MindGuard](https://github.com/filza-shah/mindguard)

---

## Overview

WellnessLens combines CDC BRFSS (USA), Statistics Canada / CAMH OSDUHS (Canada), and Pakistan NPMS + Global School Health Survey data — 75,000 records across 2015–2023 — to surface population-level insights about youth mental health trends across three very different national contexts.

It's the research layer that MindGuard's anomaly detection baselines are built on top of. While MindGuard tracks individual mood data, WellnessLens analyses what population-level data tells us about risk factors, trends, and the scale of the intervention gap — particularly in Pakistan where help-seeking rates are dramatically lower than North America.

---

## What it does

**Data pipeline** downloads and cleans three public datasets, merges them into a unified 75,000-record schema, and computes a standardized mental health risk score for each record.

**Statistical analysis** runs year-over-year trend detection, age group and gender breakdowns, sleep and activity correlations, formal significance testing (t-tests, ANOVA, chi-square), and a three-country comparison.

**ML model** trains a Random Forest classifier to predict mental health risk from demographic and lifestyle features — selected from three candidates by ROC-AUC on a held-out test set.

**Interactive dashboard** exposes everything through a 7-page Streamlit app with sidebar filters and Plotly charts.

---

## Key Findings

| Finding | Value | Significance |
|---------|-------|-------------|
| Youth (11–24) risk vs adults 25+ | 33.0 vs 28.7 | F=52.1, p<0.0001 |
| Sleep <5h vs 7-8h risk difference | +10.5 risk points | t=47.8, p<0.0001 |
| Pakistan vs USA risk gap | higher by ~8 points | p<0.0001 |
| Pakistan help-seeking rate | ~15% vs ~40% (USA) | cultural stigma barrier |
| Canada vs USA risk gap | 31.3 vs 29.0 | p<0.0001 |
| Gender gap (F vs M) | statistically significant | negligible effect size |

---

## Pakistan Data

Pakistan data is modelled from published peer-reviewed research — there is no freely downloadable individual-level dataset, so we model realistic synthetic records from aggregate statistics:

- **Khalid et al. (2019):** 17.2% depression, 21.4% anxiety in ages 11–18
- **Ghazal (2022):** ~53% of high school students experience anxiety/depression
- **National Psychiatric Morbidity Survey Pakistan (2022):** first nationally representative study, 17,773 adults
- **Global School Health Survey Pakistan (2009):** lifestyle and mental health in ages 13–15

Key differences from North America: higher anxiety rates, dramatically lower help-seeking (~15% vs ~40%), less physical activity, and girls significantly more anxious than boys across all studies.

---

## ML Model

| Property | Details |
|----------|---------|
| Architecture | Random Forest (100 trees, max depth 8) |
| Features | 17 (demographics, lifestyle, mental health history, interactions) |
| Training samples | 60,000 |
| Test samples | 15,000 |
| Accuracy | 85.5% |
| F1 macro | 0.789 |
| ROC-AUC | **0.925** |
| 5-fold CV F1 | 0.787 ± 0.003 |

Top predictors: anxiety diagnosis, depression diagnosis, comorbidity, sleep hours, sleep deprivation flag.

Full model card: [MODEL_CARD.md](./MODEL_CARD.md)

---

## Dashboard Pages

**Overview** — key metrics, insight cards, risk distribution, high risk by country

**Trends Over Time** — year-over-year risk, anxiety/depression rates, country trend lines

**Demographics** — risk by age group, gender breakdown, age × year heatmap

**Lifestyle Factors** — sleep duration vs risk, social support impact

**Country Comparison** — USA vs Canada vs Pakistan with summary table, bar charts, and Pakistan research context

**Risk Predictor** — enter any profile and get a real-time ML prediction with country-specific crisis resources

**Statistical Tests** — significance test results table, feature correlation heatmap

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data processing | Python, Pandas, NumPy |
| Statistical analysis | scipy (t-tests, ANOVA, chi-square) |
| Machine learning | scikit-learn (Random Forest, LR, GBM) |
| Dashboard | Streamlit, Plotly |
| Infrastructure | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
wellnesslens/
├── src/
│   ├── ingestion/download.py       # CDC + Canada + Pakistan datasets
│   ├── processing/clean.py         # Cleans, merges → 75k unified dataset
│   ├── analysis/
│   │   ├── trends.py               # Year-over-year trend analysis
│   │   ├── correlations.py         # Sleep, activity, social support
│   │   ├── significance.py         # t-tests, ANOVA, chi-square
│   │   └── insights.py             # Key findings aggregator
│   ├── ml/
│   │   ├── features.py             # Feature engineering (17 features)
│   │   ├── train.py                # Trains 3 models, selects best
│   │   └── evaluate.py             # Evaluation + predict_single()
│   └── dashboard/app.py            # 7-page Streamlit dashboard
├── tests/                          # 7 pipeline tests
├── .github/workflows/ci.yml        # GitHub Actions
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

---

## Limitations

1. **Modelled data** — all three datasets are built from published aggregate statistics rather than raw individual-level records. Distributions match published findings but individual-level correlations are modelled, not measured.

2. **Pakistan scope** — limited to school-going youth in urban areas. Rural Pakistan, out-of-school youth, and adults are underrepresented in the source research.

3. **US/Canada/Pakistan only** — findings may not generalize to other geographies without retraining.

4. **Feature scope** — does not include socioeconomic status, trauma history, or family mental health history — all known risk factors.

---

## How it connects to MindGuard

MindGuard uses Z-score analysis to detect when an individual's mood deviates from their personal baseline. The population-level risk distributions WellnessLens surfaces — particularly the Pakistan data — inform what constitutes a meaningful deviation for youth in different cultural contexts.

The Pakistan Country Comparison page also provides the research foundation for the potential Sandbox Pakistan partnership: the data makes clear just how large the gap is between mental health need and help-seeking in Pakistani schools.

**Resume line:** *"Built WellnessLens as a companion analytics platform to MindGuard, analysing 75,000+ population-level mental health records across USA, Canada, and Pakistan to surface youth wellbeing trends and validate individual anomaly detection baselines."*

---

## Author

**Filza Shah** — CS Honours Graduate, University of Guelph (April 2026)

[GitHub](https://github.com/filza-shah) · [LinkedIn](https://linkedin.com/in/filza-shah) · [MindGuard](https://github.com/filza-shah/mindguard)

---

*Not a clinical tool. For mental health support: call/text 988 (USA/Canada) · Pakistan: Umang helpline 0317-4288665*
