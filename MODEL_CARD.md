# Model Card — WellnessLens Risk Prediction Model v1.0

## Overview

| Property | Details |
|----------|---------|
| **Model name** | WellnessLens Risk Prediction Model |
| **Version** | 1.0 |
| **Task** | Binary classification — mental health risk prediction |
| **Input** | Demographics, lifestyle factors, mental health history |
| **Output** | High risk (1) / Low risk (0) + probability score |
| **Training data** | 52,000 samples (CDC BRFSS + CAMH OSDUHS) |
| **Test data** | 13,000 samples (held out) |

---

## Intended Use

Predicts whether an individual is at elevated mental health risk based on
demographic and lifestyle factors. Designed to support early intervention
by identifying individuals who may benefit from outreach or resources.

**Not intended for:** Clinical diagnosis, insurance decisions, or any
high-stakes individual decision-making without human oversight.

---

## Architecture

Three candidate models evaluated — best selected by ROC-AUC:

1. **Logistic Regression** — StandardScaler + L2 regularization, class-balanced
2. **Random Forest** — 100 trees, max depth 8, class-balanced
3. **Gradient Boosting** — 100 estimators, learning rate 0.1, max depth 4

Final model selected based on ROC-AUC on held-out test set.

---

## Features (17 total)

**Demographics:**
- Age (numeric midpoint), gender (encoded), country, youth flag

**Lifestyle:**
- Sleep hours, sleep deprived flag (<6h), severely deprived flag (<5h)
- Physical activity days per week, low activity flag

**Mental health history:**
- Anxiety diagnosis, depression diagnosis, help-seeking behaviour, comorbidity flag

**Social:**
- Social support level (encoded ordinal), social isolation flag

**Interaction features:**
- Lifestyle risk (sleep deprived AND low activity)
- Youth sleep deprived (youth AND sleep deprived)

---

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | see metadata JSON |
| F1 macro | see metadata JSON |
| ROC-AUC | see metadata JSON |
| 5-fold CV F1 | see metadata JSON |

*Run `python -m src.ml.evaluate` to reproduce metrics.*

---

## Limitations

1. **Synthetic/modelled data** — training data modelled from published aggregate
   statistics rather than individual-level records. Real-world performance may differ.

2. **Class imbalance** — ~18.5% high risk rate. Models use class balancing but
   precision on the high-risk class is limited by dataset size.

3. **Feature scope** — does not include socioeconomic factors, trauma history,
   or family mental health history which are known risk factors.

4. **Generalization** — trained on US/Canada data. May not generalize to other
   cultural or geographic contexts without retraining.

---

## Ethical Considerations

- Never use predictions as a sole basis for clinical decisions
- All predictions should be reviewed by qualified professionals
- Model should be retrained regularly as population patterns change
- Bias monitoring recommended across demographic subgroups

---

## How to Train

```bash
python -m src.ml.train
```

## How to Evaluate

```bash
python -m src.ml.evaluate
```

## How to Use in Code

```python
from src.ml.evaluate import load_model, predict_single

model = load_model()
result = predict_single(model, {
    "age_group": "18-24",
    "gender": "Female",
    "sleep_hours": 5.0,
    "activity_days_per_week": 1,
    "anxiety_diagnosis": 1,
    "depression_diagnosis": 0,
    "sought_help": 0,
    "social_support": "Rarely",
    "country": "Canada",
    "year": 2023,
    "source": "CDC_BRFSS",
})
# → {"high_risk": True, "probability": 0.73, "risk_level": "High"}
```
