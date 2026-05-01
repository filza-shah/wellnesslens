# WellnessLens 🔍

**Population-level youth mental health analytics platform.**

Combines CDC BRFSS and Statistics Canada / CAMH OSDUHS datasets (65,000+ records) to surface insights about youth mental wellbeing trends — the research layer that MindGuard's anomaly detection baselines are built on.

## Status
- [x] Milestone 1 — Data pipeline + folder structure
- [ ] Milestone 2 — Statistical analysis
- [ ] Milestone 3 — ML risk prediction model
- [ ] Milestone 4 — Streamlit dashboard
- [ ] Milestone 5 — Deployed live
- [ ] Milestone 6 — Polish + docs

## Stack
Python · Pandas · scikit-learn · Streamlit · PostgreSQL · Docker

## Run locally

```bash
cp .env.example .env
pip install -r requirements.txt

# Download and process data
python -m src.ingestion.download
python -m src.processing.clean

# Run dashboard
streamlit run src/dashboard/app.py
```

Or with Docker:
```bash
docker-compose up --build
```
