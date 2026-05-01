# src/dashboard/app.py — add this block at the very top before st.set_page_config
# This ensures data and model exist on Streamlit Cloud

import sys
from pathlib import Path

# ── Auto-setup on first run ───────────────────────────────────────────────────
# Runs data pipeline and model training if files don't exist.
# This makes the app work out of the box on Streamlit Cloud.
def _auto_setup():
    raw     = Path("data/raw/cdc_mental_health.csv")
    processed = Path("data/processed/unified_mental_health.csv")
    model   = Path("data/models/risk_model.joblib")

    if not raw.exists():
        from src.ingestion.download import download_all
        download_all()

    if not processed.exists():
        from src.processing.clean import run_pipeline
        run_pipeline()

    if not model.exists():
        from src.ml.train import train
        train()

_auto_setup()
