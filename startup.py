#!/usr/bin/env python3
# startup.py
#
# Runs on Streamlit Cloud startup.
# Downloads and processes data if not already present,
# then trains the ML model if not already trained.
#
# Streamlit Cloud runs this via the "before script" in packages.txt
# or you can call it directly in app.py

import os
import sys
from pathlib import Path

def setup():
    """Full setup pipeline for fresh deployment."""

    print("WellnessLens — Running startup pipeline...")

    # ── Step 1: Download data if not present ──────────────────────────────────
    raw_cdc    = Path("data/raw/cdc_mental_health.csv")
    raw_canada = Path("data/raw/canada_youth_mental_health.csv")

    if not raw_cdc.exists() or not raw_canada.exists():
        print("Downloading datasets...")
        from src.ingestion.download import download_all
        download_all()
    else:
        print("Raw data already present — skipping download")

    # ── Step 2: Process data if not present ──────────────────────────────────
    processed = Path("data/processed/unified_mental_health.csv")

    if not processed.exists():
        print("Processing data...")
        from src.processing.clean import run_pipeline
        run_pipeline()
    else:
        print("Processed data already present — skipping processing")

    # ── Step 3: Train model if not present ───────────────────────────────────
    model_path = Path("data/models/risk_model.joblib")

    if not model_path.exists():
        print("Training ML model...")
        from src.ml.train import train
        train()
    else:
        print("Model already trained — skipping training")

    print("Startup complete. Launching dashboard...")


if __name__ == "__main__":
    setup()
