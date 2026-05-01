# src/dashboard/app.py
# Milestone 1 placeholder — full dashboard built in Milestone 4

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="WellnessLens",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 WellnessLens")
st.subheader("Youth Mental Health Analytics Platform")

st.markdown("""
**Data sources:** CDC BRFSS + Statistics Canada / CAMH OSDUHS

**Status:** Data pipeline running. Full dashboard coming in Milestone 4.
""")

# Check if processed data exists
data_path = Path("data/processed/unified_mental_health.csv")

if data_path.exists():
    df = pd.read_csv(data_path)
    st.success(f"✅ Dataset loaded: {len(df):,} records from {df['source'].nunique()} sources")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Year Range", f"{df['year'].min()}–{df['year'].max()}")
    col3.metric("Countries", df['country'].nunique())
    col4.metric("High Risk Rate", f"{df['high_risk'].mean():.1%}")

    st.dataframe(df.head(20))
else:
    st.warning("⚠️ No processed data found. Run the pipeline first:")
    st.code("python -m src.ingestion.download\npython -m src.processing.clean")
