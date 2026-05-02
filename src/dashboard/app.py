# src/dashboard/app.py — WellnessLens Full Dashboard

import sys
import subprocess
import os
from pathlib import Path

# ── Fix module path for Streamlit Cloud ──────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Auto-setup ────────────────────────────────────────────────────────────────
def _run(module):
    subprocess.run([sys.executable, "-m", module], check=True, cwd=str(ROOT))

if not Path(ROOT / "data/raw/pakistan_youth_mental_health.csv").exists():
    _run("src.ingestion.download")

if not Path(ROOT / "data/processed/unified_mental_health.csv").exists():
    _run("src.ingestion.download")
    _run("src.processing.clean")

if not Path(ROOT / "data/models/risk_model.joblib").exists():
    _run("src.ml.train")

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WellnessLens — Youth Mental Health Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #0F172A !important; }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #E2E8F0 !important; }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stMultiSelect > div > div {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        color: #E2E8F0 !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg,
    [data-testid="stSidebar"] .stMultiSelect svg { fill: #94A3B8 !important; }
    [data-testid="stSidebar"] input {
        color: #E2E8F0 !important;
        background-color: #1E293B !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div { background-color: #334155 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p { color: #CBD5E1 !important; }
    [data-testid="metric-container"] {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .insight-card {
        background: white; border: 1px solid #E2E8F0;
        border-radius: 12px; padding: 20px; margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); color: #334155;
    }
    .insight-high   { border-left: 4px solid #EF4444; }
    .insight-medium { border-left: 4px solid #F59E0B; }
    .insight-low    { border-left: 4px solid #14B8A6; }
    .insight-purple { border-left: 4px solid #8B5CF6; }
    h1 { color: #0F172A !important; font-weight: 800 !important; }
    h2 { color: #0F172A !important; font-weight: 700 !important; }
    h3 { color: #334155 !important; font-weight: 600 !important; }
    hr { border-color: #E2E8F0; margin: 24px 0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "teal": "#14B8A6", "navy": "#0F172A", "red": "#EF4444",
    "amber": "#F59E0B", "green": "#10B981", "purple": "#8B5CF6",
    "orange": "#F97316", "slate": "#64748B",
}
COUNTRY_COLORS = {"USA": "#14B8A6", "Canada": "#0F172A", "Pakistan": "#8B5CF6"}

# ── Inlined predict_single — avoids import path issues on Streamlit Cloud ─────
def predict_single(model, profile: dict) -> dict:
    """Predict risk for a single profile. Inlined to avoid import path issues."""
    df_p = pd.DataFrame([profile])

    # Encode categoricals
    gender_map = {"Male": 0, "Female": 1, "Non-binary": 2, "Non-binary/Other": 2}
    support_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Usually": 3, "Always": 4}
    age_midpoints = {
        "11-12": 11.5, "12-13": 12.5, "13-14": 13.5, "14-15": 14.5,
        "15-16": 15.5, "16-17": 16.5, "17-18": 17.5, "18-24": 21.0,
        "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55-64": 59.5, "65+": 70.0,
    }

    df_p["gender_encoded"] = df_p["gender"].map(gender_map).fillna(2)
    df_p["social_support_encoded"] = df_p["social_support"].map(support_map).fillna(2)
    df_p["country_encoded"] = (df_p["country"] == "Canada").astype(int)
    df_p["age_numeric"] = df_p["age_group"].map(age_midpoints).fillna(30.0)
    df_p["is_youth"] = (df_p["age_numeric"] < 25).astype(int)

    # Interaction features
    df_p["sleep_deprived"] = (df_p["sleep_hours"] < 6).astype(int)
    df_p["severely_sleep_deprived"] = (df_p["sleep_hours"] < 5).astype(int)
    df_p["low_activity"] = (df_p["activity_days_per_week"] <= 1).astype(int)
    df_p["lifestyle_risk"] = df_p["sleep_deprived"] * df_p["low_activity"]
    df_p["socially_isolated"] = (df_p["social_support_encoded"] <= 1).astype(int)
    df_p["comorbid"] = (
        (df_p["anxiety_diagnosis"] == 1) & (df_p["depression_diagnosis"] == 1)
    ).astype(int)
    df_p["youth_sleep_deprived"] = df_p["is_youth"] * df_p["sleep_deprived"]

    feature_cols = [
        "age_numeric", "gender_encoded", "country_encoded", "is_youth",
        "sleep_hours", "sleep_deprived", "severely_sleep_deprived",
        "activity_days_per_week", "low_activity",
        "anxiety_diagnosis", "depression_diagnosis", "sought_help", "comorbid",
        "social_support_encoded", "socially_isolated",
        "lifestyle_risk", "youth_sleep_deprived",
    ]

    X = df_p[feature_cols].fillna(0)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "high_risk": bool(prediction),
        "probability": round(float(probability), 4),
        "risk_level": (
            "Critical" if probability >= 0.75 else
            "High"     if probability >= 0.50 else
            "Moderate" if probability >= 0.30 else
            "Low"
        ),
    }


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = ROOT / "data/processed/unified_mental_health.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_resource
def load_model():
    try:
        mp = ROOT / "data/models/risk_model.joblib"
        return joblib.load(mp) if mp.exists() else None
    except Exception:
        return None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 WellnessLens")
    st.markdown("*Youth Mental Health Analytics*")
    st.markdown("---")
    st.markdown("### Filters")

    df_raw = load_data()

    if df_raw is not None:
        countries   = ["All"] + sorted(df_raw["country"].unique().tolist())
        sel_country = st.selectbox("Country", countries, key="country_filter")
        min_y, max_y = int(df_raw["year"].min()), int(df_raw["year"].max())
        year_range  = st.slider("Year Range", min_y, max_y, (min_y, max_y), key="year_filter")
        age_order   = ["11-12","12-13","13-14","14-15","15-16","16-17","17-18",
                       "18-24","25-34","35-44","45-54","55-64","65+"]
        avail_ages  = [a for a in age_order if a in df_raw["age_group"].unique()]
        sel_ages    = st.multiselect("Age Groups", avail_ages, default=avail_ages, key="age_filter")
        genders     = ["All"] + sorted(df_raw["gender"].unique().tolist())
        sel_gender  = st.selectbox("Gender", genders, key="gender_filter")

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", [
        "📊 Overview", "📈 Trends Over Time", "👥 Demographics",
        "😴 Lifestyle Factors", "🌍 Country Comparison",
        "🤖 Risk Predictor", "🔬 Statistical Tests",
    ], label_visibility="collapsed", key="nav")

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("CDC BRFSS (USA) · CAMH OSDUHS (Canada)")
    st.markdown("Pakistan NPMS + GSHS")
    st.markdown("Years: 2015–2023 · Records: 75,000")


# ── Apply filters ─────────────────────────────────────────────────────────────
def apply_filters(df):
    if sel_country != "All":
        df = df[df["country"] == sel_country]
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    if sel_ages:
        df = df[df["age_group"].isin(sel_ages)]
    if sel_gender != "All":
        df = df[df["gender"] == sel_gender]
    return df

if df_raw is None:
    st.error("⚠️ No data found. Run the pipeline first.")
    st.stop()

df = apply_filters(df_raw.copy())


# ════════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Youth Mental Health Analytics")
    st.markdown(f"Analysing **{len(df):,} records** across **{df['country'].nunique()} countries** · {df['year'].min()}–{df['year'].max()}")
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("High Risk Rate", f"{df['high_risk'].mean()*100:.1f}%")
    c3.metric("Anxiety Rate", f"{df['anxiety_diagnosis'].mean()*100:.1f}%")
    c4.metric("Depression Rate", f"{df['depression_diagnosis'].mean()*100:.1f}%")
    c5.metric("Help-Seeking Rate", f"{df['sought_help'].mean()*100:.1f}%")

    st.markdown("---")
    st.subheader("Key Findings")
    col_a, col_b = st.columns(2)

    with col_a:
        ym = df["age_group"].isin(["11-12","12-13","13-14","14-15","15-16","16-17","17-18","18-24"])
        yr, ar = df[ym]["mental_health_risk_score"].mean(), df[~ym]["mental_health_risk_score"].mean()
        st.markdown(f"""<div class="insight-card insight-high">
            <strong>🔴 Youth Risk Higher Than Adults</strong><br>
            Youth (11–24): <strong>{yr:.1f}</strong> vs Adults 25+: <strong>{ar:.1f}</strong> — a {yr-ar:.1f} point gap
        </div>""", unsafe_allow_html=True)

        ls = df[df["sleep_hours"]<5]["mental_health_risk_score"].mean()
        gs = df[df["sleep_hours"]>=7]["mental_health_risk_score"].mean()
        st.markdown(f"""<div class="insight-card insight-high">
            <strong>😴 Sleep Deprivation Adds +{ls-gs:.1f} Risk Points</strong><br>
            &lt;5h sleep scores <strong>{ls:.1f}</strong> vs <strong>{gs:.1f}</strong> for 7h+
        </div>""", unsafe_allow_html=True)

    with col_b:
        pak = df[df["country"]=="Pakistan"]["mental_health_risk_score"].mean() if "Pakistan" in df["country"].values else None
        usa = df[df["country"]=="USA"]["mental_health_risk_score"].mean() if "USA" in df["country"].values else None
        if pak and usa:
            st.markdown(f"""<div class="insight-card insight-purple">
                <strong>🇵🇰 Pakistan vs USA</strong><br>
                Pakistan avg risk: <strong>{pak:.1f}</strong> vs USA: <strong>{usa:.1f}</strong>
                — driven by higher anxiety and lower help-seeking
            </div>""", unsafe_allow_html=True)

        hr_h = df[df["high_risk"]==1]["sought_help"].mean()*100
        lr_h = df[df["high_risk"]==0]["sought_help"].mean()*100
        st.markdown(f"""<div class="insight-card insight-medium">
            <strong>🆘 Help-Seeking Gap</strong><br>
            High-risk: <strong>{hr_h:.0f}%</strong> vs Low-risk: <strong>{lr_h:.0f}%</strong>.
            Stigma is a major barrier especially in Pakistan.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(df, x="mental_health_risk_score", nbins=40,
            color_discrete_sequence=[C["teal"]], template="plotly_white",
            labels={"mental_health_risk_score":"Risk Score"})
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("High Risk Rate by Country")
        cr = df.groupby("country")["high_risk"].mean().reset_index()
        cr["high_risk"] = (cr["high_risk"]*100).round(1)
        fig = px.bar(cr, x="country", y="high_risk", color="country",
            color_discrete_map=COUNTRY_COLORS,
            labels={"high_risk":"High Risk Rate (%)","country":""}, template="plotly_white")
        fig.update_layout(height=300, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TRENDS OVER TIME
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends Over Time":
    st.title("Mental Health Trends Over Time")
    st.markdown("---")

    yearly = df.groupby("year").agg(
        avg_risk=("mental_health_risk_score","mean"),
        anxiety_rate=("anxiety_diagnosis","mean"),
        depression_rate=("depression_diagnosis","mean"),
        help_rate=("sought_help","mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["avg_risk"],
        mode="lines+markers", line=dict(color=C["teal"], width=3), marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(20,184,166,0.1)"))
    fig.update_layout(template="plotly_white", height=350,
        xaxis_title="Year", yaxis_title="Avg Risk Score", margin=dict(t=10))
    st.subheader("Average Risk Score Over Time")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["anxiety_rate"]*100,
            name="Anxiety", line=dict(color=C["red"], width=2)))
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["depression_rate"]*100,
            name="Depression", line=dict(color=C["purple"], width=2)))
        fig.update_layout(template="plotly_white", height=300,
            yaxis_title="Rate (%)", legend=dict(orientation="h",y=-0.3), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["year"], y=yearly["help_rate"]*100,
            line=dict(color=C["green"], width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"))
        fig.update_layout(template="plotly_white", height=300,
            yaxis_title="Help-Seeking Rate (%)", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Trends by Country")
    cy = df.groupby(["year","country"])["mental_health_risk_score"].mean().reset_index()
    fig = px.line(cy, x="year", y="mental_health_risk_score", color="country",
        markers=True, color_discrete_map=COUNTRY_COLORS,
        labels={"mental_health_risk_score":"Avg Risk Score"}, template="plotly_white")
    fig.update_layout(height=350, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    st.title("Demographics Analysis")
    st.markdown("---")

    age_order = ["11-12","12-13","13-14","14-15","15-16","16-17","17-18",
                 "18-24","25-34","35-44","45-54","55-64","65+"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Score by Age Group")
        adf = df.groupby("age_group")["mental_health_risk_score"].mean().reset_index()
        adf["order"] = adf["age_group"].map({a:i for i,a in enumerate(age_order)})
        adf = adf.sort_values("order")
        fig = px.bar(adf, x="age_group", y="mental_health_risk_score",
            color="mental_health_risk_score",
            color_continuous_scale=[[0,"#CCFBF1"],[0.5,C["teal"]],[1,C["navy"]]],
            labels={"mental_health_risk_score":"Avg Risk Score","age_group":"Age Group"},
            template="plotly_white")
        fig.update_layout(height=350, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("High Risk Rate by Gender")
        gdf = df.groupby("gender")["high_risk"].mean().reset_index()
        gdf["high_risk"] = (gdf["high_risk"]*100).round(1)
        fig = px.bar(gdf, x="gender", y="high_risk", color="gender",
            color_discrete_sequence=[C["teal"],C["purple"],C["amber"],C["orange"]],
            labels={"high_risk":"High Risk Rate (%)","gender":""}, template="plotly_white")
        fig.update_layout(height=350, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Rate Heatmap: Age × Year")
    pivot = df.groupby(["age_group","year"])["high_risk"].mean().unstack()*100
    pivot = pivot.reindex([a for a in age_order if a in pivot.index])
    fig = px.imshow(pivot,
        color_continuous_scale=[[0,"#F0FDFA"],[0.5,C["teal"]],[1,"#0F766E"]],
        labels={"color":"High Risk %"}, aspect="auto", template="plotly_white")
    fig.update_layout(height=420, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# LIFESTYLE FACTORS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "😴 Lifestyle Factors":
    st.title("Lifestyle Factor Analysis")
    st.markdown("---")

    df_s = df.copy()
    df_s["sleep_bucket"] = pd.cut(df_s["sleep_hours"],
        bins=[0,5,6,7,8,9,12], labels=["<5h","5-6h","6-7h","7-8h","8-9h","9h+"])
    sa = df_s.groupby("sleep_bucket", observed=True).agg(
        avg_risk=("mental_health_risk_score","mean"),
        high_risk_rate=("high_risk","mean"),
    ).reset_index()

    st.subheader("Sleep Hours vs Mental Health Risk")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(sa, x="sleep_bucket", y="avg_risk", color="avg_risk",
            color_continuous_scale=[[0,C["teal"]],[1,C["red"]]],
            labels={"avg_risk":"Avg Risk Score","sleep_bucket":"Sleep"}, template="plotly_white")
        fig.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(sa, x="sleep_bucket", y="high_risk_rate", color="high_risk_rate",
            color_continuous_scale=[[0,C["teal"]],[1,C["red"]]],
            labels={"high_risk_rate":"High Risk Rate","sleep_bucket":"Sleep"}, template="plotly_white")
        fig.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    lv = sa[sa["sleep_bucket"]=="<5h"]["avg_risk"].values
    hv = sa[sa["sleep_bucket"]=="7-8h"]["avg_risk"].values
    if len(lv) and len(hv):
        st.info(f"💡 People sleeping <5h score **{lv[0]:.1f}** vs **{hv[0]:.1f}** for 7-8h (p<0.0001)")

    st.subheader("Social Support Level vs Risk")
    sup_order = ["Never","Rarely","Sometimes","Usually","Always"]
    sup = df.groupby("social_support")["mental_health_risk_score"].mean().reset_index()
    sup["order"] = sup["social_support"].map({s:i for i,s in enumerate(sup_order)})
    sup = sup.sort_values("order")
    fig = px.line(sup, x="social_support", y="mental_health_risk_score", markers=True,
        color_discrete_sequence=[C["navy"]],
        labels={"mental_health_risk_score":"Avg Risk Score","social_support":"Support Level"},
        template="plotly_white")
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# COUNTRY COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Country Comparison":
    st.title("USA · Canada · Pakistan Comparison")
    st.markdown("How youth mental health differs across three countries with very different contexts")
    st.markdown("---")

    cs = df.groupby("country").agg(
        records=("year","count"),
        avg_risk=("mental_health_risk_score","mean"),
        high_risk_rate=("high_risk","mean"),
        anxiety_rate=("anxiety_diagnosis","mean"),
        depression_rate=("depression_diagnosis","mean"),
        help_seeking_rate=("sought_help","mean"),
        avg_sleep=("sleep_hours","mean"),
        avg_activity=("activity_days_per_week","mean"),
    ).round(3).reset_index()

    disp = cs.copy()
    for col in ["high_risk_rate","anxiety_rate","depression_rate","help_seeking_rate"]:
        disp[col] = (disp[col]*100).round(1).astype(str) + "%"
    disp["avg_risk"] = disp["avg_risk"].round(1)
    disp["avg_sleep"] = disp["avg_sleep"].round(1)
    disp.columns = ["Country","Records","Avg Risk","High Risk","Anxiety","Depression",
                    "Help-Seeking","Avg Sleep (h)","Avg Activity (d/wk)"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("---")
    metrics = ["avg_risk","anxiety_rate","depression_rate","help_seeking_rate"]
    labels  = ["Avg Risk Score","Anxiety Rate","Depression Rate","Help-Seeking Rate"]
    c1, c2  = st.columns(2)
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        col = c1 if i%2==0 else c2
        with col:
            v = cs[["country",m]].copy()
            if m != "avg_risk": v[m] = v[m]*100
            fig = px.bar(v, x="country", y=m, color="country",
                color_discrete_map=COUNTRY_COLORS, title=lab,
                labels={m:lab,"country":""}, template="plotly_white")
            fig.update_layout(height=280, margin=dict(t=40), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Pakistan Context")
    st.markdown("""
    The Pakistan data is modelled from published peer-reviewed research:

    - **53% of high school students** experience anxiety/depression *(Ghazal, 2022)*
    - **17.2% depression, 21.4% anxiety** in ages 11–18 *(Khalid et al., 2019)*
    - Girls significantly more anxious than boys across all studies
    - **Help-seeking is ~15%** vs ~40% in North America — driven by cultural stigma
    - National Psychiatric Morbidity Survey (2022) — first nationally representative study

    This is why platforms like **Sandbox Pakistan** matter — the gap between need and help-seeking is massive.
    """)


# ════════════════════════════════════════════════════════════════════════════════
# RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.title("Individual Risk Predictor")
    st.markdown("Predict mental health risk using the trained Random Forest model (AUC=0.925)")
    st.markdown("---")

    model = load_model()
    if model is None:
        st.warning("⚠️ Model not found. Run: `python -m src.ml.train`")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Demographics")
            age_group = st.selectbox("Age Group", [
                "11-12","12-13","13-14","14-15","15-16","16-17","17-18",
                "18-24","25-34","35-44","45-54","55-64","65+"], index=7)
            gender  = st.selectbox("Gender", ["Male","Female","Non-binary"])
            country = st.selectbox("Country", ["USA","Canada","Pakistan"])
        with c2:
            st.subheader("Lifestyle")
            sleep_h  = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5)
            act_days = st.slider("Activity Days/Week", 0, 7, 3)
            soc_sup  = st.selectbox("Social Support",
                ["Never","Rarely","Sometimes","Usually","Always"], index=2)
        with c3:
            st.subheader("Mental Health History")
            anxiety    = st.checkbox("Anxiety diagnosis")
            depression = st.checkbox("Depression diagnosis")
            s_help     = st.checkbox("Has sought professional help")

        st.markdown("---")
        if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
            profile = {
                "age_group": age_group, "gender": gender,
                "sleep_hours": sleep_h, "activity_days_per_week": act_days,
                "anxiety_diagnosis": int(anxiety), "depression_diagnosis": int(depression),
                "sought_help": int(s_help), "social_support": soc_sup,
                "country": country, "year": 2023, "source": "CDC_BRFSS",
            }
            res = predict_single(model, profile)

            cr1, cr2, cr3 = st.columns(3)
            cr1.metric("Risk Level", res["risk_level"])
            cr2.metric("Probability", f"{res['probability']:.1%}")
            cr3.metric("Prediction", "High Risk" if res["high_risk"] else "Low Risk")
            st.progress(res["probability"])

            if res["high_risk"]:
                st.error("⚠️ Elevated risk detected. Please reach out to a mental health professional.")
                if country == "Pakistan":
                    st.info("🇵🇰 Pakistan resources: Umang helpline 0317-4288665 · Rozan Counselling 051-2890505")
                else:
                    st.info("🆘 Crisis resources: Call/text 988 (USA/Canada) · Text HOME to 741741")
            else:
                st.success("✅ Low risk based on the provided factors.")


# ════════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Tests":
    st.title("Statistical Significance Testing")
    st.markdown("---")

    from scipy import stats
    tests = []

    for label, g1n, g2n, col in [
        ("Gender: Female vs Male", "Female", "Male", "gender"),
        ("Country: Pakistan vs USA", "Pakistan", "USA", "country"),
        ("Country: Canada vs USA", "Canada", "USA", "country"),
    ]:
        g1 = df[df[col]==g1n]["mental_health_risk_score"]
        g2 = df[df[col]==g2n]["mental_health_risk_score"]
        if len(g1) > 0 and len(g2) > 0:
            t, p = stats.ttest_ind(g1, g2)
            tests.append({"Test":label, "Method":"t-test",
                "Statistic":f"t={t:.3f}", "p-value":f"{p:.4f}",
                "Significant":"✅ Yes" if p<0.05 else "❌ No",
                "Finding":f"{g1n} ({g1.mean():.2f}) vs {g2n} ({g2.mean():.2f})"})

    ls = df[df["sleep_hours"]<6]["mental_health_risk_score"]
    gs = df[df["sleep_hours"]>=7]["mental_health_risk_score"]
    if len(ls) and len(gs):
        t, p = stats.ttest_ind(ls, gs)
        tests.append({"Test":"Sleep: <6h vs ≥7h","Method":"t-test",
            "Statistic":f"t={t:.3f}","p-value":f"{p:.2e}",
            "Significant":"✅ Yes" if p<0.05 else "❌ No",
            "Finding":f"<6h ({ls.mean():.2f}) vs ≥7h ({gs.mean():.2f})"})

    ags = [g for _,g in df.groupby("age_group")["mental_health_risk_score"]]
    if len(ags) > 1:
        f, p = stats.f_oneway(*ags)
        tests.append({"Test":"Age Group Differences","Method":"ANOVA",
            "Statistic":f"F={f:.3f}","p-value":f"{p:.2e}",
            "Significant":"✅ Yes" if p<0.05 else "❌ No",
            "Finding":f"{len(ags)} groups compared"})

    st.dataframe(pd.DataFrame(tests), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Feature Correlation Matrix")
    nc = ["mental_health_risk_score","sleep_hours","activity_days_per_week",
          "anxiety_diagnosis","depression_diagnosis","sought_help"]
    corr = df[nc].corr().round(3)
    fig = px.imshow(corr,
        color_continuous_scale=[[0,"#FEF2F2"],[0.5,"white"],[1,"#F0FDFA"]],
        text_auto=True, template="plotly_white")
    fig.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
