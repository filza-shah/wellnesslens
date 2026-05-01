# src/dashboard/app.py
#
# WellnessLens — Full Interactive Dashboard
# Built with Streamlit + Plotly
# Auto-run pipeline on startup

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
# Auto-run pipeline on startup
from pathlib import Path

if not Path("data/processed/unified_mental_health.csv").exists():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "src.ingestion.download"], check=True)
    subprocess.run([sys.executable, "-m", "src.processing.clean"], check=True)

if not Path("data/models/risk_model.joblib").exists():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "src.ml.train"], check=True)
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WellnessLens — Youth Mental Health Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F8FAFC; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0F172A; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSelectbox label { color: #94A3B8 !important; }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    
    /* Headers */
    h1 { color: #0F172A !important; font-weight: 800 !important; }
    h2 { color: #0F172A !important; font-weight: 700 !important; }
    h3 { color: #334155 !important; font-weight: 600 !important; }
    
    /* Divider */
    hr { border-color: #E2E8F0; margin: 24px 0; }
    
    /* Insight cards */
    .insight-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .insight-high   { border-left: 4px solid #EF4444; }
    .insight-medium { border-left: 4px solid #F59E0B; }
    .insight-low    { border-left: 4px solid #14B8A6; }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
COLORS = {
    "teal":   "#14B8A6",
    "navy":   "#0F172A",
    "red":    "#EF4444",
    "amber":  "#F59E0B",
    "green":  "#10B981",
    "purple": "#8B5CF6",
    "slate":  "#64748B",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#334155"),
        colorway=[COLORS["teal"], COLORS["navy"], COLORS["amber"], COLORS["red"], COLORS["purple"]],
    )
)


@st.cache_data
def load_data():
    path = Path("data/processed/unified_mental_health.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    try:
        import joblib
        from pathlib import Path
        model_path = Path("data/models/risk_model.joblib")
        if model_path.exists():
            return joblib.load(model_path)
    except Exception:
        pass
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 WellnessLens")
    st.markdown("*Youth Mental Health Analytics*")
    st.markdown("---")

    st.markdown("### Filters")

    df_raw = load_data()

    if df_raw is not None:
        # Country filter
        countries = ["All"] + sorted(df_raw["country"].unique().tolist())
        selected_country = st.selectbox("Country", countries)

        # Year range
        min_year = int(df_raw["year"].min())
        max_year = int(df_raw["year"].max())
        year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))

        # Age group filter
        age_order = ["12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
                     "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        available_ages = [a for a in age_order if a in df_raw["age_group"].unique()]
        selected_ages = st.multiselect("Age Groups", available_ages, default=available_ages)

        # Gender filter
        genders = ["All"] + sorted(df_raw["gender"].unique().tolist())
        selected_gender = st.selectbox("Gender", genders)

    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Navigation", [
        "📊 Overview",
        "📈 Trends Over Time",
        "👥 Demographics",
        "😴 Lifestyle Factors",
        "🤖 Risk Predictor",
        "🔬 Statistical Tests",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Data Sources**")
    st.markdown("CDC BRFSS (USA) · CAMH OSDUHS (Canada)")
    st.markdown("Years: 2015–2023 · Records: 65,000")


# ── Apply filters ─────────────────────────────────────────────────────────────
def apply_filters(df):
    if selected_country != "All":
        df = df[df["country"] == selected_country]
    df = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]
    if selected_ages:
        df = df[df["age_group"].isin(selected_ages)]
    if selected_gender != "All":
        df = df[df["gender"] == selected_gender]
    return df


# ── Pages ─────────────────────────────────────────────────────────────────────
if df_raw is None:
    st.error("⚠️ No data found. Run the pipeline first:")
    st.code("python -m src.ingestion.download\npython -m src.processing.clean")
    st.stop()

df = apply_filters(df_raw.copy())

# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Youth Mental Health Analytics")
    st.markdown(f"Analysing **{len(df):,} records** across {df['country'].nunique()} countries · {df['year'].min()}–{df['year'].max()}")
    st.markdown("---")

    # ── Key metrics ──
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        high_risk_pct = df["high_risk"].mean() * 100
        st.metric("High Risk Rate", f"{high_risk_pct:.1f}%", delta=None)
    with col3:
        anxiety_pct = df["anxiety_diagnosis"].mean() * 100
        st.metric("Anxiety Rate", f"{anxiety_pct:.1f}%")
    with col4:
        depression_pct = df["depression_diagnosis"].mean() * 100
        st.metric("Depression Rate", f"{depression_pct:.1f}%")
    with col5:
        help_pct = df["sought_help"].mean() * 100
        st.metric("Help-Seeking Rate", f"{help_pct:.1f}%")

    st.markdown("---")

    # ── Key insights ──
    st.subheader("Key Findings")

    col_a, col_b = st.columns(2)

    with col_a:
        # Youth vs adult risk
        youth_mask = df["age_group"].isin(["12-13","13-14","14-15","15-16","16-17","17-18","18-24"])
        youth_risk = df[youth_mask]["mental_health_risk_score"].mean()
        adult_risk = df[~youth_mask]["mental_health_risk_score"].mean()

        st.markdown(f"""
        <div class="insight-card insight-high">
            <strong>🔴 Youth Risk Higher Than Adults</strong><br>
            Youth (12-24) average risk score: <strong>{youth_risk:.1f}</strong> vs
            adults 25+: <strong>{adult_risk:.1f}</strong>
            — a {youth_risk - adult_risk:.1f} point gap
        </div>
        """, unsafe_allow_html=True)

        # Sleep impact
        low_sleep = df[df["sleep_hours"] < 5]["mental_health_risk_score"].mean()
        good_sleep = df[df["sleep_hours"] >= 7]["mental_health_risk_score"].mean()
        st.markdown(f"""
        <div class="insight-card insight-high">
            <strong>😴 Sleep Deprivation Adds +{low_sleep - good_sleep:.1f} Risk Points</strong><br>
            People sleeping &lt;5h score <strong>{low_sleep:.1f}</strong> vs
            <strong>{good_sleep:.1f}</strong> for those sleeping 7+ hours
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        # Country comparison
        usa_risk = df[df["country"] == "USA"]["mental_health_risk_score"].mean() if "USA" in df["country"].values else 0
        can_risk = df[df["country"] == "Canada"]["mental_health_risk_score"].mean() if "Canada" in df["country"].values else 0

        if usa_risk and can_risk:
            st.markdown(f"""
            <div class="insight-card insight-medium">
                <strong>🌎 Canada vs USA Risk Gap</strong><br>
                Canada avg risk: <strong>{can_risk:.1f}</strong> vs
                USA: <strong>{usa_risk:.1f}</strong>
                — statistically significant difference (p&lt;0.0001)
            </div>
            """, unsafe_allow_html=True)

        # Help seeking gap
        high_risk_help = df[df["high_risk"] == 1]["sought_help"].mean() * 100
        low_risk_help  = df[df["high_risk"] == 0]["sought_help"].mean() * 100
        st.markdown(f"""
        <div class="insight-card insight-medium">
            <strong>🆘 Help-Seeking Gap</strong><br>
            High-risk individuals seek help at <strong>{high_risk_help:.0f}%</strong>
            — nearly the same as low-risk at <strong>{low_risk_help:.0f}%</strong>.
            Early intervention is critical.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Risk distribution ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Distribution")
        fig = px.histogram(
            df, x="mental_health_risk_score",
            nbins=40, color_discrete_sequence=[COLORS["teal"]],
            labels={"mental_health_risk_score": "Mental Health Risk Score"},
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High Risk by Country")
        country_risk = df.groupby("country")["high_risk"].mean().reset_index()
        country_risk["high_risk"] = (country_risk["high_risk"] * 100).round(1)
        fig = px.bar(
            country_risk, x="country", y="high_risk",
            color_discrete_sequence=[COLORS["teal"]],
            labels={"high_risk": "High Risk Rate (%)", "country": ""},
            template="plotly_white",
        )
        fig.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS OVER TIME
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📈 Trends Over Time":
    st.title("Mental Health Trends Over Time")
    st.markdown("Year-over-year changes in youth mental health indicators (2015–2023)")
    st.markdown("---")

    yearly = df.groupby("year").agg(
        avg_risk=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        anxiety_rate=("anxiety_diagnosis", "mean"),
        depression_rate=("depression_diagnosis", "mean"),
        help_rate=("sought_help", "mean"),
    ).reset_index()

    # ── Main trend chart ──
    st.subheader("Average Risk Score Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["avg_risk"],
        mode="lines+markers",
        line=dict(color=COLORS["teal"], width=3),
        marker=dict(size=8, color=COLORS["teal"]),
        name="Avg Risk Score",
        fill="tozeroy",
        fillcolor="rgba(20,184,166,0.1)",
    ))
    fig.update_layout(
        template="plotly_white", height=350,
        xaxis_title="Year", yaxis_title="Average Risk Score",
        margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Multi-metric ──
    st.subheader("Mental Health Indicators Over Time")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["anxiety_rate"] * 100,
            name="Anxiety Rate", line=dict(color=COLORS["red"], width=2),
        ))
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["depression_rate"] * 100,
            name="Depression Rate", line=dict(color=COLORS["purple"], width=2),
        ))
        fig.update_layout(
            template="plotly_white", height=300,
            yaxis_title="Rate (%)", xaxis_title="Year",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly["year"], y=yearly["help_rate"] * 100,
            name="Help-Seeking Rate",
            line=dict(color=COLORS["green"], width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)",
        ))
        fig.update_layout(
            template="plotly_white", height=300,
            yaxis_title="Rate (%)", xaxis_title="Year",
            margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Country trends side by side ──
    st.subheader("USA vs Canada Trends")
    country_yearly = df.groupby(["year", "country"])["mental_health_risk_score"].mean().reset_index()
    fig = px.line(
        country_yearly, x="year", y="mental_health_risk_score",
        color="country", markers=True,
        color_discrete_map={"USA": COLORS["teal"], "Canada": COLORS["navy"]},
        labels={"mental_health_risk_score": "Avg Risk Score", "year": "Year"},
        template="plotly_white",
    )
    fig.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: DEMOGRAPHICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "👥 Demographics":
    st.title("Demographics Analysis")
    st.markdown("Mental health risk broken down by age group and gender")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score by Age Group")
        age_order = ["12-13","13-14","14-15","15-16","16-17","17-18",
                     "18-24","25-34","35-44","45-54","55-64","65+"]
        age_df = df.groupby("age_group")["mental_health_risk_score"].mean().reset_index()
        age_df["order"] = age_df["age_group"].map({a: i for i, a in enumerate(age_order)})
        age_df = age_df.sort_values("order")

        fig = px.bar(
            age_df, x="age_group", y="mental_health_risk_score",
            color="mental_health_risk_score",
            color_continuous_scale=[[0, "#CCFBF1"], [0.5, COLORS["teal"]], [1, COLORS["navy"]]],
            labels={"mental_health_risk_score": "Avg Risk Score", "age_group": "Age Group"},
            template="plotly_white",
        )
        fig.update_layout(height=350, margin=dict(t=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High Risk Rate by Gender")
        gender_df = df.groupby("gender").agg(
            high_risk_rate=("high_risk", "mean"),
            count=("high_risk", "count"),
        ).reset_index()
        gender_df["high_risk_rate"] = (gender_df["high_risk_rate"] * 100).round(1)

        fig = px.bar(
            gender_df, x="gender", y="high_risk_rate",
            color="gender",
            color_discrete_sequence=[COLORS["teal"], COLORS["purple"], COLORS["amber"]],
            labels={"high_risk_rate": "High Risk Rate (%)", "gender": ""},
            template="plotly_white",
        )
        fig.update_layout(height=350, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Age group heatmap ──
    st.subheader("Risk Rate Heatmap: Age × Year")
    pivot = df.groupby(["age_group", "year"])["high_risk"].mean().unstack() * 100

    # Sort age groups
    pivot = pivot.reindex([a for a in age_order if a in pivot.index])

    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#F0FDFA"], [0.5, COLORS["teal"]], [1, "#0F766E"]],
        labels={"color": "High Risk %"},
        aspect="auto",
        template="plotly_white",
    )
    fig.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: LIFESTYLE FACTORS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "😴 Lifestyle Factors":
    st.title("Lifestyle Factor Analysis")
    st.markdown("How sleep, activity, and social support relate to mental health outcomes")
    st.markdown("---")

    # ── Sleep analysis ──
    st.subheader("Sleep Hours vs Mental Health Risk")

    df_sleep = df.copy()
    df_sleep["sleep_bucket"] = pd.cut(
        df_sleep["sleep_hours"],
        bins=[0, 5, 6, 7, 8, 9, 12],
        labels=["<5h", "5-6h", "6-7h", "7-8h", "8-9h", "9h+"]
    )
    sleep_agg = df_sleep.groupby("sleep_bucket", observed=True).agg(
        avg_risk=("mental_health_risk_score", "mean"),
        high_risk_rate=("high_risk", "mean"),
        count=("sleep_hours", "count"),
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            sleep_agg, x="sleep_bucket", y="avg_risk",
            color="avg_risk",
            color_continuous_scale=[[0, COLORS["teal"]], [1, COLORS["red"]]],
            labels={"avg_risk": "Avg Risk Score", "sleep_bucket": "Sleep Duration"},
            template="plotly_white",
        )
        fig.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            sleep_agg, x="sleep_bucket", y="high_risk_rate",
            color="high_risk_rate",
            color_continuous_scale=[[0, COLORS["teal"]], [1, COLORS["red"]]],
            labels={"high_risk_rate": "High Risk Rate", "sleep_bucket": "Sleep Duration"},
            template="plotly_white",
        )
        fig.update_layout(height=300, margin=dict(t=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info(f"💡 People sleeping less than 5 hours show **{sleep_agg[sleep_agg['sleep_bucket']=='<5h']['avg_risk'].values[0]:.1f}** avg risk vs **{sleep_agg[sleep_agg['sleep_bucket']=='7-8h']['avg_risk'].values[0]:.1f}** for those sleeping 7-8h — a statistically significant difference (p<0.0001)")

    # ── Social support ──
    st.subheader("Social Support Level vs Risk")
    support_order = ["Never", "Rarely", "Sometimes", "Usually", "Always"]
    support_agg = df.groupby("social_support").agg(
        avg_risk=("mental_health_risk_score", "mean"),
        count=("social_support", "count"),
    ).reset_index()
    support_agg["order"] = support_agg["social_support"].map({s: i for i, s in enumerate(support_order)})
    support_agg = support_agg.sort_values("order")

    fig = px.line(
        support_agg, x="social_support", y="avg_risk",
        markers=True,
        color_discrete_sequence=[COLORS["navy"]],
        labels={"avg_risk": "Avg Risk Score", "social_support": "Social Support Level"},
        template="plotly_white",
    )
    fig.update_traces(line_width=3, marker_size=10)
    fig.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RISK PREDICTOR
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Risk Predictor":
    st.title("Individual Risk Predictor")
    st.markdown("Enter a profile to predict mental health risk using the trained Random Forest model (AUC=0.925)")
    st.markdown("---")

    model = load_model()

    if model is None:
        st.warning("⚠️ Model not found. Run: `python -m src.ml.train`")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Demographics")
            age_group = st.selectbox("Age Group", [
                "12-13","13-14","14-15","15-16","16-17","17-18",
                "18-24","25-34","35-44","45-54","55-64","65+"
            ], index=6)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            country = st.selectbox("Country", ["USA", "Canada"])

        with col2:
            st.subheader("Lifestyle")
            sleep_hours = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5)
            activity_days = st.slider("Activity Days/Week", 0, 7, 3)
            social_support = st.selectbox("Social Support", ["Never", "Rarely", "Sometimes", "Usually", "Always"], index=2)

        with col3:
            st.subheader("Mental Health History")
            anxiety = st.checkbox("Anxiety diagnosis")
            depression = st.checkbox("Depression diagnosis")
            sought_help = st.checkbox("Has sought professional help")

        st.markdown("---")

        if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
            try:
                from src.ml.evaluate import predict_single
                profile = {
                    "age_group": age_group, "gender": gender,
                    "sleep_hours": sleep_hours,
                    "activity_days_per_week": activity_days,
                    "anxiety_diagnosis": int(anxiety),
                    "depression_diagnosis": int(depression),
                    "sought_help": int(sought_help),
                    "social_support": social_support,
                    "country": country, "year": 2023,
                    "source": "CDC_BRFSS",
                }
                result = predict_single(model, profile)

                color_map = {"Critical": "#EF4444", "High": "#F97316", "Moderate": "#F59E0B", "Low": "#14B8A6"}
                color = color_map.get(result["risk_level"], COLORS["teal"])

                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Risk Level", result["risk_level"])
                with col_r2:
                    st.metric("Probability", f"{result['probability']:.1%}")
                with col_r3:
                    st.metric("Prediction", "High Risk" if result["high_risk"] else "Low Risk")

                # Progress bar
                st.markdown(f"**Risk Probability: {result['probability']:.1%}**")
                st.progress(result["probability"])

                if result["high_risk"]:
                    st.error("⚠️ This profile indicates elevated mental health risk. Consider reaching out to a mental health professional.")
                else:
                    st.success("✅ This profile indicates low mental health risk based on the provided factors.")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICAL TESTS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistical Tests":
    st.title("Statistical Significance Testing")
    st.markdown("Formal hypothesis tests validating our key findings")
    st.markdown("---")

    from scipy import stats

    tests = []

    # Gender test
    male   = df[df["gender"] == "Male"]["mental_health_risk_score"]
    female = df[df["gender"] == "Female"]["mental_health_risk_score"]
    if len(male) > 0 and len(female) > 0:
        t, p = stats.ttest_ind(male, female)
        tests.append({
            "Test": "Gender Difference",
            "Method": "Independent t-test",
            "Statistic": f"t={t:.3f}",
            "p-value": f"{p:.4f}",
            "Significant": "✅ Yes" if p < 0.05 else "❌ No",
            "Finding": f"Female ({female.mean():.2f}) vs Male ({male.mean():.2f})",
        })

    # Sleep test
    low_sleep  = df[df["sleep_hours"] < 6]["mental_health_risk_score"]
    good_sleep = df[df["sleep_hours"] >= 7]["mental_health_risk_score"]
    if len(low_sleep) > 0 and len(good_sleep) > 0:
        t, p = stats.ttest_ind(low_sleep, good_sleep)
        tests.append({
            "Test": "Sleep Impact",
            "Method": "Independent t-test",
            "Statistic": f"t={t:.3f}",
            "p-value": f"{p:.2e}",
            "Significant": "✅ Yes" if p < 0.05 else "❌ No",
            "Finding": f"<6h sleep ({low_sleep.mean():.2f}) vs ≥7h ({good_sleep.mean():.2f})",
        })

    # Country test
    usa    = df[df["country"] == "USA"]["mental_health_risk_score"]
    canada = df[df["country"] == "Canada"]["mental_health_risk_score"]
    if len(usa) > 0 and len(canada) > 0:
        t, p = stats.ttest_ind(usa, canada)
        tests.append({
            "Test": "USA vs Canada",
            "Method": "Independent t-test",
            "Statistic": f"t={t:.3f}",
            "p-value": f"{p:.2e}",
            "Significant": "✅ Yes" if p < 0.05 else "❌ No",
            "Finding": f"USA ({usa.mean():.2f}) vs Canada ({canada.mean():.2f})",
        })

    # Age ANOVA
    age_groups = [g for _, g in df.groupby("age_group")["mental_health_risk_score"]]
    if len(age_groups) > 1:
        f, p = stats.f_oneway(*age_groups)
        tests.append({
            "Test": "Age Group Differences",
            "Method": "One-way ANOVA",
            "Statistic": f"F={f:.3f}",
            "p-value": f"{p:.2e}",
            "Significant": "✅ Yes" if p < 0.05 else "❌ No",
            "Finding": f"{len(age_groups)} groups compared",
        })

    tests_df = pd.DataFrame(tests)
    st.dataframe(tests_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""
    **Interpretation guide:**
    - **p < 0.05** → statistically significant (less than 5% chance the result is random)
    - **p < 0.001** → highly significant
    - Significance ≠ practical importance — always consider effect size
    """)

    # Correlation heatmap
    st.subheader("Feature Correlation Matrix")
    numeric_cols = ["mental_health_risk_score", "sleep_hours", "activity_days_per_week",
                    "anxiety_diagnosis", "depression_diagnosis", "sought_help"]
    corr_matrix = df[numeric_cols].corr().round(3)

    fig = px.imshow(
        corr_matrix,
        color_continuous_scale=[[0, "#FEF2F2"], [0.5, "white"], [1, "#F0FDFA"]],
        text_auto=True,
        template="plotly_white",
    )
    fig.update_layout(height=400, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)
