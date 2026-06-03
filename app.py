import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import joblib
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tea Garden Climate Risk Analyzer",
    page_icon="🍵",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 12})

# ─────────────────────────────────────────────
# Load Models
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "Model"
    stress_model = joblib.load(os.path.join(base, "tea_stress_model.pkl"))
    risk_model   = joblib.load(os.path.join(base, "tea_risk_model.pkl"))
    le_stress    = joblib.load(os.path.join(base, "le_stress.pkl"))
    le_risk      = joblib.load(os.path.join(base, "le_risk.pkl"))
    feat_path    = os.path.join(base, "feature_names.pkl")
    features     = joblib.load(feat_path) if os.path.exists(feat_path) else [
        "avg_temperature","rainfall","humidity",
        "wind_speed","cloud_cover","aqi","month","week"
    ]
    return stress_model, risk_model, le_stress, le_risk, features

stress_model, risk_model, le_stress, le_risk, MODEL_FEATURES = load_artifacts()

# ─────────────────────────────────────────────
# Sidebar — Climate Inputs
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🌦️ Climate Inputs")
    st.markdown("Adjust sliders to match current field conditions.")

    avg_temperature = st.slider("🌡️ Avg Temperature (°C)", 5.0,  45.0, 26.0, 0.5)
    rainfall        = st.slider("🌧️ Rainfall (mm)",         0.0, 300.0, 80.0, 1.0)
    humidity        = st.slider("💧 Humidity (%)",          10.0, 100.0, 70.0, 1.0)
    wind_speed      = st.slider("💨 Wind Speed (km/h)",      0.0,  50.0, 10.0, 0.5)
    cloud_cover     = st.slider("☁️ Cloud Cover (%)",         0.0, 100.0, 50.0, 1.0)
    aqi             = st.slider("🏭 AQI",                      0,   500,  100,   5)
    month = st.selectbox(
        "📅 Month", list(range(1, 13)), index=4,
        format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"][x - 1]
    )
    week = st.slider("📆 Week of Year", 1, 52, 20)
    st.markdown("---")
    st.caption("Trained on Kaggle Indian Climate Dataset 2024–25")

# ─────────────────────────────────────────────
# Build Input Row
# ─────────────────────────────────────────────
_base_vals = {
    "avg_temperature": avg_temperature,
    "rainfall":        rainfall,
    "humidity":        humidity,
    "wind_speed":      wind_speed,
    "cloud_cover":     cloud_cover,
    "aqi":             aqi,
    "month":           month,
    "week":            week,
    "month_sin":       np.sin(2 * np.pi * month / 12),
    "month_cos":       np.cos(2 * np.pi * month / 12),
}
input_df = pd.DataFrame(
    [[_base_vals.get(f, 0) for f in MODEL_FEATURES]],
    columns=MODEL_FEATURES
)

# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────
stress_enc   = stress_model.predict(input_df)[0]
risk_enc     = risk_model.predict(input_df)[0]
stress_label = le_stress.inverse_transform([stress_enc])[0]
risk_label   = le_risk.inverse_transform([risk_enc])[0]
stress_proba = stress_model.predict_proba(input_df)[0]
risk_proba   = risk_model.predict_proba(input_df)[0]
stress_conf  = stress_proba.max()
risk_conf    = risk_proba.max()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;'>🍵 Tea Garden Climate Risk Analyzer</h1>"
    "<p style='text-align:center;color:grey;'>Early-warning system · ML + rule-based engine · "
    "Indian Climate Dataset 2024–25</p>",
    unsafe_allow_html=True
)
st.markdown("---")

tab_pred, tab_eda, tab_about = st.tabs([
    "🔮 Prediction & Alerts",
    "📊 Dataset Explorer",
    "ℹ️ About"
])

# ══════════════════════════════════════════════
# TAB 1 — PREDICTION & ALERTS
# ══════════════════════════════════════════════
with tab_pred:

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{avg_temperature} °C")
    col2.metric("🌧️ Rainfall",    f"{rainfall} mm")
    col3.metric("💧 Humidity",    f"{humidity} %")
    col4.metric("🏭 AQI",         str(aqi))

    st.markdown("---")
    left, right = st.columns(2)

    # Stress Level
    with left:
        st.subheader("🌿 Plant Stress Level")
        if stress_label == "Severe Stress":
            st.error(f"🚨 **{stress_label}** — Confidence: {stress_conf:.0%}")
        elif stress_label == "Mild Stress":
            st.warning(f"⚠️ **{stress_label}** — Confidence: {stress_conf:.0%}")
        else:
            st.success(f"✅ **{stress_label}** — Confidence: {stress_conf:.0%}")

        fig_s, ax_s = plt.subplots(figsize=(4, 2.2))
        bars_s = ax_s.barh(le_stress.classes_, stress_proba,
                           color=["#2ca02c","#ff7f0e","#d62728"])
        ax_s.set_xlim(0, 1)
        ax_s.set_xlabel("Probability")
        ax_s.set_title("Stress Probability Breakdown")
        for bar, p in zip(bars_s, stress_proba):
            ax_s.text(bar.get_width() + 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"{p:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)

    # Risk Level
    with right:
        st.subheader("⚠️ Climate Risk Level")
        risk_map = {
            "Low":      ("success", "✅"),
            "Moderate": ("warning", "⚠️"),
            "High":     ("error",   "🔴"),
            "Extreme":  ("error",   "🚨"),
        }
        fn, icon = risk_map.get(risk_label, ("info", "ℹ️"))
        getattr(st, fn)(f"{icon} **{risk_label} Risk** — Confidence: {risk_conf:.0%}")

        fig_r, ax_r = plt.subplots(figsize=(4, 2.6))
        palette_r = ["#2ca02c","#ff7f0e","#d62728","#8b0000"]
        bars_r = ax_r.barh(le_risk.classes_, risk_proba,
                           color=palette_r[:len(le_risk.classes_)])
        ax_r.set_xlim(0, 1)
        ax_r.set_xlabel("Probability")
        ax_r.set_title("Risk Probability Breakdown")
        for bar, p in zip(bars_r, risk_proba):
            ax_r.text(bar.get_width() + 0.01,
                      bar.get_y() + bar.get_height() / 2,
                      f"{p:.2f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close(fig_r)

    st.markdown("---")

    # Recommendations
    st.subheader("📋 Recommended Actions")
    RECS = {
        "Healthy": {
            "color": "success",
            "actions": [
                "✅ Climate conditions are favorable for tea cultivation.",
                "🌿 Continue regular plantation practices — no intervention needed.",
                "📋 Perform weekly field monitoring as standard practice.",
                "🌱 Good window for pruning, fertilization, or new planting.",
            ]
        },
        "Mild Stress": {
            "color": "warning",
            "actions": [
                "💧 Apply light supplemental irrigation if rainfall drops further.",
                "🔍 Monitor for early signs of pest and fungal activity.",
                "🌿 Avoid heavy pruning until conditions stabilize.",
                "📊 Increase monitoring frequency to every 3–4 days.",
                "🧪 Consider foliar spray to support leaf health.",
            ]
        },
        "Severe Stress": {
            "color": "error",
            "actions": [
                "🚿 Activate emergency irrigation immediately.",
                "🚫 Halt all pruning, harvesting, and transplanting operations.",
                "🌾 Apply thick mulch layer to conserve soil moisture.",
                "🐛 Inspect for pest/disease outbreaks — stressed plants are vulnerable.",
                "☂️ Deploy shade nets if temperature exceeds 35 °C.",
                "📞 Consult agronomist for crop-loss mitigation if stress persists > 3 days.",
            ]
        },
    }
    rec = RECS.get(stress_label, RECS["Healthy"])
    for action in rec["actions"]:
        if rec["color"] == "success":
            st.success(action)
        elif rec["color"] == "warning":
            st.warning(action)
        else:
            st.error(action)

    st.markdown("---")

    # Rule-Based Alerts
    st.subheader("🔔 Rule-Based Alerts")
    alerts = []
    if avg_temperature > 35:
        alerts.append(("🌡️ Heat Alert",    "Temperature above 35 °C — high transpiration stress.", "error"))
    elif avg_temperature < 10:
        alerts.append(("❄️ Cold Alert",    "Temperature below 10 °C — risk of frost damage.",     "error"))
    if humidity < 40:
        alerts.append(("🏜️ Low Humidity", "Humidity below 40% — accelerated moisture loss.",      "warning"))
    elif humidity > 90:
        alerts.append(("💦 High Humidity","Humidity above 90% — elevated fungal disease risk.",   "warning"))
    if rainfall == 0:
        alerts.append(("🌵 Drought Alert","No rainfall recorded — initiate irrigation protocol.", "error"))
    elif rainfall > 150:
        alerts.append(("🌊 Flood Alert",  "Rainfall > 150 mm — check drainage channels.",         "error"))
    if aqi > 200:
        alerts.append(("🏭 Air Quality",  f"AQI = {aqi} (Very Poor) — restrict outdoor work.",   "warning"))
    if wind_speed > 30:
        alerts.append(("💨 Wind Alert",   "Wind speed > 30 km/h — risk of physical crop damage.", "warning"))

    if alerts:
        for title, msg, level in alerts:
            if level == "error":
                st.error(f"**{title}**: {msg}")
            else:
                st.warning(f"**{title}**: {msg}")
    else:
        st.success("✅ No active alerts. All parameters within normal range.")


# ══════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ══════════════════════════════════════════════
with tab_eda:
    st.subheader("📂 Upload Dataset for EDA")
    uploaded = st.file_uploader(
        "Upload `climate_clean_labeled.csv` (output of EDA notebook)", type=["csv"]
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

        st.markdown("### 📋 Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows",    f"{len(df):,}")
        c2.metric("Columns", df.shape[1])
        if "stress_level" in df.columns:
            c3.metric("Stress classes", df["stress_level"].nunique())
        if "city" in df.columns:
            c4.metric("Cities", df["city"].nunique())

        with st.expander("🔍 Raw data sample"):
            st.dataframe(df.head(50), use_container_width=True)
        with st.expander("📈 Descriptive statistics"):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            st.dataframe(df[num_cols].describe().T.round(2), use_container_width=True)

        # Label distributions
        if "stress_level" in df.columns and "risk_level" in df.columns:
            st.markdown("### 🏷️ Label Distributions")
            lc1, lc2 = st.columns(2)
            with lc1:
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                order_s = ["Healthy","Mild Stress","Severe Stress"]
                vals_s  = df["stress_level"].value_counts().reindex(order_s, fill_value=0)
                ax1.bar(vals_s.index, vals_s.values,
                        color=["#2ca02c","#ff7f0e","#d62728"], edgecolor="white")
                ax1.set_title("Stress Level Distribution")
                for i, v in enumerate(vals_s.values):
                    ax1.text(i, v + 10, str(v), ha="center", fontsize=9)
                plt.tight_layout(); st.pyplot(fig1); plt.close(fig1)

            with lc2:
                fig2, ax2 = plt.subplots(figsize=(5, 3))
                order_r = ["Low","Moderate","High","Extreme"]
                vals_r  = df["risk_level"].value_counts().reindex(order_r, fill_value=0)
                ax2.bar(vals_r.index, vals_r.values,
                        color=["#2ca02c","#ff7f0e","#d62728","#8b0000"], edgecolor="white")
                ax2.set_title("Risk Level Distribution")
                for i, v in enumerate(vals_r.values):
                    ax2.text(i, v + 10, str(v), ha="center", fontsize=9)
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        # Correlation heatmap
        base_num = ["avg_temperature","rainfall","humidity","wind_speed","cloud_cover","aqi"]
        avail    = [c for c in base_num if c in df.columns]
        if len(avail) >= 3:
            st.markdown("### 🔥 Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            corr = df[avail].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                        mask=mask, linewidths=0.4, ax=ax3, square=True)
            ax3.set_title("Feature Correlation Matrix")
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        # Monthly trend
        if "month" in df.columns and avail:
            st.markdown("### 📅 Monthly Averages")
            monthly    = df.groupby("month")[avail].mean()
            feat_choice = st.selectbox("Select feature", avail)
            month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                            "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig4, ax4 = plt.subplots(figsize=(9, 3))
            ax4.plot(monthly.index, monthly[feat_choice],
                     marker="o", color="steelblue", linewidth=2)
            ax4.fill_between(monthly.index, monthly[feat_choice],
                             alpha=0.12, color="steelblue")
            ax4.set_xticks(range(1, 13))
            ax4.set_xticklabels(month_labels)
            ax4.set_title(f"Monthly Avg — {feat_choice}")
            ax4.set_ylabel(feat_choice)
            plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

        # Boxplot
        if "stress_level" in df.columns and avail:
            st.markdown("### 📦 Feature vs Stress Level")
            box_feat = st.selectbox("Select feature for boxplot", avail, key="box")
            order_sv = [o for o in ["Healthy","Mild Stress","Severe Stress"]
                        if o in df["stress_level"].unique()]
            fig5, ax5 = plt.subplots(figsize=(7, 4))
            sns.boxplot(data=df, x="stress_level", y=box_feat,
                        order=order_sv, palette="Set2", ax=ax5)
            ax5.set_title(f"{box_feat} vs Stress Level")
            ax5.set_xlabel("")
            plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    else:
        st.info("👆 Upload the labeled CSV (output of `eda_feature_eng.ipynb`) to explore the dataset.")


# ══════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 🍵 Tea Garden Climate Risk Analyzer

    A lightweight, explainable ML prototype that classifies **plant stress** and **climate risk**
    for tea garden management based on meteorological inputs.

    ### 🧠 Models
    | Task | Algorithm | Notes |
    |------|-----------|-------|
    | Stress Level | Random Forest | Best accuracy, balanced class weights |
    | Risk Level   | Random Forest | Same architecture, 4-class output |
    | Baseline     | Decision Tree | Interpretable, pruned to depth 12 |
    | Linear       | Logistic Regression | Scaled pipeline |

    ### 🌡️ Features Used
    `avg_temperature` · `rainfall` · `humidity` · `wind_speed` · `cloud_cover` · `aqi` ·
    `month` · `week` · rolling 7-day aggregates · deviation from rolling mean · cyclical month encoding

    ### 🏷️ Labels
    - **Stress Level** → Healthy / Mild Stress / Severe Stress
    - **Risk Level** → Low / Moderate / High / Extreme

    Both derived from a domain-rule **Climate Stress Index (CSI)** built during feature engineering.

    ### 📦 Dataset
    [Kaggle — Indian Climate Dataset 2024–2025](https://www.kaggle.com/datasets/ankushnarwade/indian-climate-dataset-20242025)

    ### 🔮 Future Scope
    - Live weather API integration for real-time inputs
    - Remote sensing signals (NDVI, satellite rainfall validation)
    - Soil moisture and leaf-wetness sensor fusion
    - Scheduled model retraining pipeline for drift detection
    """)

st.markdown("---")
st.caption(
    "🍵 Tea Garden Climate Risk Analyzer · "
    "ML models trained offline · Loaded via joblib · Streamlit UI"
)
