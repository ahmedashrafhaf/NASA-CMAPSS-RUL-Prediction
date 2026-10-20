import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- High-Level System Configuration ---
st.set_page_config(
    page_title="BEFORE ZERO | Industrial Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# --- Professional High-Contrast CSS ---
st.markdown("""
<style>
[data-testid="stMetricValue"] {
    color: #002D62 !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #1C1C1E !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}
[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 2px solid #0047AB !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background-color: #f1f3f5;
    border-radius: 5px;
    padding: 10px 20px;
    font-weight: bold;
    color: #495057;
}
.stTabs [aria-selected="true"] {
    background-color: #0047AB !important;
    color: white !important;
}
.insight-card {
    background-color: #eef4ff;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #0047AB;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

# --- Asset & Model Loader ---
@st.cache_resource
def load_phm_engine():
    model = joblib.load("rf_model.pkl")
    features = [
        'num_of_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3',
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
        'sensor_17', 'sensor_20', 'sensor_21'
    ]
    return model, features

model, training_features = load_phm_engine()

# --- Sidebar ---
with st.sidebar:
    st.title("BEFORE ZERO")
    st.caption("Prognostics & Health Management (PHM) System")
    st.divider()
    uploaded_file = st.file_uploader("📥 Synchronize Telemetry Data", type=["csv"])
    st.divider()
    if uploaded_file:
        st.success("✅ Telemetry Stream Active")
    else:
        st.info("Awaiting CSV input...")

# --- Main Dashboard ---
st.title("✈️ Fleet Reliability & RUL Analytics")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    id_col = data.columns[0]
    data[id_col] = data[id_col].astype(int)

    missing_cols = set(training_features) - set(data.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[training_features])

    preds = model.predict(X_scaled)
    data["Predicted_RUL"] = np.maximum(0, preds).astype(int)

    data["Status"] = data["Predicted_RUL"].apply(
        lambda x: "🚨 Critical" if x <= 30 else ("⚠️ Warning" if x <= 75 else "✅ Healthy")
    )

    fleet_latest = data.groupby(id_col).tail(1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Assets Active", len(fleet_latest))
    k2.metric("Mean RUL (Fleet)", f"{fleet_latest['Predicted_RUL'].mean():.0f} Cycles")
    k3.metric("Critical Alerts", len(fleet_latest[fleet_latest["Status"].str.contains("Critical")]))
    k4.metric(
        "Reliability Index",
        f"{(fleet_latest['Status'].str.contains('Healthy').sum() / len(fleet_latest) * 100):.1f}%"
    )

    st.divider()

    tab_fleet, tab_diagnostics, tab_sensors = st.tabs(
        ["🌐 Fleet Risk Portfolio", "🔬 Asset Digital Twin", "⚙️ Sensor Sensitivity"]
    )

    # --- TAB 1: Fleet ---
    with tab_fleet:
        st.markdown(
            """
            <div class='insight-card'>
                <b style="color:#0047AB; font-size:18px;">
                    Fleet Intelligence Summary
                </b>
                <span style="color:#495057;">
                    : Analyze life expectancy distribution across all units.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns([1, 1.2])
        with c1:
            fig_pie = px.pie(
                fleet_latest,
                names="Status",
                hole=0.55,
                color="Status",
                color_discrete_map={
                    "✅ Healthy": "#2ecc71",
                    "⚠️ Warning": "#f1c40f",
                    "🚨 Critical": "#e74c3c"
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            fig_hist = px.histogram(
                fleet_latest,
                x="Predicted_RUL",
                nbins=20,
                labels={"Predicted_RUL": "Cycles to Failure"},
                color_discrete_sequence=["#0047AB"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # --- TAB 2: Digital Twin ---
    with tab_diagnostics:
        col_ctrl, col_chart = st.columns([1, 2.2])

        with col_ctrl:
            asset_ids = sorted(data[id_col].unique())
            selected_id = st.selectbox("Select Asset Serial Number", asset_ids)

            unit_data = data[data[id_col] == selected_id]
            curr_val = unit_data["Predicted_RUL"].iloc[-1]
            curr_stat = unit_data["Status"].iloc[-1]

            st.subheader(f"Diagnosis: Unit {selected_id}")

            if "Critical" in curr_stat:
                st.error(f"⚠️ IMMEDIATE GROUNDING — RUL = {curr_val} cycles")
            elif "Warning" in curr_stat:
                st.warning("🔧 Preventive maintenance required soon")
            else:
                st.success("✔️ Asset operational and healthy")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=curr_val,
                gauge={
                    "axis": {"range": [0, 250]},
                    "bar": {"color": "#0047AB"},
                    "steps": [
                        {"range": [0, 30], "color": "#f8d7da"},
                        {"range": [30, 75], "color": "#fff3cd"},
                        {"range": [75, 250], "color": "#d4edda"}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_chart:
            fig_trend = px.line(
                unit_data,
                x="num_of_cycles",
                y="Predicted_RUL",
                labels={
                    "num_of_cycles": "Cycles Completed",
                    "Predicted_RUL": "Remaining Useful Life"
                }
            )
            fig_trend.update_traces(line_color="#0047AB", line_width=4)
            st.plotly_chart(fig_trend, use_container_width=True)

    # --- TAB 3: XAI ---
    with tab_sensors:
        st.markdown(
            """
            <div class='insight-card'>
                <b style="color:#0047AB;">Explainable AI (XAI):</b>
                <span style="color:#1C1C1E;"> Sensor contribution to RUL prediction.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Sensor": training_features,
            "Impact": importances
        }).sort_values("Impact", ascending=True)

        fig_imp = px.bar(
            feat_df,
            x="Impact",
            y="Sensor",
            orientation="h",
            color="Impact",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, use_container_width=True)

else:
    st.warning("🛡️ Please upload engine telemetry CSV via the sidebar to begin analysis.")
