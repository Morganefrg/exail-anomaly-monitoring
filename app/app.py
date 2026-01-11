import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DATA_PATH = "data/sensor_readings.csv"
MODEL_PATH = "models/anomaly_model.joblib"
FEATURES = ["temperature", "vibration", "current", "pressure"]

st.set_page_config(page_title="Anomaly Monitoring", layout="wide")
st.title("Anomaly Monitoring for Sensor Data")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

st.sidebar.header("Controls")
window = st.sidebar.slider("Number of last points to display", 200, 2000, 600, step=50)

view = df.tail(window).copy()
X = view[FEATURES].copy()

# IsolationForest:
# - predict: 1 normal, -1 anomalie
pred = model.predict(X)
view["is_anomaly"] = (pred == -1).astype(int)

# score_samples: plus petit = plus anormal
scores = model.named_steps["clf"].score_samples(model.named_steps["scaler"].transform(X))
view["anomaly_score"] = scores

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Recent sensor signals")
    fig, ax = plt.subplots()
    ax.plot(view["time"], view["temperature"], label="temperature")
    ax.plot(view["time"], view["vibration"], label="vibration")
    ax.plot(view["time"], view["current"], label="current")
    ax.plot(view["time"], view["pressure"], label="pressure")
    ax.set_xlabel("time")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Anomaly summary")
    n_anom = int(view["is_anomaly"].sum())
    st.metric("Detected anomalies (window)", n_anom)

    latest = view.iloc[-1]
    if latest["is_anomaly"] == 1:
        st.error("Latest point looks anomalous.")
    else:
        st.success("Latest point looks normal.")

    st.write("Last points with anomaly flag:")
    st.dataframe(view[["time"] + FEATURES + ["is_anomaly", "anomaly_score"]].tail(20), use_container_width=True)

st.divider()
st.subheader("Explainability (simple)")
st.write(
    "This is a simple anomaly detector: it learns what 'normal' sensor behavior looks like and flags unusual patterns. "
    "In a real industrial setting, thresholds and alerts are calibrated with engineering teams."
)
