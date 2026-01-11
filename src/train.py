import os
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/sensor_readings.csv"
MODEL_PATH = "models/anomaly_model.joblib"

FEATURES = ["temperature", "vibration", "current", "pressure"]

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()

    # Pipeline: standardisation + Isolation Forest
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(
            n_estimators=200,
            contamination=0.02,  # % anomalies attendues (ajuste si besoin)
            random_state=42
        ))
    ])

    model.fit(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"OK: modèle sauvegardé -> {MODEL_PATH}")
    print("Features:", FEATURES)

if __name__ == "__main__":
    main()
