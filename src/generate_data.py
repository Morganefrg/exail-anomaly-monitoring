import numpy as np
import pandas as pd

OUT_PATH = "data/sensor_readings.csv"

def main(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Signaux "normaux" (bruit + petites variations)
    temperature = 40 + 0.01*t + rng.normal(0, 0.6, n)
    vibration   = 1.2 + 0.2*np.sin(t/60) + rng.normal(0, 0.08, n)
    current     = 10 + 0.1*np.sin(t/120) + rng.normal(0, 0.25, n)
    pressure    = 3.0 + rng.normal(0, 0.05, n)

    df = pd.DataFrame({
        "time": t,
        "temperature": temperature,
        "vibration": vibration,
        "current": current,
        "pressure": pressure,
    })

    # Injecter des anomalies (ex: surchauffe, vibration anormale, surconsommation)
    anomaly = np.zeros(n, dtype=int)

    # 1) pics de vibration
    idx = rng.choice(np.arange(500, n-500), size=20, replace=False)
    df.loc[idx, "vibration"] += rng.uniform(1.5, 3.0, size=len(idx))
    anomaly[idx] = 1

    # 2) dérive de température (surchauffe progressive)
    start = 2500
    df.loc[start:start+300, "temperature"] += np.linspace(0, 15, 301)
    anomaly[start:start+301] = 1

    # 3) surconsommation (courant)
    idx2 = rng.choice(np.arange(800, n-800), size=25, replace=False)
    df.loc[idx2, "current"] += rng.uniform(3, 6, size=len(idx2))
    anomaly[idx2] = 1

    # 4) chute de pression
    idx3 = rng.choice(np.arange(700, n-700), size=10, replace=False)
    df.loc[idx3, "pressure"] -= rng.uniform(0.5, 1.0, size=len(idx3))
    anomaly[idx3] = 1

    df["anomaly_true"] = anomaly  # juste pour tester, pas nécessaire en vrai

    df.to_csv(OUT_PATH, index=False)
    print(f"OK: dataset créé -> {OUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    main()
