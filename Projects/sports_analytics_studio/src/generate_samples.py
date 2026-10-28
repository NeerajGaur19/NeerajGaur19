"""Generate synthetic demo datasets. Run: python -m src.generate_samples"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(41)
n = 1200
ath = pd.DataFrame({
    "athlete_id": [f"A{i:04d}" for i in range(n)],
    "sport": rng.choice(["Football", "Basketball", "Rugby", "Athletics"], n),
    "position": rng.choice(["Forward", "Midfielder", "Defender", "Goalkeeper", "Guard", "Center", "Sprinter"], n),
    "age": rng.integers(16, 36, n),
    "height_cm": rng.normal(178, 10, n).round(1),
    "weight_kg": rng.normal(76, 13, n).round(1),
    "session_date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
    "training_load": rng.normal(420, 100, n).clip(80, 850).round(1),
    "distance_km": rng.normal(7.5, 2.2, n).clip(1, 15).round(2),
    "duration_min": rng.normal(75, 18, n).clip(20, 130).round(1),
    "top_speed_kmh": rng.normal(29, 4, n).clip(15, 40).round(1),
    "heart_rate_avg": rng.normal(148, 15, n).clip(90, 190).round(0),
    "heart_rate_max": rng.normal(185, 8, n).clip(145, 205).round(0),
    "hrv_ms": rng.normal(58, 15, n).clip(15, 120).round(1),
    "sleep_hours": rng.normal(7.4, 1.1, n).clip(3, 11).round(1),
    "recovery_score": rng.normal(76, 13, n).clip(20, 100).round(0),
    "sprints": rng.poisson(22, n),
    "accel_events": rng.poisson(48, n),
    "decel_events": rng.poisson(43, n),
    "rpe": rng.integers(1, 11, n),
    "performance_score": rng.normal(72, 12, n).clip(20, 100).round(1),
})
ath["avg_speed_kmh"] = (ath["distance_km"] / (ath["duration_min"] / 60)).round(2)
logit = -3.1 + .004 * ath["training_load"] + .06 * ath["rpe"] - .025 * ath["recovery_score"] + .04 * (8 - ath["sleep_hours"])
ath["injury_within_7_days"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
ath.to_csv(OUT / "athlete_performance.csv", index=False)
inj = ath[["athlete_id","sport","position","age","training_load","distance_km","top_speed_kmh","heart_rate_avg","hrv_ms","sleep_hours","recovery_score","rpe","injury_within_7_days"]].copy()
inj["acute_load_7d"] = (inj["training_load"] * rng.uniform(4.5, 7.5, n)).round(1)
inj["chronic_load_28d"] = (inj["training_load"] * rng.uniform(3.2, 5.4, n)).round(1)
inj["previous_injury"] = rng.binomial(1, .22, n)
inj.to_csv(OUT / "injury_risk.csv", index=False)
tal = ath[["athlete_id","sport","position","age","height_cm","weight_kg","top_speed_kmh","distance_km","performance_score","recovery_score"]].copy()
tal["coach_rating"] = rng.normal(7.1, 1.2, n).clip(1,10).round(1)
tal["technical_score"] = rng.normal(72,13,n).clip(10,100).round(1)
tal["potential_label"] = np.where((tal.performance_score > 78) & (tal.technical_score > 72), "High", np.where(tal.performance_score > 63, "Medium", "Developing"))
tal.to_csv(OUT / "talent_identification.csv", index=False)
ev = pd.DataFrame({
    "event_id":[f"E{i:05d}" for i in range(n)], "match_date":pd.date_range("2025-02-01",periods=n,freq="D").astype(str),
    "competition":rng.choice(["League","Cup","Friendly","Playoff"],n),
    "home_team":rng.choice(["Falcons","United","Rovers","Athletic","City"],n),
    "away_team":rng.choice(["Falcons","United","Rovers","Athletic","City"],n),
    "venue":rng.choice(["Home","Away","Neutral"],n), "possession_pct":rng.normal(51,12,n).clip(15,85).round(1),
    "passes_attempted":rng.integers(180,800,n), "passes_completed":rng.integers(130,720,n),
    "shots":rng.poisson(12,n), "shots_on_target":rng.poisson(5,n), "xg":rng.gamma(2,.7,n).round(2),
    "corners":rng.poisson(5,n), "fouls":rng.poisson(11,n), "yellow_cards":rng.poisson(1.5,n),
    "red_cards":rng.binomial(1,.04,n), "goals_for":rng.poisson(1.5,n), "goals_against":rng.poisson(1.2,n),
    "attendance":rng.integers(800,65000,n), "temperature_c":rng.normal(19,7,n).round(1), "humidity_pct":rng.integers(25,96,n)
})
ev["passes_completed"] = np.minimum(ev["passes_completed"], ev["passes_attempted"])
ev["shots_on_target"] = np.minimum(ev["shots_on_target"], ev["shots"])
ev.to_csv(OUT / "event_analytics.csv", index=False)
sensor = ath[["athlete_id","session_date","training_load","distance_km","top_speed_kmh","heart_rate_avg","hrv_ms","sleep_hours"]].copy()
idx = rng.choice(n,35,replace=False)
sensor.loc[idx,"top_speed_kmh"] *= rng.uniform(1.8,2.5,len(idx))
sensor.loc[rng.choice(n,25,replace=False),"heart_rate_avg"] = rng.uniform(30,60,25)
sensor.to_csv(OUT / "sensor_anomalies.csv",index=False)
print(f"Synthetic samples written to {OUT}")
