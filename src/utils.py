from __future__ import annotations

import numpy as np
import pandas as pd
from .iqa import calculate_iqa

REQUIRED_COLUMNS = [
    "date", "station_id", "x_km", "y_km", "anthropic_pressure",
    "rain_mm", "flow_m3_s", "tide_m", "salinity_psu",
    "water_temp_c", "delta_temp_c", "ph", "do_mg_l", "dbo_mg_l",
    "ecoli_mpn_100ml", "total_n_mg_l", "total_p_mg_l",
    "turbidity_ntu", "total_solids_mg_l",
]

def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing

def add_forecast_iqa(forecast: pd.DataFrame) -> pd.DataFrame:
    out = forecast.copy()
    out["iqa"] = out.apply(calculate_iqa, axis=1)
    return out

def monte_carlo_iqa(forecast: pd.DataFrame, n: int = 160, seed: int = 42):
    rng = np.random.default_rng(seed)
    targets = [
        "ph", "do_mg_l", "dbo_mg_l", "ecoli_mpn_100ml", "water_temp_c",
        "delta_temp_c", "total_n_mg_l", "total_p_mg_l",
        "turbidity_ntu", "total_solids_mg_l",
    ]
    rows = []
    for _, r in forecast.iterrows():
        sims = []
        for _ in range(n):
            sample = r.to_dict()
            for t in targets:
                sd = float(r.get(f"{t}_sd", 0))
                if sd > 0:
                    if t == "ecoli_mpn_100ml":
                        # Amostragem log-normal aproximada.
                        mu = np.log(max(float(r[t]), 1))
                        rel = min(sd / max(float(r[t]), 1), 1.5)
                        sample[t] = float(np.exp(rng.normal(mu, rel)))
                    else:
                        sample[t] = float(rng.normal(float(r[t]), sd))
            # Limites físicos básicos.
            sample["ph"] = np.clip(sample["ph"], 0, 14)
            for t in ["do_mg_l","dbo_mg_l","ecoli_mpn_100ml","delta_temp_c",
                      "total_n_mg_l","total_p_mg_l","turbidity_ntu","total_solids_mg_l"]:
                sample[t] = max(sample[t], 0.0001)
            sims.append(calculate_iqa(sample))
        sims = np.asarray(sims, dtype=float)
        rows.append({
            "date": r["date"],
            "iqa_p10": float(np.nanpercentile(sims, 10)),
            "iqa_p50": float(np.nanpercentile(sims, 50)),
            "iqa_p90": float(np.nanpercentile(sims, 90)),
        })
    return pd.DataFrame(rows)
