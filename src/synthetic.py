from __future__ import annotations

import numpy as np
import pandas as pd

from .iqa import oxygen_saturation_mg_l

def generate_synthetic_estuary(
    n_stations: int = 14,
    n_days: int = 420,
    seed: int = 42,
    end_date: str | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()

    x = np.linspace(4, 96, n_stations)
    y = 20 + 7*np.sin(x/14) + rng.normal(0, 1.2, n_stations)
    # Pressão maior em dois polos sintéticos.
    pressure = (
        0.12
        + 0.72*np.exp(-((x-38)/13)**2)
        + 0.45*np.exp(-((x-78)/10)**2)
    )
    pressure = np.clip(pressure, 0, 1)

    seasonal_rain = 18 + 15*np.sin(2*np.pi*(doy-35)/365) + 6*np.sin(4*np.pi*doy/365)
    rain = np.maximum(0, seasonal_rain + rng.gamma(2.0, 4.0, n_days) - 7)
    flow = 900 + 20*np.convolve(rain, np.ones(7)/7, mode="same") + 180*np.sin(2*np.pi*(doy-20)/365)
    flow += rng.normal(0, 45, n_days)
    flow = np.maximum(flow, 300)
    tide = 1.25*np.sin(2*np.pi*np.arange(n_days)/14.77) + rng.normal(0, 0.12, n_days)
    air_temp = 27.2 + 2.1*np.sin(2*np.pi*(doy-260)/365) + rng.normal(0, 0.7, n_days)

    frames = []
    for idx in range(n_stations):
        p = pressure[idx]
        xi, yi = x[idx], y[idx]
        along = xi / 100.0

        water_temp = air_temp - 0.7 + 0.6*along + rng.normal(0, 0.45, n_days)
        salinity = 1.8 + 29*along - 0.0055*(flow-900) + 0.55*tide + rng.normal(0, 0.7, n_days)
        salinity = np.clip(salinity, 0, 35)

        turbidity = 8 + 0.65*rain + 34*p + 0.017*flow + 6*np.maximum(tide, 0) + rng.normal(0, 5, n_days)
        turbidity = np.clip(turbidity, 1, 350)

        total_solids = 32 + 2.3*turbidity + 45*p + rng.normal(0, 22, n_days)
        total_solids = np.clip(total_solids, 15, 1600)

        tn = 0.30 + 0.014*rain + 1.20*p + 0.00045*flow + rng.normal(0, 0.12, n_days)
        tn = np.clip(tn, 0.05, 8)

        tp = 0.018 + 0.0010*rain + 0.16*p + 0.00003*flow + rng.normal(0, 0.015, n_days)
        tp = np.clip(tp, 0.004, 1.2)

        dbo = 0.8 + 0.012*rain + 2.6*p + 0.12*tn + rng.normal(0, 0.25, n_days)
        dbo = np.clip(dbo, 0.2, 20)

        log_ecoli = 1.15 + 0.016*rain + 2.15*p - 0.026*salinity + rng.normal(0, 0.22, n_days)
        ecoli = np.clip(10**log_ecoli, 1, 5e6)

        ph = 7.15 + 0.018*salinity - 0.18*p + 0.05*np.sin(2*np.pi*doy/30) + rng.normal(0, 0.08, n_days)
        ph = np.clip(ph, 5.8, 8.8)

        od_sat_pct = 102 - 4.4*dbo - 0.34*np.maximum(water_temp-25, 0) + 0.0025*(flow-900)
        od_sat_pct += 0.18*salinity + rng.normal(0, 3.2, n_days)
        od_sat_pct = np.clip(od_sat_pct, 38, 112)
        sat_mg_l = oxygen_saturation_mg_l(water_temp, salinity)
        do_mg_l = np.clip(sat_mg_l * od_sat_pct / 100, 1.5, 12)

        delta_temp = np.abs(0.4 + 1.3*p + 0.04*np.maximum(water_temp-29, 0) + rng.normal(0, 0.35, n_days))
        chl = np.clip(1.5 + 3.1*tn + 38*tp - 0.015*turbidity + rng.normal(0, 2, n_days), 0.2, 80)
        conductivity = np.clip(180 + salinity*1550 + rng.normal(0, 120, n_days), 100, 60000)

        frames.append(pd.DataFrame({
            "date": dates,
            "station_id": f"E{idx+1:02d}",
            "x_km": xi,
            "y_km": yi,
            "anthropic_pressure": p,
            "rain_mm": rain + rng.normal(0, 1.2, n_days),
            "flow_m3_s": flow + rng.normal(0, 20, n_days),
            "tide_m": tide + rng.normal(0, 0.05, n_days),
            "water_temp_c": water_temp,
            "delta_temp_c": delta_temp,
            "salinity_psu": salinity,
            "conductivity_us_cm": conductivity,
            "chlorophyll_ug_l": chl,
            "ph": ph,
            "do_mg_l": do_mg_l,
            "dbo_mg_l": dbo,
            "ecoli_mpn_100ml": ecoli,
            "total_n_mg_l": tn,
            "total_p_mg_l": tp,
            "turbidity_ntu": turbidity,
            "total_solids_mg_l": total_solids,
            "source": "synthetic",
            "is_imputed": False,
        }))

    out = pd.concat(frames, ignore_index=True)
    out["rain_mm"] = out["rain_mm"].clip(lower=0)
    return out.sort_values(["date", "station_id"]).reset_index(drop=True)
