from __future__ import annotations

import numpy as np
import pandas as pd

# Pesos oficiais divulgados pela ANA para o IQA.
WEIGHTS = {
    "do_saturation_pct": 0.17,
    "ecoli_mpn_100ml": 0.15,
    "ph": 0.12,
    "dbo_mg_l": 0.10,
    "delta_temp_c": 0.10,
    "total_n_mg_l": 0.10,
    "total_p_mg_l": 0.10,
    "turbidity_ntu": 0.08,
    "total_solids_mg_l": 0.08,
}

# Curvas discretizadas para o protótipo. Elas reproduzem a forma geral das
# curvas NSF/CETESB, mas NÃO substituem a tabela/curva oficial para uso regulatório.
CURVES = {
    "do_saturation_pct": (
        np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]),
        np.array([1, 3, 8, 15, 25, 40, 58, 75, 88, 96, 100, 98, 92, 84, 74]),
    ),
    "ecoli_mpn_100ml": (
        np.array([1, 10, 100, 1_000, 10_000, 100_000, 1_000_000], dtype=float),
        np.array([100, 95, 82, 50, 24, 8, 2], dtype=float),
    ),
    "ph": (
        np.array([2, 4, 5, 6, 6.5, 7, 8, 8.5, 9, 10, 12], dtype=float),
        np.array([2, 5, 22, 55, 82, 95, 95, 88, 72, 35, 2], dtype=float),
    ),
    "dbo_mg_l": (
        np.array([0, 1, 2, 3, 5, 8, 10, 20, 30], dtype=float),
        np.array([100, 96, 90, 82, 65, 45, 34, 10, 2], dtype=float),
    ),
    "delta_temp_c": (
        np.array([0, 1, 2, 5, 10, 15, 20, 25], dtype=float),
        np.array([100, 98, 95, 84, 58, 32, 12, 2], dtype=float),
    ),
    "total_n_mg_l": (
        np.array([0, 0.2, 0.5, 1, 2, 5, 10, 20, 40], dtype=float),
        np.array([100, 97, 92, 82, 68, 45, 26, 10, 2], dtype=float),
    ),
    "total_p_mg_l": (
        np.array([0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1, 2], dtype=float),
        np.array([100, 98, 95, 88, 74, 55, 28, 12, 2], dtype=float),
    ),
    "turbidity_ntu": (
        np.array([0, 5, 10, 25, 50, 100, 200, 500, 1000], dtype=float),
        np.array([100, 97, 93, 83, 68, 48, 29, 10, 2], dtype=float),
    ),
    "total_solids_mg_l": (
        np.array([0, 25, 50, 100, 200, 500, 1000, 2000], dtype=float),
        np.array([100, 98, 95, 88, 73, 46, 23, 3], dtype=float),
    ),
}

PHYSICAL_CLIPS = {
    "ph": (0.0, 14.0),
    "dbo_mg_l": (0.0, None),
    "ecoli_mpn_100ml": (1.0, None),
    "total_n_mg_l": (0.0, None),
    "total_p_mg_l": (0.0, None),
    "turbidity_ntu": (0.0, None),
    "total_solids_mg_l": (0.0, None),
    "delta_temp_c": (0.0, None),
    "do_mg_l": (0.0, None),
    "water_temp_c": (-2.0, 45.0),
}

def oxygen_saturation_mg_l(temp_c: float | np.ndarray, salinity_psu: float | np.ndarray = 0.0):
    """Aproximação de OD de saturação para fins do simulador."""
    t = np.asarray(temp_c, dtype=float)
    s = np.asarray(salinity_psu, dtype=float)
    fresh = 14.652 - 0.41022*t + 0.007991*t**2 - 0.000077774*t**3
    salinity_factor = np.clip(1.0 - 0.0065*s, 0.72, 1.0)
    return np.maximum(fresh * salinity_factor, 1.0)

def do_saturation_percent(do_mg_l, temp_c, salinity_psu=0.0):
    sat = oxygen_saturation_mg_l(temp_c, salinity_psu)
    return np.clip(np.asarray(do_mg_l, dtype=float) / sat * 100.0, 0, 160)

def _interp_quality(name: str, value):
    x, q = CURVES[name]
    v = np.asarray(value, dtype=float)
    if name == "ecoli_mpn_100ml":
        v = np.log10(np.maximum(v, 1.0))
        x = np.log10(x)
    return np.interp(v, x, q, left=q[0], right=q[-1])

def quality_subindices(row: pd.Series | dict) -> dict:
    data = dict(row)
    if "do_saturation_pct" not in data or pd.isna(data.get("do_saturation_pct")):
        if all(k in data for k in ("do_mg_l", "water_temp_c")):
            data["do_saturation_pct"] = float(
                do_saturation_percent(
                    data["do_mg_l"],
                    data["water_temp_c"],
                    data.get("salinity_psu", 0.0),
                )
            )
    out = {}
    for name in WEIGHTS:
        val = data.get(name, np.nan)
        if pd.isna(val):
            out[name] = np.nan
        else:
            out[name] = float(np.clip(_interp_quality(name, val), 1, 100))
    return out

def calculate_iqa(row: pd.Series | dict) -> float:
    qis = quality_subindices(row)
    if any(pd.isna(qis[k]) for k in WEIGHTS):
        return np.nan
    log_iqa = sum(WEIGHTS[k] * np.log(max(qis[k], 1e-9)) for k in WEIGHTS)
    return float(np.clip(np.exp(log_iqa), 0, 100))

def classify_iqa(value: float) -> str:
    if pd.isna(value):
        return "Sem cálculo"
    if value < 19:
        return "Péssima"
    if value < 36:
        return "Ruim"
    if value < 51:
        return "Regular"
    if value < 79:
        return "Boa"
    return "Ótima"

def compute_iqa_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "do_saturation_pct" not in out.columns:
        out["do_saturation_pct"] = do_saturation_percent(
            out["do_mg_l"].to_numpy(),
            out["water_temp_c"].to_numpy(),
            out.get("salinity_psu", pd.Series(0.0, index=out.index)).to_numpy(),
        )
    out["iqa"] = out.apply(calculate_iqa, axis=1)
    out["iqa_class"] = out["iqa"].apply(classify_iqa)
    for p in WEIGHTS:
        out[f"q_{p}"] = out.apply(lambda r: quality_subindices(r)[p], axis=1)
    return out
