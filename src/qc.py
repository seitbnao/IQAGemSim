from __future__ import annotations

import numpy as np
import pandas as pd

PHYSICAL_LIMITS = {
    "ph": (0.0, 14.0),
    "do_mg_l": (0.0, 25.0),
    "dbo_mg_l": (0.0, 100.0),
    "ecoli_mpn_100ml": (0.0, 1e9),
    "water_temp_c": (-2.0, 45.0),
    "delta_temp_c": (0.0, 35.0),
    "total_n_mg_l": (0.0, 100.0),
    "total_p_mg_l": (0.0, 20.0),
    "turbidity_ntu": (0.0, 5000.0),
    "total_solids_mg_l": (0.0, 10000.0),
    "salinity_psu": (0.0, 45.0),
    "rain_mm": (0.0, 1000.0),
    "flow_m3_s": (0.0, None),
}

def apply_qc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["qc_exclude"] = False
    out["qc_suspect"] = False
    out["qc_flags"] = ""

    flags = [[] for _ in range(len(out))]
    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        invalid = values.isna()
        if lo is not None:
            invalid |= values < lo
        if hi is not None:
            invalid |= values > hi
        for i in np.where(invalid.to_numpy())[0]:
            flags[i].append(f"{col}:fora_limite_fisico")
        out.loc[invalid, "qc_exclude"] = True

    # Detecção de saltos robusta por estação: suspeito, mas não exclui.
    jump_cols = ["ph", "do_mg_l", "turbidity_ntu", "salinity_psu", "total_n_mg_l", "total_p_mg_l"]
    if "station_id" in out.columns and "date" in out.columns:
        tmp = out.copy()
        tmp["_order"] = np.arange(len(tmp))
        tmp = tmp.sort_values(["station_id", "date"])
        for col in jump_cols:
            if col not in tmp.columns:
                continue
            diff = tmp.groupby("station_id")[col].diff().abs()
            med = diff.groupby(tmp["station_id"]).transform("median")
            mad = (diff - med).abs().groupby(tmp["station_id"]).transform("median")
            threshold = med + 8 * mad.replace(0, np.nan)
            suspect = (diff > threshold) & threshold.notna()
            original_idx = tmp.loc[suspect, "_order"].astype(int).to_numpy()
            out.iloc[original_idx, out.columns.get_loc("qc_suspect")] = True
            for i in original_idx:
                flags[i].append(f"{col}:salto_suspeito")

    out["qc_flags"] = [";".join(x) for x in flags]
    return out
