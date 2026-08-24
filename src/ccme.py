from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_OBJECTIVES = {
    "ph": {"min": 6.5, "max": 8.5, "label": "pH"},
    "do_mg_l": {"min": 5.0, "label": "OD (mg/L)"},
    "dbo_mg_l": {"max": 5.0, "label": "DBO (mg/L)"},
    "ecoli_mpn_100ml": {"max": 1000.0, "label": "E. coli (NMP/100 mL)"},
    "total_n_mg_l": {"max": 2.0, "label": "N total (mg/L)"},
    "total_p_mg_l": {"max": 0.10, "label": "P total (mg/L)"},
    "turbidity_ntu": {"max": 100.0, "label": "Turbidez (NTU)"},
}

def _excursion(value: float, rule: dict) -> tuple[bool, float]:
    if pd.isna(value):
        return False, np.nan
    if "max" in rule and value > rule["max"]:
        return True, value / rule["max"] - 1.0
    if "min" in rule and value < rule["min"]:
        if value <= 0:
            return True, rule["min"] / 1e-9 - 1.0
        return True, rule["min"] / value - 1.0
    return False, 0.0

def calculate_ccme_wqi(df: pd.DataFrame, objectives: dict | None = None) -> dict:
    objectives = objectives or DEFAULT_OBJECTIVES
    variables = [v for v in objectives if v in df.columns]
    if not variables or df.empty:
        return {"ccme_wqi": np.nan, "F1": np.nan, "F2": np.nan, "F3": np.nan,
                "failed_variables": 0, "failed_tests": 0, "total_tests": 0}

    failed_vars = set()
    excursions = []
    total_tests = 0
    failed_tests = 0

    for var in variables:
        for value in df[var].dropna().astype(float):
            total_tests += 1
            failed, exc = _excursion(value, objectives[var])
            if failed:
                failed_vars.add(var)
                failed_tests += 1
                excursions.append(exc)
            else:
                excursions.append(0.0)

    if total_tests == 0:
        return {"ccme_wqi": np.nan, "F1": np.nan, "F2": np.nan, "F3": np.nan,
                "failed_variables": 0, "failed_tests": 0, "total_tests": 0}

    F1 = len(failed_vars) / len(variables) * 100.0
    F2 = failed_tests / total_tests * 100.0
    nse = float(np.sum(excursions) / total_tests)
    F3 = nse / (0.01 * nse + 0.01) if nse > 0 else 0.0
    wqi = 100.0 - np.sqrt(F1**2 + F2**2 + F3**2) / 1.732
    return {
        "ccme_wqi": float(np.clip(wqi, 0, 100)),
        "F1": float(F1),
        "F2": float(F2),
        "F3": float(F3),
        "failed_variables": len(failed_vars),
        "failed_tests": failed_tests,
        "total_tests": total_tests,
    }

def classify_ccme(value: float) -> str:
    if pd.isna(value):
        return "Sem cálculo"
    if value >= 95:
        return "Excelente"
    if value >= 80:
        return "Boa"
    if value >= 65:
        return "Razoável"
    if value >= 45:
        return "Marginal"
    return "Ruim"

def objective_status_table(df: pd.DataFrame, objectives: dict | None = None) -> pd.DataFrame:
    objectives = objectives or DEFAULT_OBJECTIVES
    rows = []
    for var, rule in objectives.items():
        if var not in df.columns:
            continue
        s = df[var].dropna().astype(float)
        failures = sum(_excursion(v, rule)[0] for v in s)
        obj = []
        if "min" in rule:
            obj.append(f">= {rule['min']}")
        if "max" in rule:
            obj.append(f"<= {rule['max']}")
        rows.append({
            "Variável": rule.get("label", var),
            "Objetivo demonstrativo": " e ".join(obj),
            "Medições": len(s),
            "Falhas": int(failures),
            "Falhas (%)": round(100 * failures / max(len(s), 1), 1),
        })
    return pd.DataFrame(rows)
