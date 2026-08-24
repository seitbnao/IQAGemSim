from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGETS = [
    "ph",
    "do_mg_l",
    "dbo_mg_l",
    "ecoli_mpn_100ml",
    "water_temp_c",
    "delta_temp_c",
    "total_n_mg_l",
    "total_p_mg_l",
    "turbidity_ntu",
    "total_solids_mg_l",
]

EXOG = [
    "rain_mm",
    "flow_m3_s",
    "tide_m",
    "salinity_psu",
    "anthropic_pressure",
    "x_km",
    "y_km",
    "doy_sin",
    "doy_cos",
]

CLIPS = {
    "ph": (4.0, 10.0),
    "do_mg_l": (0.1, 20.0),
    "dbo_mg_l": (0.01, 80.0),
    "ecoli_mpn_100ml": (1.0, 1e8),
    "water_temp_c": (5.0, 40.0),
    "delta_temp_c": (0.0, 30.0),
    "total_n_mg_l": (0.0, 50.0),
    "total_p_mg_l": (0.0, 10.0),
    "turbidity_ntu": (0.0, 5000.0),
    "total_solids_mg_l": (0.0, 10000.0),
}

@dataclass
class ModelBundle:
    models: dict
    metrics: pd.DataFrame
    features: dict

def _prepare(df: pd.DataFrame, target: str):
    d = df.copy().sort_values(["station_id", "date"])
    d["date"] = pd.to_datetime(d["date"])
    d["doy_sin"] = np.sin(2*np.pi*d["date"].dt.dayofyear/365.25)
    d["doy_cos"] = np.cos(2*np.pi*d["date"].dt.dayofyear/365.25)
    d[f"{target}_lag1"] = d.groupby("station_id")[target].shift(1)
    d[f"{target}_lag7"] = d.groupby("station_id")[target].shift(7)
    features = EXOG + [f"{target}_lag1", f"{target}_lag7"]
    d = d.dropna(subset=features + [target])
    return d, features

def train_models(df: pd.DataFrame, random_state: int = 42) -> ModelBundle:
    models, features_map, rows = {}, {}, []
    for target in TARGETS:
        if target not in df.columns:
            continue
        d, features = _prepare(df, target)
        if len(d) < 100:
            continue
        cutoff = d["date"].quantile(0.80)
        train = d[d["date"] < cutoff]
        test = d[d["date"] >= cutoff]
        if len(test) < 20:
            continue

        model = RandomForestRegressor(
            n_estimators=110,
            max_depth=16,
            min_samples_leaf=2,
            max_features=0.85,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(train[features], train[target])
        pred = model.predict(test[features])
        baseline = test[f"{target}_lag1"].to_numpy()
        rows.append({
            "target": target,
            "MAE_RF": mean_absolute_error(test[target], pred),
            "RMSE_RF": mean_squared_error(test[target], pred)**0.5,
            "R2_RF": r2_score(test[target], pred),
            "MAE_persistencia": mean_absolute_error(test[target], baseline),
            "ganho_MAE_pct": 100*(mean_absolute_error(test[target], baseline)-mean_absolute_error(test[target], pred))
                            / max(mean_absolute_error(test[target], baseline), 1e-9),
        })
        models[target] = model
        features_map[target] = features

    return ModelBundle(models=models, metrics=pd.DataFrame(rows), features=features_map)

def _clip(target, value):
    lo, hi = CLIPS.get(target, (None, None))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return float(value)

def _future_exog(history: pd.DataFrame, future_date: pd.Timestamp, scenario: dict):
    recent = history.sort_values("date").tail(21)
    last = history.sort_values("date").iloc[-1]
    day_no = (future_date - pd.to_datetime(history["date"]).max()).days
    rain = float(recent["rain_mm"].median()) * (1 + scenario.get("rainfall_pct", 0)/100)
    flow = float(recent["flow_m3_s"].median()) * (1 + scenario.get("flow_pct", 0)/100)
    tide_base = float(recent["tide_m"].mean()) + 0.9*np.sin(2*np.pi*day_no/14.77)
    tide = tide_base * (1 + scenario.get("tide_amplitude_pct", 0)/100)
    salinity = float(recent["salinity_psu"].median()) + scenario.get("salinity_shift_psu", 0)
    return {
        "rain_mm": max(rain, 0),
        "flow_m3_s": max(flow, 1),
        "tide_m": tide,
        "salinity_psu": max(salinity, 0),
        "anthropic_pressure": float(last["anthropic_pressure"]),
        "x_km": float(last["x_km"]),
        "y_km": float(last["y_km"]),
        "doy_sin": np.sin(2*np.pi*future_date.dayofyear/365.25),
        "doy_cos": np.cos(2*np.pi*future_date.dayofyear/365.25),
    }

def _apply_scenario_perturbation(target: str, value: float, scenario: dict) -> float:
    nutrient = scenario.get("nutrient_load_pct", 0) / 100.0
    turb = scenario.get("turbidity_load_pct", 0) / 100.0
    point = scenario.get("point_source_intensity", 0) / 100.0

    if target == "total_n_mg_l":
        value *= 1 + nutrient
    elif target == "total_p_mg_l":
        value *= 1 + 1.15*nutrient
    elif target == "turbidity_ntu":
        value *= 1 + turb
    elif target == "total_solids_mg_l":
        value *= 1 + 0.75*turb
    elif target == "dbo_mg_l":
        value *= 1 + 1.8*point
    elif target == "ecoli_mpn_100ml":
        value *= 10**(1.35*point)
    elif target == "do_mg_l":
        value *= 1 - 0.25*point - 0.08*nutrient
    return _clip(target, value)

def forecast_station(
    df: pd.DataFrame,
    bundle: ModelBundle,
    station_id: str,
    horizon: int = 7,
    scenario: dict | None = None,
) -> pd.DataFrame:
    scenario = scenario or {}
    hist = df[df["station_id"] == station_id].copy().sort_values("date")
    hist["date"] = pd.to_datetime(hist["date"])
    if hist.empty:
        raise ValueError("Estação sem histórico.")

    # Histórico mutável para lags recursivos.
    work = hist.copy()
    rows = []
    last_date = work["date"].max()

    for step in range(1, horizon+1):
        future_date = last_date + pd.Timedelta(days=step)
        exog = _future_exog(work, future_date, scenario)
        row = {"date": future_date, "station_id": station_id, **exog}

        for target, model in bundle.models.items():
            features = bundle.features[target]
            series = work[target].dropna().to_list()
            lag1 = series[-1] if series else np.nan
            lag7 = series[-7] if len(series) >= 7 else lag1
            feat = dict(exog)
            feat[f"{target}_lag1"] = lag1
            feat[f"{target}_lag7"] = lag7
            X = pd.DataFrame([[feat[f] for f in features]], columns=features)

            tree_preds = np.array([est.predict(X.to_numpy())[0] for est in model.estimators_], dtype=float)
            mean = _apply_scenario_perturbation(target, float(tree_preds.mean()), scenario)
            sd = float(tree_preds.std(ddof=0))
            row[target] = mean
            row[f"{target}_sd"] = max(sd, 1e-6)

        rows.append(row)

        # Adiciona previsão ao histórico para alimentar os próximos lags.
        new = {c: np.nan for c in work.columns}
        new.update(row)
        work = pd.concat([work, pd.DataFrame([new])], ignore_index=True)

    return pd.DataFrame(rows)

def feature_importance(bundle: ModelBundle, target: str) -> pd.DataFrame:
    model = bundle.models[target]
    names = bundle.features[target]
    out = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)
