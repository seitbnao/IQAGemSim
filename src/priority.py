from __future__ import annotations

import numpy as np
import pandas as pd
from .spatial import interpolate_surface, distance_to_nearest_station, normalize_grid

def monitoring_priority_surface(
    latest: pd.DataFrame,
    weights: dict | None = None,
    grid_size: int = 65,
):
    weights = weights or {"risk": 0.35, "uncertainty": 0.30, "pressure": 0.20, "distance": 0.15}
    total = sum(weights.values()) or 1.0
    weights = {k: v/total for k, v in weights.items()}

    iqa = interpolate_surface(latest, "iqa", grid_size=grid_size)
    pressure = interpolate_surface(latest, "anthropic_pressure", grid_size=grid_size)
    risk = normalize_grid(100 - iqa["surface"])
    uncertainty = normalize_grid(iqa["variance"])
    press = normalize_grid(pressure["surface"])
    distance = normalize_grid(distance_to_nearest_station(iqa["x"], iqa["y"], latest))

    priority = 100 * (
        weights["risk"]*risk
        + weights["uncertainty"]*uncertainty
        + weights["pressure"]*press
        + weights["distance"]*distance
    )
    return {
        "x": iqa["x"],
        "y": iqa["y"],
        "priority": priority,
        "risk": 100*risk,
        "uncertainty": 100*uncertainty,
        "pressure": 100*press,
        "distance": 100*distance,
        "method": iqa["method"],
        "weights": weights,
    }
