from __future__ import annotations

import numpy as np
import pandas as pd

def _idw(x, y, z, gx, gy, power=2.0):
    xx, yy = np.meshgrid(gx, gy)
    surface = np.zeros_like(xx, dtype=float)
    variance = np.zeros_like(xx, dtype=float)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            d = np.hypot(x - xx[i, j], y - yy[i, j])
            if np.any(d < 1e-9):
                k = np.argmin(d)
                surface[i, j] = z[k]
                variance[i, j] = 0
                continue
            w = 1.0 / np.maximum(d, 1e-6)**power
            w /= w.sum()
            m = np.sum(w*z)
            surface[i, j] = m
            variance[i, j] = np.sum(w*(z-m)**2)
    return surface, variance

def interpolate_surface(
    df: pd.DataFrame,
    value_col: str,
    grid_size: int = 65,
    variogram_model: str = "spherical",
):
    pts = df[["x_km", "y_km", value_col]].dropna().copy()
    pts = pts.groupby(["x_km", "y_km"], as_index=False)[value_col].mean()
    if len(pts) < 4:
        raise ValueError("São necessários pelo menos 4 pontos válidos para interpolação.")

    x = pts["x_km"].to_numpy(float)
    y = pts["y_km"].to_numpy(float)
    z = pts[value_col].to_numpy(float)
    pad_x = max((x.max()-x.min())*0.08, 1)
    pad_y = max((y.max()-y.min())*0.15, 1)
    gx = np.linspace(x.min()-pad_x, x.max()+pad_x, grid_size)
    gy = np.linspace(y.min()-pad_y, y.max()+pad_y, grid_size)

    try:
        from pykrige.ok import OrdinaryKriging
        ok = OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
            pseudo_inv=True,
        )
        surf, var = ok.execute("grid", gx, gy)
        surf = np.asarray(surf, dtype=float)
        var = np.maximum(np.asarray(var, dtype=float), 0)
        method = f"Kriging ordinário ({variogram_model})"
    except Exception:
        surf, var = _idw(x, y, z, gx, gy)
        method = "IDW (fallback)"
    return {"x": gx, "y": gy, "surface": surf, "variance": var, "method": method}

def distance_to_nearest_station(gx, gy, stations: pd.DataFrame):
    xx, yy = np.meshgrid(gx, gy)
    dist = np.full(xx.shape, np.inf)
    sx = stations["x_km"].to_numpy(float)
    sy = stations["y_km"].to_numpy(float)
    for x, y in zip(sx, sy):
        dist = np.minimum(dist, np.hypot(xx-x, yy-y))
    return dist

def normalize_grid(a):
    a = np.asarray(a, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros_like(a)
    lo, hi = np.nanpercentile(a[finite], [2, 98])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a-lo)/(hi-lo), 0, 1)
