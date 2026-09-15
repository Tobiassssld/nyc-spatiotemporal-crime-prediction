"""
Compute Global Moran's I for crime intensity per region for each day type.

How inputs are located:
- Crime counts: loads the same processed dataset as `run_experiments.py` by default
  (`data/FEATURE_MATRIX.csv`) and uses `region_id`, `date`, `complaint_count`.
- Centroids: uses the grid polygons used by `run_experiments.py`
  (`data/nyc_grid_2km_active.shp`), computing centroids from geometry.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


DAY_TYPE_DEFS = {
    "wd": "Mon-Thu",
    "fri": "Friday",
    "we": "Sat-Sun",
}


@dataclass(frozen=True)
class MoranResult:
    day_type: str
    N: int
    k: int
    morans_I: float
    p_value: float
    x_mean: float
    x_std: float


def _die(msg: str, exit_code: int = 2) -> None:
    print(f"[morans_daytype] ERROR: {msg}", file=sys.stderr)
    raise SystemExit(exit_code)


def load_daily_counts(feature_matrix_path: str) -> pd.DataFrame:
    if not os.path.exists(feature_matrix_path):
        _die(
            f"Missing '{feature_matrix_path}'. Expected the processed feature "
            "matrix used by the experiment pipeline."
        )

    df = pd.read_csv(feature_matrix_path)
    required = {"region_id", "date"}
    missing = sorted(required - set(df.columns))
    if missing:
        _die(f"'{feature_matrix_path}' missing required columns: {missing}")

    # Prefer the TCP target column; otherwise try common alternatives.
    count_col = None
    for cand in ("complaint_count", "crime_count", "count", "y"):
        if cand in df.columns:
            count_col = cand
            break
    if count_col is None:
        _die(
            f"'{feature_matrix_path}' must contain a daily count column; expected one of "
            "['complaint_count', 'crime_count', 'count', 'y']."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        _die(f"Failed to parse some 'date' values in '{feature_matrix_path}'.")

    df["region_id"] = pd.to_numeric(df["region_id"], errors="coerce")
    if df["region_id"].isna().any():
        _die(f"Failed to parse some 'region_id' values in '{feature_matrix_path}'.")
    df["region_id"] = df["region_id"].astype(int)

    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
    if df[count_col].isna().any():
        _die(f"Failed to parse some '{count_col}' values in '{feature_matrix_path}'.")

    # If already daily region-day rows, this is a no-op; if incident-level, aggregates.
    daily = (
        df.groupby(["region_id", "date"], as_index=False)[count_col]
        .sum()
        .rename(columns={count_col: "count"})
    )
    return daily


def compute_daytype_means(daily: pd.DataFrame) -> pd.DataFrame:
    if not {"region_id", "date", "count"} <= set(daily.columns):
        _die("Internal error: daily data must contain region_id, date, count.")

    dow = daily["date"].dt.dayofweek  # 0=Mon ... 6=Sun
    day_type = np.full(len(daily), "", dtype=object)
    day_type[(dow >= 0) & (dow <= 3)] = "wd"
    day_type[dow == 4] = "fri"
    day_type[dow >= 5] = "we"
    daily = daily.copy()
    daily["day_type"] = day_type

    means = (
        daily.groupby(["region_id", "day_type"], as_index=False)["count"]
        .mean()
        .rename(columns={"count": "mean_count"})
    )
    pivot = (
        means.pivot(index="region_id", columns="day_type", values="mean_count")
        .reindex(columns=["wd", "fri", "we"])
        .sort_index()
    )
    pivot = pivot.reset_index()
    return pivot


def load_centroids_from_grid(grid_path: str, regions_sorted: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Returns:
      coords: (N, 2) array, either (x, y) in projected units or (lon, lat) in degrees.
      projected: True if coordinates are projected (Euclidean kNN), else haversine.
    """
    try:
        import geopandas as gpd  # type: ignore
    except Exception as e:  # pragma: no cover
        _die(
            "geopandas is required to read the grid shapefile and compute centroids. "
            "Install the repo dependencies (e.g., `pip install -r requirements.txt`). "
            f"Import error: {e}"
        )

    if not os.path.exists(grid_path):
        _die(
            f"Missing '{grid_path}'. Expected the active grid shapefile used by the experiment pipeline "
            "(or adjust the path in this script)."
        )

    gdf = gpd.read_file(grid_path)
    if "region_id" not in gdf.columns:
        _die(f"'{grid_path}' missing required column 'region_id'.")

    gdf["region_id"] = pd.to_numeric(gdf["region_id"], errors="coerce")
    if gdf["region_id"].isna().any():
        _die(f"Failed to parse some 'region_id' values in '{grid_path}'.")
    gdf["region_id"] = gdf["region_id"].astype(int)

    gdf = gdf.set_index("region_id")
    missing = sorted(set(regions_sorted.tolist()) - set(gdf.index.tolist()))
    if missing:
        _die(f"Grid is missing {len(missing)} region_ids present in the data (e.g., {missing[:10]}).")

    gdf = gdf.loc[regions_sorted].copy()
    centroids = gdf.geometry.centroid

    xs = centroids.x.to_numpy()
    ys = centroids.y.to_numpy()

    projected = False
    try:
        projected = bool(gdf.crs) and bool(getattr(gdf.crs, "is_projected", False))
    except Exception:
        projected = False

    coords = np.column_stack([xs, ys])
    return coords, projected


def knn_neighbors(coords: np.ndarray, k: int, *, projected: bool) -> np.ndarray:
    """
    Returns integer neighbor indices of shape (N, k). Excludes self.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        _die("coords must be an (N, 2) array.")
    N = coords.shape[0]
    if N <= k:
        _die(f"Need N > k to build kNN weights (got N={N}, k={k}).")

    if projected:
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
    else:
        # coords are (lon, lat) in degrees
        lon = np.deg2rad(coords[:, 0])
        lat = np.deg2rad(coords[:, 1])
        dlon = lon[:, None] - lon[None, :]
        dlat = lat[:, None] - lat[None, :]
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        dist = 6371000.0 * c  # meters

    np.fill_diagonal(dist, np.inf)
    nn = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]

    # Deterministic ordering among the k nearest (useful for reproducibility).
    row = np.arange(N)[:, None]
    order = np.argsort(dist[row, nn], axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    return nn


def morans_I_from_knn(x: np.ndarray, neighbors: np.ndarray) -> float:
    """
    Global Moran's I with row-standardized kNN weights (w_ij = 1/k for j in N(i)).
    Implements:
      I = (N/S0) * (sum_i sum_j w_ij z_i z_j) / (sum_i z_i^2)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        _die("x must be a 1D array.")
    if neighbors.ndim != 2:
        _die("neighbors must be an (N, k) array of indices.")
    N, k = neighbors.shape
    if x.shape[0] != N:
        _die(f"x length must match neighbors N (got len(x)={x.shape[0]}, N={N}).")

    xbar = float(np.mean(x))
    z = x - xbar
    denom = float(np.sum(z * z))
    if denom == 0.0:
        return float("nan")

    w = 1.0 / float(k)
    Wy = w * np.sum(z[neighbors], axis=1)
    numer = float(np.sum(z * Wy))
    S0 = float(N)  # row-standardized: each row sums to 1
    I = (float(N) / S0) * (numer / denom)
    return float(I)


def permutation_test_morans_I(
    x: np.ndarray,
    neighbors: np.ndarray,
    *,
    permutations: int = 999,
    seed: int = 123,
) -> Tuple[float, float]:
    """
    Returns (I_observed, p_value) using a two-sided pseudo p-value.
    """
    I_obs = morans_I_from_knn(x, neighbors)
    if not np.isfinite(I_obs):
        return I_obs, float("nan")

    rng = np.random.default_rng(seed)
    perm_Is = np.empty(permutations, dtype=float)
    for r in range(permutations):
        perm_x = rng.permutation(x)
        perm_Is[r] = morans_I_from_knn(perm_x, neighbors)

    extreme = np.sum(np.abs(perm_Is) >= abs(I_obs))
    p = float((extreme + 1) / (permutations + 1))
    return I_obs, p


def run(
    feature_matrix_path: str = "data/FEATURE_MATRIX.csv",
    grid_path: str = "data/nyc_grid_2km_active.shp",
    k: int = 8,
    permutations: int = 999,
    seed: int = 123,
    out_csv: str = "results/tables/morans_daytype_results.csv",
) -> None:
    daily = load_daily_counts(feature_matrix_path)
    means = compute_daytype_means(daily)

    regions_sorted = means["region_id"].to_numpy(dtype=int)
    coords, projected = load_centroids_from_grid(grid_path, regions_sorted)
    neighbors = knn_neighbors(coords, k=k, projected=projected)

    results: list[MoranResult] = []
    for day_type in ("wd", "fri", "we"):
        if day_type not in means.columns:
            _die(f"Missing computed column '{day_type}' in day-type mean table.")

        x = means[day_type].to_numpy(dtype=float)
        if np.isnan(x).any():
            _die(
                f"NaNs found in x_{day_type}. This usually means some regions "
                "have no data for that day type. Check the date range and "
                "region coverage."
            )

        I, p = permutation_test_morans_I(
            x, neighbors, permutations=permutations, seed=seed
        )
        results.append(
            MoranResult(
                day_type=day_type,
                N=int(len(x)),
                k=int(k),
                morans_I=float(I),
                p_value=float(p),
                x_mean=float(np.mean(x)),
                x_std=float(np.std(x, ddof=0)),
            )
        )

    # Console summary
    coord_mode = "projected (Euclidean)" if projected else "geographic (haversine)"
    print("--- Global Moran's I by Day Type ---")
    print(f"N regions: {len(regions_sorted)}")
    print(f"k (kNN):   {k}")
    print(f"coords:    {coord_mode}")
    for r in results:
        label = DAY_TYPE_DEFS.get(r.day_type, r.day_type)
        I_str = f"{r.morans_I:.6f}" if np.isfinite(r.morans_I) else "nan"
        p_str = f"{r.p_value:.6f}" if np.isfinite(r.p_value) else "nan"
        print(f"{r.day_type:>3} ({label:<7})  I={I_str}  p={p_str}")

    out_df = pd.DataFrame(
        [
            {
                "day_type": r.day_type,
                "N": r.N,
                "k": r.k,
                "morans_I": r.morans_I,
                "p_value": r.p_value,
                "x_mean": r.x_mean,
                "x_std": r.x_std,
            }
            for r in results
        ]
    )
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved '{out_csv}'")


if __name__ == "__main__":
    run()
