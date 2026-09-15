"""Data loading and spatial/temporal preparation utilities."""

import holidays
import numpy as np
import pandas as pd
import torch

def get_holiday_tensor(dates, device):
    us_holidays = holidays.US()  # or holidays.US(state="NY")

    # DatetimeIndex -> list of python date objects
    holiday_dates = {d for d in (dt.date() for dt in dates) if d in us_holidays}

    is_holiday = pd.Index([dt.date() for dt in dates]).isin(holiday_dates)

    H_tensor = torch.tensor(is_holiday.astype(float), dtype=torch.float32).unsqueeze(1).to(device)
    print(f"Holiday tensor shape: {H_tensor.shape}. Found {int(H_tensor.sum().item())} holidays.")
    return H_tensor


def reshape_for_tcp(X_df,
                    feature_cols=None,
                    target_col="complaint_count"):
    """
    Convert feature matrix (region_id, date, columns...) into
    TCP-style tensors and generate day-type masks.

    Returns:
    X_tensor : ndarray of shape (K, N, M)
    Y        : ndarray of shape (K, N)
    regions  : ndarray of region_ids of length N
    dates    : DatetimeIndex of length K
    masks    : dict containing boolean numpy arrays for 'wd', 'fri', 'we'
    """

    df = X_df.copy()

    # Ensure proper dtypes and sorting
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "region_id"])

    # Unique regions and dates in fixed order
    regions = np.sort(df["region_id"].unique())
    dates = pd.Index(sorted(df["date"].unique()))

    N = len(regions)
    K = len(dates)

    # Pivot logic
    def pivot_col(col):
        table = (
            df.pivot(index="date", columns="region_id", values=col)
            .reindex(index=dates, columns=regions)
        )
        return table.values.astype(float)  # shape (K, N)

    # Target tensor (K, N)
    Y = pivot_col(target_col)

    # Stack feature tensors along last axis -> (K, N, M)
    if feature_cols:
        feat_arrays = [pivot_col(col)[:, :, None] for col in feature_cols]
        X_tensor = np.concatenate(feat_arrays, axis=-1)
    else:
        X_tensor = np.zeros((K, N, 0))

    print(f"X_tensor shape: {X_tensor.shape}  (K, N, M)")
    print(f"Y shape:        {Y.shape}        (K, N)")

    # --- Generate Day-Type Masks ---
    # 0=Monday, 6=Sunday
    dayofweek = dates.dayofweek.values

    masks = {
        "wd": (dayofweek >= 0) & (dayofweek <= 3),  # Mon-Thu
        "fri": (dayofweek == 4),  # Fri
        "we": (dayofweek >= 5)  # Sat-Sun
    }

    return X_tensor, Y, regions, dates, masks


def build_neighbor_pairs(grid_active, regions):
    """
    Build list of neighbor index pairs from grid polygons.
    """
    # Ensure GeoDataFrame is in same order as regions
    grid = grid_active.set_index("region_id").loc[regions].reset_index()

    sindex = grid.sindex
    neighbor_pairs = set()

    for i, geom in enumerate(grid.geometry):
        possible = list(sindex.intersection(geom.bounds))
        for j in possible:
            if i == j:
                continue
            if geom.touches(grid.geometry.iloc[j]):
                a, b = sorted((i, j))
                neighbor_pairs.add((a, b))

    neighbor_pairs = sorted(neighbor_pairs)
    print(f"Found {len(neighbor_pairs)} neighbor pairs")
    return neighbor_pairs


def process_neighbor_indices(neighbor_pairs, device):
    """
    Helper to convert list of tuples [(i, j), ...] into two tensors
    for vectorized indexing.
    """
    if not neighbor_pairs:
        return None, None

    idx_i = [p[0] for p in neighbor_pairs]
    idx_j = [p[1] for p in neighbor_pairs]

    return torch.tensor(idx_i, device=device), torch.tensor(idx_j, device=device)
