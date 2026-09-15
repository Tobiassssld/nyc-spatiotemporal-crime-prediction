"""Temporal splitting and prediction metrics/evaluation helpers."""

import numpy as np
import torch

from .forecasting import forecast_weights_with_alpha

def compute_armse(y_true, y_pred):
    """
    Compute average RMSE across regions (aRMSE).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for aRMSE.")
    if y_true.ndim != 2:
        raise ValueError("aRMSE expects 2D arrays shaped (K, N).")
    rmse_per_region = np.sqrt(np.nanmean((y_pred - y_true) ** 2, axis=0))
    return float(np.nanmean(rmse_per_region))


def build_rolling_splits(K, train_size, val_size, test_size, step_size,
                         device=None, drop_last=True):
    """
    Create rolling (walk-forward) train/val/test splits over time indices.

    When train_size + val_size + test_size is smaller than K, successive
    splits move forward by step_size. If the three windows span all K time
    steps, the function returns one chronological split.
    """
    if any(s <= 0 for s in [train_size, val_size, test_size, step_size]):
        raise ValueError("train_size, val_size, test_size, and step_size must be > 0.")
    if K < train_size + val_size + test_size:
        raise ValueError("Not enough time steps for the requested split sizes.")

    splits = []
    start = 0
    max_start = K - (train_size + val_size + test_size)
    while start <= max_start:
        train_start = start
        train_end = train_start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        train_idx = torch.arange(train_start, train_end, device=device)
        val_idx = torch.arange(train_end, val_end, device=device)
        test_idx = torch.arange(val_end, test_end, device=device)

        splits.append({
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "train_val_idx": torch.arange(train_start, val_end, device=device),
        })
        start += step_size

    if not splits and not drop_last:
        train_start = 0
        train_end = min(K, train_size)
        val_end = min(K, train_end + val_size)
        test_end = min(K, val_end + test_size)
        splits.append({
            "train_idx": torch.arange(train_start, train_end, device=device),
            "val_idx": torch.arange(train_end, val_end, device=device),
            "test_idx": torch.arange(val_end, test_end, device=device),
            "train_val_idx": torch.arange(train_start, val_end, device=device),
        })

    return splits


def eval_with_alpha_prediction(model, X_torch, Y_torch, train_idx, eval_idx,
                               window_size=7, alpha_ridge=1e-4, H_torch=None,
                               use_holiday_offset=True):
    """
    Evaluate a model by forecasting W tensors with the sliding-window alpha stage.
    """
    eval_len = int(eval_idx.shape[0])
    if eval_len == 0:
        return 0.0, np.array([]), np.array([])

    with torch.no_grad():
        device = X_torch.device
        dtype = X_torch.dtype

        W_history = model.W.detach()[train_idx].detach().cpu().numpy()
        horizon = eval_len

        weight_forecasts, _ = forecast_weights_with_alpha(
            W_history,
            horizon,
            window_size=window_size,
            ridge=alpha_ridge
        )

        W_forecast = torch.as_tensor(weight_forecasts, device=device, dtype=dtype)
        X_eval = X_torch[eval_idx]

        W_eff = W_forecast
        if use_holiday_offset and hasattr(model, "B") and (H_torch is not None):
            H_eval = H_torch[eval_idx]
            holiday_offset = (H_eval @ model.B).unsqueeze(1)  # (horizon, 1, M)
            W_eff = W_eff + holiday_offset                    # broadcast over N

        y_hat_eval = (X_eval * W_eff).sum(dim=-1)
        y_true_eval = Y_torch[eval_idx]

    y_pred_2d = y_hat_eval.detach().cpu().numpy()
    y_true_2d = y_true_eval.detach().cpu().numpy()
    armse = compute_armse(y_true_2d, y_pred_2d)
    return armse, y_true_2d.ravel(), y_pred_2d.ravel()


def eval_baseline(model, X_torch, Y_torch, train_idx, eval_idx,
                  alpha_window=7, alpha_ridge=1e-4):
    return eval_with_alpha_prediction(
        model, X_torch, Y_torch,
        train_idx, eval_idx,
        window_size=alpha_window,
        alpha_ridge=alpha_ridge
    )


def eval_extended(model, X_torch, Y_torch, H_torch, train_idx, eval_idx,
                  alpha_window=7, alpha_ridge=1e-4):
    return eval_with_alpha_prediction(
        model, X_torch, Y_torch,
        train_idx, eval_idx,
        window_size=alpha_window,
        alpha_ridge=alpha_ridge,
        H_torch=H_torch,
    )
