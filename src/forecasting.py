"""Alpha-stage forecasting of time-varying TCP model weights."""

import numpy as np

def _fit_alpha_for_region(W_history, window_size=7, ridge=1e-4):
    """
    Learn alpha coefficients for a single region by solving Eq. (20) in the TCP paper.
    W_history: numpy array of shape (T_hist, M)
    Returns: (alpha vector, effective_window_size)
    """
    window_size = max(1, int(window_size))
    T_hist, M = W_history.shape

    if T_hist == 0:
        return np.array([1.0]), 1
    if T_hist == 1:
        return np.array([1.0]), 1

    g = min(window_size, T_hist - 1)
    num_samples = (T_hist - g) * M
    if num_samples <= 0:
        alpha = np.ones(g) / g
        return alpha, g

    rows = []
    targets = []
    for idx in range(g, T_hist):
        past = W_history[idx - g: idx]  # shape (g, M)
        rows.append(past.T)             # (M, g)
        targets.append(W_history[idx])  # (M,)

    X = np.concatenate(rows, axis=0)    # (#samples, g)
    y = np.concatenate(targets, axis=0) # (#samples,)
    XtX = X.T @ X + ridge * np.eye(g)
    Xty = X.T @ y
    alpha = np.linalg.solve(XtX, Xty)
    return alpha, g


def _forecast_region_weights(W_history, alpha, horizon):
    """
    Generate future W vectors for a region using learned alpha coefficients.
    """
    g = len(alpha)
    if horizon <= 0:
        return np.zeros((0, W_history.shape[1]))

    history = W_history[-g:].copy()
    if history.shape[0] < g:
        pad = np.repeat(history[:1], g - history.shape[0], axis=0)
        history = np.concatenate([pad, history], axis=0)

    preds = []
    for _ in range(horizon):
        new_w = np.tensordot(alpha, history, axes=(0, 0))
        preds.append(new_w)
        if g > 1:
            history = np.concatenate([history[1:], new_w[None, :]], axis=0)
        else:
            history[0] = new_w
    return np.stack(preds, axis=0)


def forecast_weights_with_alpha(W_history, horizon, window_size=7, ridge=1e-4):
    """
    Learn per-region alpha coefficients from historical W values and forecast
    future weights for a contiguous horizon.
    W_history: numpy array (T_hist, N, M)
    Returns:
        forecasts: numpy array (horizon, N, M)
        alphas: list of numpy arrays, one per region
    """
    if horizon <= 0:
        return np.zeros((0, ) + W_history.shape[1:]), []

    T_hist, N, _ = W_history.shape
    forecasts = np.zeros((horizon, N, W_history.shape[2]))
    alphas = []
    for n in range(N):
        region_hist = W_history[:, n, :]
        alpha, g = _fit_alpha_for_region(region_hist, window_size=window_size, ridge=ridge)
        # Ensure history has enough length for the chosen window
        if region_hist.shape[0] < g:
            if region_hist.shape[0] == 0:
                seed = np.zeros((1, W_history.shape[2]))
            else:
                seed = region_hist[-1:]
            effective_hist = np.repeat(seed, g, axis=0)
        else:
            effective_hist = region_hist
        region_forecast = _forecast_region_weights(effective_hist, alpha, horizon)
        forecasts[:, n, :] = region_forecast
        alphas.append(alpha)
    return forecasts, alphas
