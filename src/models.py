"""Temporal-Correlated Predictor model definitions and regularized objectives."""

import numpy as np
import torch

class TCPBaseline(torch.nn.Module):
    def __init__(self, K, N, M):
        super().__init__()
        # Time-Region-Feature weights
        self.W = torch.nn.Parameter(torch.zeros(K, N, M))

    def forward(self, X):
        """
        X: (K, N, M)
        returns y_hat: (K, N)
        """
        return (X * self.W).sum(dim=-1)


class TCPModel(torch.nn.Module):
    def __init__(self, K, N, M):
        super().__init__()
        # W: Base weights mapping features to crime counts (Time, Region, Feature)
        self.W = torch.nn.Parameter(torch.zeros(K, N, M))

        # B: Holiday offset weights (Holiday_Type, Feature)
        # Assuming binary holiday type (is_holiday or not), so dim is (1, M)
        self.B = torch.nn.Parameter(torch.zeros(1, M))

    def forward(self, X, H):
        """
        X: shape (K, N, M) Features
        H: shape (K, 1)    Holiday indicators
        Returns predictions y_hat: shape (K, N)
        """
        # Calculate effective weights: W_eff = W + (H * B)
        # H @ B -> (K, 1) @ (1, M) -> (K, M)
        # Unsqueeze to broadcast over N regions -> (K, 1, M)
        holiday_offset = (H @ self.B).unsqueeze(1)

        W_eff = self.W + holiday_offset

        # Elementwise multiply and sum over feature dimension
        y_hat = (X * W_eff).sum(dim=-1)
        return y_hat


def tcp_loss_baseline(model, X, Y, train_idx, neighbor_indices,
                      lam=1.0, mu=1.0):
    """
    Baseline loss:
      1. MSE on training days
      2. Temporal smoothness on W
      3. Spatial smoothness on W (single mu)
    """
    # 1. Data fit on training window
    # Compute predictions only for training indices (avoid full-K forward).
    X_train = X[train_idx]
    W_train = model.W[train_idx]
    y_hat_train = (X_train * W_train).sum(dim=-1)
    Y_train = Y[train_idx]
    data_loss = torch.mean((y_hat_train - Y_train) ** 2)

    # 2. Temporal smoothness
    W_t = W_train[1:, :, :]
    W_tm1 = W_train[:-1, :, :]
    temp_loss = torch.mean((W_t - W_tm1) ** 2)

    # 3. Spatial smoothness (all days treated the same)
    idx_i, idx_j = neighbor_indices
    if idx_i is not None:
        diff_sq = (W_train[:, idx_i, :] - W_train[:, idx_j, :]) ** 2
        spat_loss = diff_sq.mean()
    else:
        spat_loss = torch.tensor(0.0, device=W_train.device)

    loss = data_loss + lam * temp_loss + mu * spat_loss
    parts = {
        "data": data_loss.item(),
        "temp": temp_loss.item(),
        "spat": spat_loss.item()
    }
    return loss, parts


def tcp_loss_extended(model, X, Y, H, train_idx, neighbor_indices, masks,
                      lam=1.0, lam7=0.0, mu_wd=1.0, mu_fri=1.0, mu_we=1.0,
                      gamma=1e-3):
    """
    Compute TCP loss with:
    1. MSE Data Fit
    2. Temporal Smoothness (on W)
    3. Regime-specific Spatial Smoothness (on W)
    4. Weekly temporal coupling (optional)
    5. Ridge Penalty on B (Holiday weights)
    """

    # Slice to training indices up front (avoid full-K forward and full-K regularizers).
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    H_train = H[train_idx]
    W_train = model.W[train_idx]

    # Compute predictions only for training indices.
    holiday_offset_train = (H_train @ model.B).unsqueeze(1)  # (K_train, 1, M)
    W_eff_train = W_train + holiday_offset_train
    y_hat_train = (X_train * W_eff_train).sum(dim=-1)

    # 1. Data fit loss (MSE) on training set
    data_loss = torch.mean((y_hat_train - Y_train) ** 2)

    # 2. Temporal smoothness: L2 norm of (W[t] - W[t-1])
    # We only smooth the base weights W, not the offsets
    W_t = W_train[1:, :, :]
    W_tm1 = W_train[:-1, :, :]
    temp_loss = torch.mean((W_t - W_tm1) ** 2)

    # 2b. Weekly temporal coupling: day t vs t-7
    if W_train.shape[0] > 7:
        W_t7  = W_train[7:, :, :]
        W_tm7 = W_train[:-7, :, :]
        weekly_loss = torch.mean((W_t7 - W_tm7) ** 2)
    else:
        weekly_loss = torch.tensor(0.0, device=W_train.device)


    # 3. Spatial smoothness (Regime-specific)
    loss_spat_wd = torch.tensor(0.0, device=W_train.device)
    loss_spat_fri = torch.tensor(0.0, device=W_train.device)
    loss_spat_we = torch.tensor(0.0, device=W_train.device)

    idx_i, idx_j = neighbor_indices

    if idx_i is not None:
        # Vectorized calculation
        diff_sq = (W_train[:, idx_i, :] - W_train[:, idx_j, :]) ** 2

        # Masks may be numpy arrays (from reshape_for_tcp) or torch tensors.
        if isinstance(masks.get("wd"), torch.Tensor):
            masks_train = {k: masks[k][train_idx] for k in ("wd", "fri", "we")}
        else:
            train_idx_np = (
                train_idx.detach().cpu().numpy()
                if isinstance(train_idx, torch.Tensor)
                else np.asarray(train_idx)
            )
            masks_train = {k: masks[k][train_idx_np] for k in ("wd", "fri", "we")}

        if masks_train["wd"].any():
            loss_spat_wd = diff_sq[masks_train["wd"]].mean()
        if masks_train["fri"].any():
            loss_spat_fri = diff_sq[masks_train["fri"]].mean()
        if masks_train["we"].any():
            loss_spat_we = diff_sq[masks_train["we"]].mean()

    # 4. Ridge penalty on holiday offsets
    ridge_b = torch.mean(model.B ** 2)

    # Total Loss
    loss = data_loss + \
           lam * temp_loss + \
           lam7 * weekly_loss + \
           mu_wd * loss_spat_wd + \
           mu_fri * loss_spat_fri + \
           mu_we * loss_spat_we + \
           gamma * ridge_b

    return loss, {
        "data": data_loss.item(),
        "temp": temp_loss.item(),
        "weekly": weekly_loss.item(),
        "spat_wd": loss_spat_wd.item(),
        "spat_fri": loss_spat_fri.item(),
        "spat_we": loss_spat_we.item(),
        "ridge_b": ridge_b.item()
    }
