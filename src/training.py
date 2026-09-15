"""Training, hyperparameter search, and extension-ablation routines."""

import torch
from tqdm import tqdm

from .models import TCPBaseline, TCPModel, tcp_loss_baseline, tcp_loss_extended
from .evaluation import eval_baseline, eval_extended
from .visualization import visualize_baseline_grid, visualize_extended_grid

def train_baseline(X_torch, Y_torch, train_idx, neighbor_indices,
                   lam, mu, lr=1e-2, num_epochs=200,
                   device="cpu", verbose=False,
                   val_idx=None, early_stopping=False,
                   early_stop_patience=20, early_stop_min_delta=0.0,
                   early_stop_eval_every=5,
                   alpha_window=7, alpha_ridge=1e-4):
    K, N, M = X_torch.shape
    model = TCPBaseline(K, N, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()
        loss, parts = tcp_loss_baseline(
            model, X_torch, Y_torch, train_idx, neighbor_indices,
            lam=lam, mu=mu
        )
        loss.backward()
        opt.step()

        if early_stopping and val_idx is not None and (epoch % early_stop_eval_every == 0):
            rmse, _, _ = eval_baseline(
                model, X_torch, Y_torch, train_idx, val_idx,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if rmse + early_stop_min_delta < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if verbose:
                        print(f"[Baseline] Early stopping at epoch {epoch} (best aRMSE={best_rmse:.4f})")
                    break

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"[Baseline] Epoch {epoch:4d}  "
                  f"Loss={loss.item():.4f}  "
                  f"Data={parts['data']:.4f}  "
                  f"Temp={parts['temp']:.4f}  "
                  f"Spat={parts['spat']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def grid_search_baseline(X_torch, Y_torch, train_idx, val_idx, neighbor_indices,
                         lam_values, mu_values,
                         lr=1e-2, num_epochs=200, device="cpu",
                         viz_path="baseline_grid_search.png",
                         alpha_window=7, alpha_ridge=1e-4,
                         early_stopping=False, early_stop_patience=20,
                         early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Run a brute-force grid search over lam/mu for the baseline model.
    Uses the validation split for model selection and produces a visualization
    summarizing validation RMSE across the grid.
    """
    device = torch.device(device)
    total_configs = len(lam_values) * len(mu_values)
    best_result = None
    best_model = None
    history = []

    with tqdm(total=total_configs, desc="Baseline Grid Search") as pbar:
        for lam in lam_values:
            for mu in mu_values:
                model = train_baseline(
                    X_torch, Y_torch, train_idx, neighbor_indices,
                    lam=lam, mu=mu,
                    lr=lr, num_epochs=num_epochs, device=device,
                    verbose=False,
                    val_idx=val_idx, early_stopping=early_stopping,
                    early_stop_patience=early_stop_patience,
                    early_stop_min_delta=early_stop_min_delta,
                    early_stop_eval_every=early_stop_eval_every,
                    alpha_window=alpha_window,
                    alpha_ridge=alpha_ridge
                )
                rmse, y_true, y_pred = eval_baseline(
                    model, X_torch, Y_torch, train_idx, val_idx,
                    alpha_window=alpha_window,
                    alpha_ridge=alpha_ridge
                )
                history.append({"lam": lam, "mu": mu, "rmse": rmse})

                if best_result is None or rmse < best_result["rmse"]:
                    best_result = {
                        "rmse": rmse,
                        "params": {"lam": lam, "mu": mu},
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }
                    best_model = model
                    pbar.set_postfix({"best_rmse": f"{rmse:.4f}"})
                else:
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                pbar.update(1)

    if best_result is None:
        raise RuntimeError("Grid search did not evaluate any configurations.")

    if viz_path:
        visualize_baseline_grid(history, viz_path)
    print(
        "[Baseline Grid Search] Best Validation aRMSE="
        f"{best_result['rmse']:.4f} with "
        f"lam={best_result['params']['lam']}, "
        f"mu={best_result['params']['mu']}"
    )

    best_result["model"] = best_model
    best_result["history"] = history
    return best_result


def grid_search_extended(X_torch, Y_torch, H_torch, train_idx, val_idx,
                         neighbor_indices, masks_torch,
                         lam,
                         lam7_values, mu_wd_values, mu_fri_values, mu_we_values, gamma_values,
                         lam_values=None,
                         lr=1e-2, num_epochs=200, device="cpu",
                         viz_path="extended_grid_search.png",
                         alpha_window=7, alpha_ridge=1e-4,
                         early_stopping=False, early_stop_patience=20,
                         early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Grid search over the extended-model regularizers that are not shared with the baseline.
    Optionally retunes lam (baseline temporal smoothness) jointly with extension parameters.
    """
    device = torch.device(device)
    lam_candidates = lam_values if lam_values is not None else [lam]
    total_configs = (len(lam_candidates) *
                     len(mu_wd_values) * len(mu_fri_values) *
                     len(mu_we_values) * len(gamma_values) * len(lam7_values))
    best_result = None
    best_model = None
    history = []

    with tqdm(total=total_configs, desc="Extended Grid Search") as pbar:
        for lam_candidate in lam_candidates:
            for lam7 in lam7_values:
                for mu_wd in mu_wd_values:
                    for mu_fri in mu_fri_values:
                        for mu_we in mu_we_values:
                            for gamma in gamma_values:
                                model = train_extended(
                                    X_torch, Y_torch, H_torch, train_idx,
                                    neighbor_indices, masks_torch,
                                    lam=lam_candidate, lam7=lam7, mu_wd=mu_wd,
                                    mu_fri=mu_fri, mu_we=mu_we,
                                    gamma=gamma,
                                    lr=lr, num_epochs=num_epochs, device=device,
                                    verbose=False,
                                    val_idx=val_idx, early_stopping=early_stopping,
                                    early_stop_patience=early_stop_patience,
                                    early_stop_min_delta=early_stop_min_delta,
                                    early_stop_eval_every=early_stop_eval_every,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge
                                )
                                rmse, y_true, y_pred = eval_extended(
                                    model, X_torch, Y_torch, H_torch, train_idx, val_idx,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge
                                )
                                history.append({
                                    "lam": lam_candidate,
                                    "lam7": lam7,
                                    "mu_wd": mu_wd,
                                    "mu_fri": mu_fri,
                                    "mu_we": mu_we,
                                    "gamma": gamma,
                                    "rmse": rmse
                                })

                                if best_result is None or rmse < best_result["rmse"]:
                                    best_result = {
                                        "rmse": rmse,
                                        "params": {
                                            "lam": lam_candidate,
                                            "lam7": lam7,
                                            "mu_wd": mu_wd,
                                            "mu_fri": mu_fri,
                                            "mu_we": mu_we,
                                            "gamma": gamma
                                        },
                                        "y_true": y_true,
                                        "y_pred": y_pred,
                                    }
                                    best_model = model
                                    pbar.set_postfix({"best_rmse": f"{rmse:.4f}"})
                                else:
                                    del model
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                pbar.update(1)

    if best_result is None:
        raise RuntimeError("Extended grid search did not evaluate any configurations.")

    if viz_path:
        visualize_extended_grid(history, viz_path)
    params = best_result["params"]
    print(
        "[Extended Grid Search] Best Validation aRMSE="
        f"{best_result['rmse']:.4f} with "
        f"lam={params.get('lam', lam)}, "
        f"lam7={params['lam7']}, mu_wd={params['mu_wd']}, "
        f"mu_fri={params['mu_fri']}, mu_we={params['mu_we']}, "
        f"gamma={params['gamma']}"
    )

    best_result["model"] = best_model
    best_result["history"] = history
    return best_result


def train_extended(X_torch, Y_torch, H_torch, train_idx,
                   neighbor_indices, masks_torch,
                   lam, lam7, mu_wd, mu_fri, mu_we, gamma,
                   lr=1e-2, num_epochs=200, device="cpu", verbose=False,
                   val_idx=None, early_stopping=False,
                   early_stop_patience=20, early_stop_min_delta=0.0,
                   early_stop_eval_every=5,
                   alpha_window=7, alpha_ridge=1e-4):
    K, N, M = X_torch.shape
    model = TCPModel(K, N, M).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()
        loss, parts = tcp_loss_extended(
            model, X_torch, Y_torch, H_torch, train_idx,
            neighbor_indices, masks_torch,
            lam=lam,lam7=lam7, mu_wd=mu_wd, mu_fri=mu_fri, mu_we=mu_we,
            gamma=gamma
        )
        loss.backward()
        opt.step()

        if early_stopping and val_idx is not None and (epoch % early_stop_eval_every == 0):
            rmse, _, _ = eval_extended(
                model, X_torch, Y_torch, H_torch, train_idx, val_idx,
                alpha_window=alpha_window,
                alpha_ridge=alpha_ridge
            )
            if rmse + early_stop_min_delta < best_rmse:
                best_rmse = rmse
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    if verbose:
                        print(f"[Extended] Early stopping at epoch {epoch} (best aRMSE={best_rmse:.4f})")
                    break

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"[Extended] Epoch {epoch:4d}  "
                  f"Loss={loss.item():.4f}  "
                  f"Data={parts['data']:.4f}  "
                  f"Temp={parts['temp']:.4f}  "
                  f"Spat_WD={parts['spat_wd']:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_eval_extended_config(X_torch, Y_torch, H_torch, train_idx, eval_idx,
                               neighbor_indices, masks_torch, lam,
                               lam7, mu_wd, mu_fri, mu_we, gamma,
                               lr=1e-2, num_epochs=200, device="cpu",
                               return_model=False,
                               alpha_window=7, alpha_ridge=1e-4,
                               early_stopping=False, early_stop_patience=20,
                               early_stop_min_delta=0.0, early_stop_eval_every=5):
    model = train_extended(
        X_torch, Y_torch, H_torch, train_idx,
        neighbor_indices, masks_torch,
        lam=lam, lam7=lam7, mu_wd=mu_wd, mu_fri=mu_fri, mu_we=mu_we,
        gamma=gamma,
        lr=lr, num_epochs=num_epochs, device=device, verbose=False,
        val_idx=eval_idx, early_stopping=early_stopping,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_eval_every=early_stop_eval_every,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    rmse, y_true, y_pred = eval_extended(
        model, X_torch, Y_torch, H_torch, train_idx, eval_idx,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    result = {
        "rmse": rmse,
        "params": {
            "lam": lam,
            "lam7": lam7,
            "mu_wd": mu_wd,
            "mu_fri": mu_fri,
            "mu_we": mu_we,
            "gamma": gamma
        },
        "y_true": y_true,
        "y_pred": y_pred,
    }
    if return_model:
        result["model"] = model
    else:
        del model
    return result


def eval_best_extension_on_test(best_entry, X_torch, Y_torch, H_torch,
                                train_val_idx, test_idx,
                                neighbor_indices, masks_torch,
                                lr=1e-2, num_epochs=200, device="cpu",
                                alpha_window=7, alpha_ridge=1e-4):
    """
    Retrain the extension config on train+val and evaluate once on the test set.
    """
    if not best_entry:
        return None

    params = best_entry["params"]
    model = train_extended(
        X_torch, Y_torch, H_torch, train_val_idx,
        neighbor_indices, masks_torch,
        lam=params["lam"],
        lam7=params["lam7"],
        mu_wd=params["mu_wd"],
        mu_fri=params["mu_fri"],
        mu_we=params["mu_we"],
        gamma=params["gamma"],
        lr=lr, num_epochs=num_epochs, device=device, verbose=False
    )
    rmse, y_true, y_pred = eval_extended(
        model, X_torch, Y_torch, H_torch, train_val_idx, test_idx,
        alpha_window=alpha_window,
        alpha_ridge=alpha_ridge
    )
    return {
        "rmse": rmse,
        "params": params,
        "y_true": y_true,
        "y_pred": y_pred
    }


def evaluate_extension_impacts(X_torch, Y_torch,
                               H_zero, H_actual,
                               train_idx, val_idx,
                               neighbor_indices, masks_torch,
                               lam,
                               lam7_values,
                               mu_wd_values, mu_fri_values, mu_we_values,
                               gamma_values,
                               mu_default, lam_values=None, mu_values=None, gamma_off_value=0.0,
                               lr=1e-2, num_epochs=200, device="cpu",
                               alpha_window=7, alpha_ridge=1e-4,
                               early_stopping=False, early_stop_patience=20,
                               early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Run isolated sweeps for each extension to quantify its standalone impact.
    Other extensions are disabled via lam7=0, mu defaults, and zeroed holidays.
    """
    results = {}
    device = torch.device(device)
    lam_candidates = lam_values if lam_values is not None else [lam]
    mu_candidates = mu_values if mu_values is not None else [mu_default]

    if lam7_values:
        history = []
        best = None
        total = len(lam_candidates) * len(mu_candidates) * len(lam7_values)
        with tqdm(total=total, desc="Weekly-only sweep") as pbar:
            for lam_candidate in lam_candidates:
                for mu_candidate in mu_candidates:
                    for lam7 in lam7_values:
                        res = train_eval_extended_config(
                            X_torch, Y_torch, H_zero, train_idx, val_idx,
                            neighbor_indices, masks_torch,
                            lam=lam_candidate,
                            lam7=lam7,
                            mu_wd=mu_candidate,
                            mu_fri=mu_candidate,
                            mu_we=mu_candidate,
                            gamma=gamma_off_value,
                            lr=lr, num_epochs=num_epochs, device=device,
                            alpha_window=alpha_window,
                            alpha_ridge=alpha_ridge,
                            early_stopping=early_stopping,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_stop_min_delta,
                            early_stop_eval_every=early_stop_eval_every
                        )
                        history.append({
                            "lam": lam_candidate,
                            "mu": mu_candidate,
                            "lam7": lam7,
                            "rmse": res["rmse"],
                        })
                        if best is None or res["rmse"] < best["rmse"]:
                            best = res
                            pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                        pbar.update(1)
        results["weekly"] = {"best": best, "history": history}
        print(
            f"[Extension Sweep] Weekly-only best aRMSE={best['rmse']:.4f} "
            f"(lam={best['params'].get('lam')}, "
            f"mu={best['params'].get('mu_wd')}, "
            f"lam7={best['params']['lam7']})"
        )

    if mu_wd_values or mu_fri_values or mu_we_values:
        history = []
        best = None
        if not (mu_wd_values and mu_fri_values and mu_we_values):
            print("[Extension Sweep] Day-type-only sweep skipped: "
                  "mu_wd_values, mu_fri_values, and mu_we_values must all be non-empty.")
            results["daytype"] = {"best": None, "history": []}
        else:
            combos = len(lam_candidates) * len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values)
            with tqdm(total=combos, desc="Day-type-only sweep") as pbar:
                for lam_candidate in lam_candidates:
                    for mu_wd in mu_wd_values:
                        for mu_fri in mu_fri_values:
                            for mu_we in mu_we_values:
                                res = train_eval_extended_config(
                                    X_torch, Y_torch, H_zero, train_idx, val_idx,
                                    neighbor_indices, masks_torch,
                                    lam=lam_candidate,
                                    lam7=0.0,
                                    mu_wd=mu_wd,
                                    mu_fri=mu_fri,
                                    mu_we=mu_we,
                                    gamma=gamma_off_value,
                                    lr=lr, num_epochs=num_epochs, device=device,
                                    alpha_window=alpha_window,
                                    alpha_ridge=alpha_ridge,
                                    early_stopping=early_stopping,
                                    early_stop_patience=early_stop_patience,
                                    early_stop_min_delta=early_stop_min_delta,
                                    early_stop_eval_every=early_stop_eval_every
                                )
                                history.append({
                                    "lam": lam_candidate,
                                    "mu_wd": mu_wd,
                                    "mu_fri": mu_fri,
                                    "mu_we": mu_we,
                                    "rmse": res["rmse"]
                                })
                                if best is None or res["rmse"] < best["rmse"]:
                                    best = res
                                    pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                                pbar.update(1)
            results["daytype"] = {"best": best, "history": history}
            print("[Extension Sweep] Day-type-only best aRMSE="
                  f"{best['rmse']:.4f} "
                  f"(lam={best['params'].get('lam')}, "
                  f"mu_wd={best['params']['mu_wd']}, "
                  f"mu_fri={best['params']['mu_fri']}, "
                  f"mu_we={best['params']['mu_we']})")

    if gamma_values:
        history = []
        best = None
        total = len(lam_candidates) * len(mu_candidates) * len(gamma_values)
        with tqdm(total=total, desc="Holiday-only sweep") as pbar:
            for lam_candidate in lam_candidates:
                for mu_candidate in mu_candidates:
                    for gamma in gamma_values:
                        res = train_eval_extended_config(
                            X_torch, Y_torch, H_actual, train_idx, val_idx,
                            neighbor_indices, masks_torch,
                            lam=lam_candidate,
                            lam7=0.0,
                            mu_wd=mu_candidate,
                            mu_fri=mu_candidate,
                            mu_we=mu_candidate,
                            gamma=gamma,
                            lr=lr, num_epochs=num_epochs, device=device,
                            alpha_window=alpha_window,
                            alpha_ridge=alpha_ridge,
                            early_stopping=early_stopping,
                            early_stop_patience=early_stop_patience,
                            early_stop_min_delta=early_stop_min_delta,
                            early_stop_eval_every=early_stop_eval_every
                        )
                        history.append({
                            "lam": lam_candidate,
                            "mu": mu_candidate,
                            "gamma": gamma,
                            "rmse": res["rmse"],
                        })
                        if best is None or res["rmse"] < best["rmse"]:
                            best = res
                            pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                        pbar.update(1)
        results["holiday"] = {"best": best, "history": history}
        print(
            f"[Extension Sweep] Holiday-only best aRMSE={best['rmse']:.4f} "
            f"(lam={best['params'].get('lam')}, "
            f"mu={best['params'].get('mu_wd')}, "
            f"gamma={best['params']['gamma']})"
        )

    return results


def evaluate_extension_combinations(X_torch, Y_torch,
                                    H_zero, H_actual,
                                    train_idx, val_idx,
                                    neighbor_indices, masks_torch,
                                    lam,
                                    lam7_values, mu_wd_values, mu_fri_values, mu_we_values,
                                    gamma_values,
                                    mu_default, lam7_default, gamma_default,
                                    lr=1e-2, num_epochs=200, device="cpu",
                                    alpha_window=7, alpha_ridge=1e-4,
                                    early_stopping=False, early_stop_patience=20,
                                    early_stop_min_delta=0.0, early_stop_eval_every=5):
    """
    Evaluate pairwise extension combinations after individual tuning.
    Combinations:
      - weekly + day-type (holiday off)
      - weekly + holiday (day-type off)
      - day-type + holiday (weekly off)
    """
    results = {}
    device = torch.device(device)

    # Weekly + Day-type (holiday off)
    if lam7_values and mu_wd_values and mu_fri_values and mu_we_values:
        history = []
        best = None
        combos = len(lam7_values) * len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values)
        with tqdm(total=combos, desc="Weekly+Daytype sweep") as pbar:
            for lam7 in lam7_values:
                for mu_wd in mu_wd_values:
                    for mu_fri in mu_fri_values:
                        for mu_we in mu_we_values:
                            res = train_eval_extended_config(
                                X_torch, Y_torch, H_zero, train_idx, val_idx,
                                neighbor_indices, masks_torch,
                                lam=lam,
                                lam7=lam7,
                                mu_wd=mu_wd,
                                mu_fri=mu_fri,
                                mu_we=mu_we,
                                gamma=0.0,
                                lr=lr, num_epochs=num_epochs, device=device,
                                alpha_window=alpha_window,
                                alpha_ridge=alpha_ridge,
                                early_stopping=early_stopping,
                                early_stop_patience=early_stop_patience,
                                early_stop_min_delta=early_stop_min_delta,
                                early_stop_eval_every=early_stop_eval_every
                            )
                            history.append({
                                "lam7": lam7,
                                "mu_wd": mu_wd,
                                "mu_fri": mu_fri,
                                "mu_we": mu_we,
                                "rmse": res["rmse"]
                            })
                            if best is None or res["rmse"] < best["rmse"]:
                                best = res
                                pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                            pbar.update(1)
        results["weekly_daytype"] = {"best": best, "history": history}

    # Weekly + Holiday (day-type off)
    if lam7_values and gamma_values:
        history = []
        best = None
        total = len(lam7_values) * len(gamma_values)
        with tqdm(total=total, desc="Weekly+Holiday sweep") as pbar:
            for lam7 in lam7_values:
                for gamma in gamma_values:
                    res = train_eval_extended_config(
                        X_torch, Y_torch, H_actual, train_idx, val_idx,
                        neighbor_indices, masks_torch,
                        lam=lam,
                        lam7=lam7,
                        mu_wd=mu_default,
                        mu_fri=mu_default,
                        mu_we=mu_default,
                        gamma=gamma,
                        lr=lr, num_epochs=num_epochs, device=device,
                        alpha_window=alpha_window,
                        alpha_ridge=alpha_ridge,
                        early_stopping=early_stopping,
                        early_stop_patience=early_stop_patience,
                        early_stop_min_delta=early_stop_min_delta,
                        early_stop_eval_every=early_stop_eval_every
                    )
                    history.append({"lam7": lam7, "gamma": gamma, "rmse": res["rmse"]})
                    if best is None or res["rmse"] < best["rmse"]:
                        best = res
                        pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                    pbar.update(1)
        results["weekly_holiday"] = {"best": best, "history": history}

    # Day-type + Holiday (weekly off)
    if mu_wd_values and mu_fri_values and mu_we_values and gamma_values:
        history = []
        best = None
        combos = len(mu_wd_values) * len(mu_fri_values) * len(mu_we_values) * len(gamma_values)
        with tqdm(total=combos, desc="Daytype+Holiday sweep") as pbar:
            for mu_wd in mu_wd_values:
                for mu_fri in mu_fri_values:
                    for mu_we in mu_we_values:
                        for gamma in gamma_values:
                            res = train_eval_extended_config(
                                X_torch, Y_torch, H_actual, train_idx, val_idx,
                                neighbor_indices, masks_torch,
                                lam=lam,
                                lam7=0.0,
                                mu_wd=mu_wd,
                                mu_fri=mu_fri,
                                mu_we=mu_we,
                                gamma=gamma,
                                lr=lr, num_epochs=num_epochs, device=device,
                                alpha_window=alpha_window,
                                alpha_ridge=alpha_ridge,
                                early_stopping=early_stopping,
                                early_stop_patience=early_stop_patience,
                                early_stop_min_delta=early_stop_min_delta,
                                early_stop_eval_every=early_stop_eval_every
                            )
                            history.append({
                                "mu_wd": mu_wd,
                                "mu_fri": mu_fri,
                                "mu_we": mu_we,
                                "gamma": gamma,
                                "rmse": res["rmse"]
                            })
                            if best is None or res["rmse"] < best["rmse"]:
                                best = res
                                pbar.set_postfix({"best_rmse": f"{res['rmse']:.4f}"})
                            pbar.update(1)
        results["daytype_holiday"] = {"best": best, "history": history}

    # Fill defaults for logging/consistency if any sweeps were skipped.
    results["defaults"] = {
        "lam": lam,
        "lam7": lam7_default,
        "mu_default": mu_default,
        "gamma": gamma_default
    }
    return results
