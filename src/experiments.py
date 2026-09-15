"""Experiment orchestration for the TCP crime-prediction pipeline."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from src.data import (
    build_neighbor_pairs,
    get_holiday_tensor,
    process_neighbor_indices,
    reshape_for_tcp,
)
from src.evaluation import build_rolling_splits, eval_baseline, eval_extended
from src.training import (
    eval_best_extension_on_test,
    evaluate_extension_combinations,
    evaluate_extension_impacts,
    grid_search_baseline,
    grid_search_extended,
    train_baseline,
    train_extended,
)
from src.visualization import (
    aggregate_extension_impacts_histories,
    aggregate_grid_history,
    generate_pattern_plots,
    plot_evaluation_summary,
    plot_spatial_prediction_comparison,
    summarize_daytype_spatial_differences,
    visualize_baseline_grid,
    visualize_baseline_sensitivity,
    visualize_extended_grid,
    visualize_extension_baseline_tuning,
    visualize_extension_parameter_sensitivity,
)


FEATURE_COLUMNS = (
    "sas_count",
    "311_count",
    "checkin_count",
    "taxi_count",
    "PRCP",
    "SNOW",
    "TMIN",
    "TMAX",
)
TARGET_COLUMN = "complaint_count"


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters and evaluation settings for an experiment run."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    learning_rate: float = 1e-2
    alpha_window: int = 28
    alpha_ridge: float = 1e-3
    sweep_epochs: int = 100
    final_epochs: int = 200
    early_stopping: bool = True
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    early_stop_eval_every: int = 5
    lam_grid: tuple[int, ...] = (1_000_000, 2_000_000, 3_000_000, 5_000_000)
    mu_grid: tuple[int, ...] = (0, 1_000, 2_000, 3_000)
    lam7_grid: tuple[int, ...] = (0, 5_000, 10_000)
    mu_wd_grid: tuple[int, ...] = (0, 200, 500)
    mu_fri_grid: tuple[int, ...] = (0, 1_000, 2_000)
    mu_we_grid: tuple[int, ...] = (0, 1_000, 5_000)
    gamma_grid: tuple[int, ...] = (0, 1, 5)

    @classmethod
    def for_mode(cls, mode: str) -> "ExperimentConfig":
        if mode == "full":
            return cls()
        if mode == "smoke":
            return cls(
                sweep_epochs=2,
                final_epochs=3,
                early_stopping=False,
                lam_grid=(1_000_000,),
                mu_grid=(1_000,),
                lam7_grid=(5_000,),
                mu_wd_grid=(200,),
                mu_fri_grid=(1_000,),
                mu_we_grid=(1_000,),
                gamma_grid=(1,),
            )
        raise ValueError(f"Unsupported experiment mode: {mode}")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_paths(project_root: Path, mode: str) -> tuple[Path, Path, Path]:
    data_dir = project_root / "data"
    required = (
        data_dir / "FEATURE_MATRIX.csv",
        data_dir / "nyc_grid_2km_active.shp",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required processed data files were not found:\n"
            f"{details}\n"
            "See data/README.md or notebooks/build_feature_matrix.ipynb."
        )

    result_root = project_root / "results"
    if mode == "smoke":
        result_root = result_root / "smoke"
    figures_dir = result_root / "figures"
    tables_dir = result_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, figures_dir, tables_dir


def _load_data(data_dir: Path, device: torch.device):
    feature_df = pd.read_csv(data_dir / "FEATURE_MATRIX.csv")
    grid = gpd.read_file(data_dir / "nyc_grid_2km_active.shp")
    X, Y, regions, dates, masks = reshape_for_tcp(
        feature_df,
        feature_cols=list(FEATURE_COLUMNS),
        target_col=TARGET_COLUMN,
    )

    X_torch = torch.from_numpy(X).float().to(device)
    Y_torch = torch.from_numpy(Y).float().to(device)
    holidays = get_holiday_tensor(dates, device)
    zero_holidays = torch.zeros_like(holidays)
    masks_torch = {
        name: torch.from_numpy(mask).to(device)
        for name, mask in masks.items()
    }
    return grid, Y, regions, dates, X_torch, Y_torch, holidays, zero_holidays, masks_torch


def _make_splits(n_days: int, device: torch.device, config: ExperimentConfig):
    train_size = max(1, int(config.train_ratio * n_days))
    val_size = max(1, int(config.val_ratio * n_days))
    test_size = max(1, n_days - train_size - val_size)
    splits = build_rolling_splits(
        n_days,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=val_size,
        device=device,
    )
    if not splits:
        raise ValueError("Temporal split configuration produced no splits.")
    return splits


def _run_extension_test(
    best_entry,
    holiday_tensor,
    X_torch,
    Y_torch,
    train_val_idx,
    test_idx,
    neighbor_indices,
    masks_torch,
    config: ExperimentConfig,
    device: torch.device,
):
    return eval_best_extension_on_test(
        best_entry,
        X_torch,
        Y_torch,
        holiday_tensor,
        train_val_idx,
        test_idx,
        neighbor_indices,
        masks_torch,
        lr=config.learning_rate,
        num_epochs=config.final_epochs,
        device=device,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
    )


def _run_split(
    split,
    X_torch,
    Y_torch,
    holidays,
    zero_holidays,
    neighbor_indices,
    masks_torch,
    config: ExperimentConfig,
    device: torch.device,
):
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]
    train_val_idx = split["train_val_idx"]

    baseline_search = grid_search_baseline(
        X_torch,
        Y_torch,
        train_idx,
        val_idx,
        neighbor_indices,
        lam_values=config.lam_grid,
        mu_values=config.mu_grid,
        lr=config.learning_rate,
        num_epochs=config.sweep_epochs,
        device=device,
        viz_path=None,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
        early_stopping=config.early_stopping,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        early_stop_eval_every=config.early_stop_eval_every,
    )
    baseline_params = baseline_search["params"]
    baseline_model = train_baseline(
        X_torch,
        Y_torch,
        train_val_idx,
        neighbor_indices,
        lam=baseline_params["lam"],
        mu=baseline_params["mu"],
        lr=config.learning_rate,
        num_epochs=config.final_epochs,
        device=device,
        verbose=False,
    )
    baseline_rmse, y_true, y_pred = eval_baseline(
        baseline_model,
        X_torch,
        Y_torch,
        train_val_idx,
        test_idx,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
    )

    lam_best = baseline_params["lam"]
    mu_best = baseline_params["mu"]
    impacts = evaluate_extension_impacts(
        X_torch,
        Y_torch,
        zero_holidays,
        holidays,
        train_idx,
        val_idx,
        neighbor_indices,
        masks_torch,
        lam=lam_best,
        lam_values=config.lam_grid,
        lam7_values=config.lam7_grid,
        mu_wd_values=config.mu_wd_grid,
        mu_fri_values=config.mu_fri_grid,
        mu_we_values=config.mu_we_grid,
        gamma_values=config.gamma_grid,
        mu_default=mu_best,
        mu_values=config.mu_grid,
        gamma_off_value=0.0,
        lr=config.learning_rate,
        num_epochs=config.sweep_epochs,
        device=device,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
        early_stopping=config.early_stopping,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        early_stop_eval_every=config.early_stop_eval_every,
    )

    impact_test = {}
    for name, holiday_tensor in (
        ("weekly", zero_holidays),
        ("daytype", zero_holidays),
        ("holiday", holidays),
    ):
        result = _run_extension_test(
            impacts.get(name, {}).get("best"),
            holiday_tensor,
            X_torch,
            Y_torch,
            train_val_idx,
            test_idx,
            neighbor_indices,
            masks_torch,
            config,
            device,
        )
        if result:
            impact_test[name] = result

    combinations = evaluate_extension_combinations(
        X_torch,
        Y_torch,
        zero_holidays,
        holidays,
        train_idx,
        val_idx,
        neighbor_indices,
        masks_torch,
        lam=lam_best,
        lam7_values=config.lam7_grid,
        mu_wd_values=config.mu_wd_grid,
        mu_fri_values=config.mu_fri_grid,
        mu_we_values=config.mu_we_grid,
        gamma_values=config.gamma_grid,
        mu_default=mu_best,
        lam7_default=0.0,
        gamma_default=0.0,
        lr=config.learning_rate,
        num_epochs=config.sweep_epochs,
        device=device,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
        early_stopping=config.early_stopping,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        early_stop_eval_every=config.early_stop_eval_every,
    )

    combination_test = {}
    for name, holiday_tensor in (
        ("weekly_daytype", zero_holidays),
        ("weekly_holiday", holidays),
        ("daytype_holiday", holidays),
    ):
        result = _run_extension_test(
            combinations.get(name, {}).get("best"),
            holiday_tensor,
            X_torch,
            Y_torch,
            train_val_idx,
            test_idx,
            neighbor_indices,
            masks_torch,
            config,
            device,
        )
        if result:
            combination_test[name] = result

    extended_search = grid_search_extended(
        X_torch,
        Y_torch,
        holidays,
        train_idx,
        val_idx,
        neighbor_indices,
        masks_torch,
        lam=lam_best,
        lam_values=config.lam_grid,
        lam7_values=config.lam7_grid,
        mu_wd_values=config.mu_wd_grid,
        mu_fri_values=config.mu_fri_grid,
        mu_we_values=config.mu_we_grid,
        gamma_values=config.gamma_grid,
        lr=config.learning_rate,
        num_epochs=config.sweep_epochs,
        device=device,
        viz_path=None,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
        early_stopping=config.early_stopping,
        early_stop_patience=config.early_stop_patience,
        early_stop_min_delta=config.early_stop_min_delta,
        early_stop_eval_every=config.early_stop_eval_every,
    )
    extended_params = extended_search["params"]
    extended_model = train_extended(
        X_torch,
        Y_torch,
        holidays,
        train_val_idx,
        neighbor_indices,
        masks_torch,
        lam=extended_params["lam"],
        lam7=extended_params["lam7"],
        mu_wd=extended_params["mu_wd"],
        mu_fri=extended_params["mu_fri"],
        mu_we=extended_params["mu_we"],
        gamma=extended_params["gamma"],
        lr=config.learning_rate,
        num_epochs=config.final_epochs,
        device=device,
        verbose=False,
    )
    extended_rmse, _, _ = eval_extended(
        extended_model,
        X_torch,
        Y_torch,
        holidays,
        train_val_idx,
        test_idx,
        alpha_window=config.alpha_window,
        alpha_ridge=config.alpha_ridge,
    )

    metrics = {
        "baseline_val": baseline_search.get("rmse"),
        "baseline": baseline_rmse,
        "weekly_val": impacts.get("weekly", {}).get("best", {}).get("rmse"),
        "daytype_val": impacts.get("daytype", {}).get("best", {}).get("rmse"),
        "holiday_val": impacts.get("holiday", {}).get("best", {}).get("rmse"),
        "weekly": impact_test.get("weekly", {}).get("rmse"),
        "daytype": impact_test.get("daytype", {}).get("rmse"),
        "holiday": impact_test.get("holiday", {}).get("rmse"),
        "weekly_daytype_val": combinations.get("weekly_daytype", {}).get("best", {}).get("rmse"),
        "weekly_holiday_val": combinations.get("weekly_holiday", {}).get("best", {}).get("rmse"),
        "daytype_holiday_val": combinations.get("daytype_holiday", {}).get("best", {}).get("rmse"),
        "weekly_daytype": combination_test.get("weekly_daytype", {}).get("rmse"),
        "weekly_holiday": combination_test.get("weekly_holiday", {}).get("rmse"),
        "daytype_holiday": combination_test.get("daytype_holiday", {}).get("rmse"),
        "all_three_val": extended_search.get("rmse"),
        "all_three": extended_rmse,
    }
    artifacts = {
        "baseline_history": baseline_search.get("history", []),
        "impacts": impacts,
        "extended_history": extended_search.get("history", []),
        "baseline_predictions": (y_true, y_pred),
        "test_idx": test_idx,
    }
    return metrics, artifacts


def _plot_summaries(
    results: pd.DataFrame,
    evaluation_name: str,
    n_splits: int,
    figures_dir: Path,
) -> None:
    suffix = " (Mean ± Std)" if n_splits > 1 else ""
    specs = (
        (
            ["baseline", "weekly", "daytype", "holiday", "all_three"],
            ["Baseline", "Weekly", "Day-type", "Holiday", "All three"],
            f"{evaluation_name} Test Comparison{suffix}",
            "test_comparison.png",
        ),
        (
            ["baseline", "weekly_daytype", "weekly_holiday", "daytype_holiday", "all_three"],
            ["Baseline", "Weekly+Daytype", "Weekly+Holiday", "Daytype+Holiday", "All three"],
            f"{evaluation_name} Test Combination Comparison{suffix}",
            "test_combinations.png",
        ),
        (
            ["baseline_val", "weekly_val", "daytype_val", "holiday_val", "all_three_val"],
            ["Baseline", "Weekly", "Day-type", "Holiday", "All three"],
            f"{evaluation_name} Validation Tuning{suffix}",
            "validation_tuning.png",
        ),
    )
    for columns, labels, title, filename in specs:
        plot_evaluation_summary(
            results,
            columns=columns,
            labels=labels,
            title=title,
            output_path=str(figures_dir / filename),
        )


def _plot_search_diagnostics(
    baseline_histories,
    impact_histories,
    extended_histories,
    figures_dir: Path,
) -> None:
    baseline = aggregate_grid_history(baseline_histories, ["lam", "mu"])
    if baseline:
        visualize_baseline_grid(baseline, str(figures_dir / "baseline_grid.png"))
        visualize_baseline_sensitivity(
            baseline,
            output_path=str(figures_dir / "baseline_sensitivity.png"),
        )

    extended = aggregate_grid_history(
        extended_histories,
        ["lam", "lam7", "mu_wd", "mu_fri", "mu_we", "gamma"],
    )
    if extended:
        visualize_extended_grid(extended, str(figures_dir / "extended_grid.png"))

    impacts = aggregate_extension_impacts_histories(impact_histories)
    if impacts:
        visualize_extension_parameter_sensitivity(
            impacts,
            output_path=str(figures_dir / "extension_parameter_sensitivity.png"),
        )
        visualize_extension_baseline_tuning(
            impacts,
            output_path=str(figures_dir / "extension_baseline_tuning.png"),
        )


def run_experiments(
    project_root: Path,
    mode: str = "full",
    device_name: str = "auto",
    seed: int | None = None,
    run_diagnostics: bool = True,
) -> pd.DataFrame:
    """Run the configured experiment pipeline and return the evaluation table."""
    config = ExperimentConfig.for_mode(mode)
    device = _resolve_device(device_name)
    _set_seed(seed)
    data_dir, figures_dir, tables_dir = _prepare_paths(project_root, mode)

    print("Loading processed data...")
    (
        grid,
        Y,
        regions,
        dates,
        X_torch,
        Y_torch,
        holidays,
        zero_holidays,
        masks_torch,
    ) = _load_data(data_dir, device)

    if run_diagnostics:
        print("\n--- Temporal and spatial diagnostics ---")
        generate_pattern_plots(
            Y=Y,
            dates=dates,
            grid_active=grid,
            regions=regions,
            max_lag=100,
            output_dir=figures_dir,
        )
        summarize_daytype_spatial_differences(
            regions=regions,
            dates=dates,
            Y=Y,
            output_csv=str(tables_dir / "daytype_spatial_summary.csv"),
            top_n=10,
        )

    n_days, n_regions, _ = X_torch.shape
    neighbor_pairs = build_neighbor_pairs(grid, regions)
    neighbor_indices = process_neighbor_indices(neighbor_pairs, device=device)
    splits = _make_splits(n_days, device, config)
    evaluation_name = "Temporal Holdout" if len(splits) == 1 else "Rolling CV"

    print("\n--- Run configuration ---")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Dataset: {n_days} days x {n_regions} regions")
    print(f"Evaluation: {evaluation_name} ({len(splits)} split(s))")
    print(f"Alpha history window: {config.alpha_window} days")
    print(f"Random seed: {seed if seed is not None else 'not fixed'}")

    rows = []
    baseline_histories = []
    impact_histories = []
    extended_histories = []
    last_predictions = None
    last_test_idx = None

    for split_number, split in enumerate(splits, start=1):
        print(f"\n--- Split {split_number}/{len(splits)} ---")
        metrics, artifacts = _run_split(
            split,
            X_torch,
            Y_torch,
            holidays,
            zero_holidays,
            neighbor_indices,
            masks_torch,
            config,
            device,
        )
        rows.append({"split": split_number, **metrics})
        baseline_histories.append(artifacts["baseline_history"])
        impact_histories.append(artifacts["impacts"])
        extended_histories.append(artifacts["extended_history"])
        last_predictions = artifacts["baseline_predictions"]
        last_test_idx = artifacts["test_idx"]

    results = pd.DataFrame(rows)
    results_path = tables_dir / "evaluation_summary.csv"
    results.to_csv(results_path, index=False)

    print(f"\n--- {evaluation_name} summary ---")
    for column in (col for col in results.columns if col != "split"):
        values = results[column].dropna()
        if values.empty:
            continue
        if len(values) == 1:
            print(f"{column}: {values.iloc[0]:.4f}")
        else:
            print(f"{column}: mean={values.mean():.4f}, std={values.std(ddof=0):.4f}")
    print(f"Saved evaluation table to '{results_path}'")

    _plot_summaries(results, evaluation_name, len(splits), figures_dir)
    _plot_search_diagnostics(
        baseline_histories,
        impact_histories,
        extended_histories,
        figures_dir,
    )

    if last_predictions is not None and last_test_idx is not None:
        y_true, y_pred = last_predictions
        spatial_path = figures_dir / "spatial_prediction_baseline.png"
        plot_spatial_prediction_comparison(
            grid_active=grid,
            y_true=y_true,
            y_pred=y_pred,
            test_idx=last_test_idx,
            dates=dates,
            n_regions=n_regions,
            output_path=str(spatial_path),
        )
        print(f"Saved spatial comparison to '{spatial_path}'")

    return results
