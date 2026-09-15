"""Plotting and result-aggregation utilities used by the experiments."""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def visualize_baseline_grid(history, output_path="baseline_grid_search.png"):
    """
    Heatmap showing validation aRMSE as a function of lam/mu for the baseline grid search.
    """
    if not history:
        return

    df = pd.DataFrame(history)
    pivot = df.pivot(index="lam", columns="mu", values="rmse")
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(6, 4))
    im = plt.imshow(
        pivot.values,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        vmin=df["rmse"].min(),
        vmax=df["rmse"].max(),
    )
    plt.colorbar(im, shrink=0.8, label="aRMSE")
    plt.xticks(np.arange(len(pivot.columns)), [str(mu) for mu in pivot.columns], rotation=45)
    plt.yticks(np.arange(len(pivot.index)), [str(lam) for lam in pivot.index])
    plt.xlabel("mu")
    plt.ylabel("lam")
    plt.title("Baseline Grid Search (Validation aRMSE)")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved baseline grid-search visualization to '{output_path}'")


def visualize_extended_grid(history, output_path="extended_grid_search.png"):
    """
    Visualize the extended grid search by projecting aRMSE onto (lam7, mu_wd)
    for each gamma, after selecting the best mu_fri/mu_we combination.
    """
    if not history:
        return

    df = pd.DataFrame(history)
    reduced = (
        df.groupby(["gamma", "lam7", "mu_wd"])["rmse"]
        .min()
        .reset_index()
    )

    gammas = sorted(reduced["gamma"].unique())
    cols = min(3, len(gammas))
    rows = math.ceil(len(gammas) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(rows, cols)
    flat_axes = axes.ravel()

    vmin = reduced["rmse"].min()
    vmax = reduced["rmse"].max()

    im = None
    for idx, gamma in enumerate(gammas):
        ax = flat_axes[idx]
        subset = reduced[reduced["gamma"] == gamma]
        pivot = subset.pivot(index="lam7", columns="mu_wd", values="rmse")
        if pivot.empty:
            ax.set_visible(False)
            continue

        pivot = pivot.sort_index().sort_index(axis=1)
        im = ax.imshow(
            pivot.values,
            cmap="plasma",
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"gamma={gamma}")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(mu) for mu in pivot.columns], rotation=45)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(lam7) for lam7 in pivot.index])
        ax.set_xlabel("mu_wd")
    for ax in flat_axes[len(gammas):]:
        ax.axis("off")
    flat_axes[0].set_ylabel("lam7")
    if im is not None:
        fig.colorbar(im, ax=flat_axes.tolist(), shrink=0.8, label="aRMSE")
    fig.suptitle(
        "Extended Grid Search (Validation aRMSE, best mu_fri/mu_we)",
        fontsize=14,
    )
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extended grid-search visualization to '{output_path}'")


def visualize_extension_impacts(baseline_rmse, impact_results,
                                output_path="extension_impacts.png",
                                title="Single-Extension Validation Comparison",
                                y_label="Validation aRMSE"):
    """
    Bar chart comparing validation aRMSE of baseline vs single-extension sweeps.
    """
    labels = ["Baseline"]
    values = [baseline_rmse]
    label_map = [
        ("weekly", "Weekly only"),
        ("daytype", "Day-type only"),
        ("holiday", "Holiday only")
    ]
    for key, label in label_map:
        if key in impact_results and impact_results[key].get("best"):
            labels.append(label)
            values.append(impact_results[key]["best"]["rmse"])

    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(values)), values, color="#ff7f0e")
    plt.xticks(range(len(values)), labels, rotation=20)
    plt.ylabel(y_label)
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved extension comparison visualization to '{output_path}'")


def visualize_extension_combinations(baseline_rmse, extended_rmse, combo_results,
                                     output_path="extension_combos.png",
                                     title="Extension Combination Validation Comparison",
                                     y_label="Validation aRMSE"):
    """
    Bar chart comparing baseline, best single-extension sweeps, and combo sweeps.
    """
    labels = ["Baseline", "All three"]
    values = [baseline_rmse, extended_rmse]

    label_map = [
        ("weekly_daytype", "Weekly+Day-type"),
        ("weekly_holiday", "Weekly+Holiday"),
        ("daytype_holiday", "Day-type+Holiday")
    ]

    for key, label in label_map:
        entry = combo_results.get(key)
        if entry and entry.get("best"):
            labels.append(label)
            values.append(entry["best"]["rmse"])

    plt.figure(figsize=(8, 4))
    bars = plt.bar(range(len(values)), values, color="#2ca02c")
    plt.xticks(range(len(values)), labels, rotation=20)
    plt.ylabel(y_label)
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved extension combo comparison to '{output_path}'")


def visualize_baseline_sensitivity(history, output_path="baseline_parameter_sensitivity.png"):
    """
    Plot aRMSE sensitivity curves for baseline lam and mu using the grid-search history.
    """
    if not history:
        print("No baseline sensitivity data to plot.")
        return

    df = pd.DataFrame(history)
    panels = []

    if {"lam", "rmse"}.issubset(df.columns):
        agg = (
            df.dropna(subset=["lam", "rmse"])
            .groupby("lam")["rmse"]
            .min()
            .reset_index()
            .sort_values("lam")
        )
        if not agg.empty:
            panels.append({
                "title": "Temporal smoothness (lam)",
                "xlabel": "lam",
                "x": agg["lam"].to_numpy(),
                "y": agg["rmse"].to_numpy()
            })

    if {"mu", "rmse"}.issubset(df.columns):
        agg = (
            df.dropna(subset=["mu", "rmse"])
            .groupby("mu")["rmse"]
            .min()
            .reset_index()
            .sort_values("mu")
        )
        if not agg.empty:
            panels.append({
                "title": "Spatial smoothness (mu)",
                "xlabel": "mu",
                "x": agg["mu"].to_numpy(),
                "y": agg["rmse"].to_numpy()
            })

    if not panels:
        print("No baseline sensitivity data to plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.plot(panel["x"], panel["y"], marker="o")
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved baseline sensitivity visualization to '{output_path}'")


def visualize_extension_parameter_sensitivity(impact_results,
                                              output_path="extension_parameter_sensitivity.png"):
    """
    Plot aRMSE sensitivity curves for each extension parameter using the sweep
    histories produced by evaluate_extension_impacts.
    """
    if not impact_results:
        print("No extension sensitivity data to plot.")
        return

    panels = []

    weekly = impact_results.get("weekly")
    if weekly and weekly.get("history"):
        df = pd.DataFrame(weekly["history"])
        if not df.empty and {"lam7", "rmse"}.issubset(df.columns):
            df = (
                df.dropna(subset=["lam7", "rmse"])
                .groupby("lam7")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam7")
            )
            if not df.empty:
                panels.append({
                    "title": "Weekly coupling (lam7)",
                    "xlabel": "lam7",
                    "lines": [{
                        "x": df["lam7"].to_numpy(),
                        "y": df["rmse"].to_numpy(),
                        "label": "lam7"
                    }]
                })

    daytype = impact_results.get("daytype")
    if daytype and daytype.get("history"):
        df = pd.DataFrame(daytype["history"])
        for col, label in [("mu_wd", "Weekday mu"), ("mu_fri", "Friday mu"), ("mu_we", "Weekend mu")]:
            if col in df and "rmse" in df:
                agg = (
                    df.dropna(subset=[col, "rmse"])
                    .groupby(col)["rmse"]
                    .min()
                    .reset_index()
                    .sort_values(col)
                )
                if not agg.empty:
                    panels.append({
                        "title": f"Day-type spatial ({label})",
                        "xlabel": col,
                        "lines": [{
                            "x": agg[col].to_numpy(),
                            "y": agg["rmse"].to_numpy(),
                            "label": label
                        }]
                    })

    holiday = impact_results.get("holiday")
    if holiday and holiday.get("history"):
        df = pd.DataFrame(holiday["history"])
        if not df.empty and {"gamma", "rmse"}.issubset(df.columns):
            df = (
                df.dropna(subset=["gamma", "rmse"])
                .groupby("gamma")["rmse"]
                .min()
                .reset_index()
                .sort_values("gamma")
            )
            if not df.empty:
                panels.append({
                    "title": "Holiday penalty (gamma)",
                    "xlabel": "gamma",
                    "lines": [{
                        "x": df["gamma"].to_numpy(),
                        "y": df["rmse"].to_numpy(),
                        "label": "gamma"
                    }]
                })

    if not panels:
        print("No extension sensitivity data to plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        for line in panel["lines"]:
            ax.plot(line["x"], line["y"], marker="o", label=line["label"])
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
        if len(panel["lines"]) > 1:
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extension sensitivity visualization to '{output_path}'")


def visualize_extension_baseline_tuning(impact_results,
                                        output_path="extension_baseline_tuning.png"):
    """
    Plot how aRMSE varies with the baseline lam/mu values used during each
    extension sweep (weekly, day-type, holiday).
    """
    if not impact_results:
        print("No extension impact data; skipping baseline tuning plot.")
        return

    panels = []

    weekly = impact_results.get("weekly")
    if weekly and weekly.get("history"):
        df = pd.DataFrame(weekly["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Weekly extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })
        if "mu" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["mu", "rmse"])
                .groupby("mu")["rmse"]
                .min()
                .reset_index()
                .sort_values("mu")
            )
            if not agg.empty:
                panels.append({
                    "title": "Weekly extension: baseline mu",
                    "xlabel": "mu",
                    "x": agg["mu"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    daytype = impact_results.get("daytype")
    if daytype and daytype.get("history"):
        df = pd.DataFrame(daytype["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Day-type extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    holiday = impact_results.get("holiday")
    if holiday and holiday.get("history"):
        df = pd.DataFrame(holiday["history"])
        if "lam" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["lam", "rmse"])
                .groupby("lam")["rmse"]
                .min()
                .reset_index()
                .sort_values("lam")
            )
            if not agg.empty:
                panels.append({
                    "title": "Holiday extension: baseline lam",
                    "xlabel": "lam",
                    "x": agg["lam"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })
        if "mu" in df and "rmse" in df:
            agg = (
                df.dropna(subset=["mu", "rmse"])
                .groupby("mu")["rmse"]
                .min()
                .reset_index()
                .sort_values("mu")
            )
            if not agg.empty:
                panels.append({
                    "title": "Holiday extension: baseline mu",
                    "xlabel": "mu",
                    "x": agg["mu"].to_numpy(),
                    "y": agg["rmse"].to_numpy()
                })

    if not panels:
        print("No baseline lam/mu variation captured in extension sweeps.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        ax.plot(panel["x"], panel["y"], marker="o")
        ax.set_title(panel["title"])
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel("Validation aRMSE")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extension baseline tuning visualization to '{output_path}'")


def plot_evaluation_summary(results_df, columns, labels, title, output_path):
    """
    Plot mean ± standard deviation across the supplied evaluation splits.
    """
    if results_df.empty:
        print("No evaluation results to plot.")
        return
    means = []
    stds = []
    for col in columns:
        series = results_df[col].dropna()
        means.append(series.mean() if not series.empty else np.nan)
        stds.append(series.std(ddof=0) if not series.empty else np.nan)

    x = np.arange(len(columns))
    fig, ax = plt.subplots(figsize=(10, 4))
    yerr = stds if len(results_df) > 1 else None
    capsize = 4 if yerr is not None else 0
    ax.bar(x, means, yerr=yerr, capsize=capsize, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("aRMSE")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation summary plot to '{output_path}'")


def aggregate_grid_history(histories, group_cols):
    """
    Aggregate grid-search histories across splits by averaging aRMSE per config.
    """
    frames = []
    for hist in histories:
        if not hist:
            continue
        frames.append(pd.DataFrame(hist))
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    if "rmse" not in df.columns:
        return []
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return []
    agg = df.groupby(group_cols)["rmse"].mean().reset_index()
    return agg.to_dict("records")


def _aggregate_param_min_over_splits(histories, param_col):
    """
    For each split, take min RMSE per param value, then average across splits.
    """
    per_split = []
    for hist in histories:
        if not hist:
            continue
        df = pd.DataFrame(hist)
        if param_col not in df.columns or "rmse" not in df.columns:
            continue
        df = df.dropna(subset=[param_col, "rmse"])
        if df.empty:
            continue
        per_split.append(df.groupby(param_col)["rmse"].min())
    if not per_split:
        return []
    combined = pd.concat(per_split, axis=1)
    mean = combined.mean(axis=1)
    return [{param_col: idx, "rmse": float(val)} for idx, val in mean.items()]


def aggregate_extension_impacts_histories(impacts_list):
    """
    Aggregate extension impact histories into a single history per extension.
    """
    weekly_histories = []
    daytype_histories = []
    holiday_histories = []
    for impacts in impacts_list:
        weekly_histories.append(impacts.get("weekly", {}).get("history", []))
        daytype_histories.append(impacts.get("daytype", {}).get("history", []))
        holiday_histories.append(impacts.get("holiday", {}).get("history", []))

    weekly_history = []
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "lam7")
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "lam")
    weekly_history += _aggregate_param_min_over_splits(weekly_histories, "mu")

    daytype_history = []
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_wd")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_fri")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "mu_we")
    daytype_history += _aggregate_param_min_over_splits(daytype_histories, "lam")

    holiday_history = []
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "gamma")
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "lam")
    holiday_history += _aggregate_param_min_over_splits(holiday_histories, "mu")

    return {
        "weekly": {"history": weekly_history},
        "daytype": {"history": daytype_history},
        "holiday": {"history": holiday_history},
    }


def plot_weekly_periodicity(Y, max_lag=100, output_path="weekly_periodicity.png"):
    """
    Plot average |c_t - c_{t+Δt}| over temporal lags,
    where Y contains daily crime counts per region.

    Y: numpy array of shape (K, N)
    """
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Expected Y to have shape (K, N); got {Y.shape}")

    K, _ = Y.shape
    max_lag = int(max_lag)
    if K < 2:
        raise ValueError("Need at least 2 time steps to compute lag differences.")
    max_lag = max(1, min(max_lag, K - 1))

    lags = np.arange(1, max_lag + 1)
    avg_abs_diff = np.empty_like(lags, dtype=float)

    for idx, lag in enumerate(lags):
        diffs = np.abs(Y[lag:] - Y[:-lag])
        avg_abs_diff[idx] = float(np.nanmean(diffs))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags, avg_abs_diff, color="teal", linewidth=1.5)
    for m in range(7, max_lag + 1, 7):
        ax.axvline(m, color="black", alpha=0.08, linewidth=1)
    ax.set_title("Weekly Periodicity (Avg |c(t) − c(t+Δt)|)")
    ax.set_xlabel("Δt (days)")
    ax.set_ylabel("Avg absolute difference")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved weekly periodicity plot to '{output_path}'")


def plot_day_of_year(dates, Y, output_path="day_of_year_pattern.png"):
    """
    Plot average citywide daily crime by day of year.

    dates: pandas Index/DatetimeIndex of length K
    Y: numpy array of shape (K, N)
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(
            f"Expected dates length K and Y shape (K, N); "
            f"got len(dates)={len(dates)}, Y={Y.shape}"
        )

    citywide_daily = np.nansum(Y, axis=1)
    series = pd.Series(citywide_daily, index=dates).sort_index()

    doy = series.index.dayofyear
    doy_avg = series.groupby(doy).mean()

    x = np.arange(1, 367)
    y = np.full_like(x, np.nan, dtype=float)
    present = doy_avg.index.values.astype(int)
    y[present - 1] = doy_avg.values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="green", linewidth=1.5)
    ax.set_title("Day-of-Year Pattern (Avg Citywide Daily Crime)")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Avg daily crime (sum over regions)")
    ax.grid(True, alpha=0.2)

    # Month ticks (use a non-leap baseline year for readability)
    baseline = pd.date_range("2019-01-01", "2019-12-31", freq="MS")
    month_starts = baseline.dayofyear.values
    ax.set_xticks(month_starts)
    ax.set_xticklabels([d.strftime("%b") for d in baseline])

    # Reference lines for New Year's Day and Christmas
    new_year = pd.Timestamp("2019-01-01").dayofyear
    christmas = pd.Timestamp("2019-12-25").dayofyear
    ax.axvline(new_year, color="black", alpha=0.08, linewidth=1)
    ax.axvline(christmas, color="black", alpha=0.08, linewidth=1)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved day-of-year plot to '{output_path}'")


def plot_dayofweek_spatial(
    grid_active,
    regions,
    dates,
    Y,
    output_path="day_of_week_spatial_distribution.png",
):
    """
    Plot seven maps (Monday through Sunday) of average crime per region.

    grid_active: GeoDataFrame with 'region_id' and 'geometry'
    regions: region_ids aligned with Y's 2nd dimension
    dates: pandas Index/DatetimeIndex of length K
    Y: numpy array of shape (K, N)
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(
            f"Expected dates length K and Y shape (K, N); "
            f"got len(dates)={len(dates)}, Y={Y.shape}"
        )
    if "region_id" not in grid_active.columns:
        raise ValueError("grid_active must contain a 'region_id' column for joining to regions.")

    regions = np.asarray(regions)
    grid_plot = grid_active.set_index("region_id").loc[regions].reset_index()

    dow = dates.dayofweek.values  # 0=Mon ... 6=Sun
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    region_means = []
    for d in range(7):
        mask = dow == d
        if not mask.any():
            region_means.append(np.full(Y.shape[1], np.nan))
        else:
            region_means.append(np.nanmean(Y[mask], axis=0))

    finite = [v[np.isfinite(v)] for v in region_means if np.isfinite(v).any()]
    all_vals = np.concatenate(finite, axis=0) if finite else np.array([], dtype=float)
    vmax = float(np.nanpercentile(all_vals, 99)) if all_vals.size else 1.0
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for d in range(7):
        ax = axes[d]
        plot_df = grid_plot.copy()
        plot_df["avg_crime"] = region_means[d]
        plot_df.plot(
            column="avg_crime",
            ax=ax,
            legend=(d == 6),
            cmap="Blues",
            vmin=0.0,
            vmax=vmax,
            missing_kwds={"color": "lightgrey", "label": "No data"},
        )
        ax.set_title(dow_names[d])
        ax.axis("off")

    axes[7].axis("off")
    plt.suptitle("Spatial Distribution by Day of Week (Avg Crime per Region)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved day-of-week spatial plot to '{output_path}'")


def summarize_daytype_spatial_differences(regions, dates, Y,
                                          output_csv="daytype_spatial_summary.csv",
                                          top_n=10):
    """
    Quantify spatial differences between day types with region-wise averages.

    Outputs a CSV with per-region means and pairwise differences, and prints
    aggregate summary statistics that can be reported numerically.
    """
    dates = pd.to_datetime(pd.Index(dates))
    Y = np.asarray(Y)
    if Y.ndim != 2 or len(dates) != Y.shape[0]:
        raise ValueError(
            f"Expected dates length K and Y shape (K, N); "
            f"got len(dates)={len(dates)}, Y={Y.shape}"
        )

    dow = dates.dayofweek.values  # 0=Mon ... 6=Sun
    masks = {
        "wd": (dow >= 0) & (dow <= 3),   # Mon-Thu
        "fri": (dow == 4),
        "we": (dow >= 5)                 # Sat-Sun
    }

    regions = np.asarray(regions)
    stats = {"region_id": regions}
    for key, mask in masks.items():
        if mask.any():
            stats[f"mean_{key}"] = np.nanmean(Y[mask], axis=0)
        else:
            stats[f"mean_{key}"] = np.full(Y.shape[1], np.nan)

    df = pd.DataFrame(stats)
    df["diff_fri_wd"] = df["mean_fri"] - df["mean_wd"]
    df["diff_we_wd"] = df["mean_we"] - df["mean_wd"]
    df["diff_we_fri"] = df["mean_we"] - df["mean_fri"]
    df["absdiff_fri_wd"] = df["diff_fri_wd"].abs()
    df["absdiff_we_wd"] = df["diff_we_wd"].abs()
    df["absdiff_we_fri"] = df["diff_we_fri"].abs()

    df.to_csv(output_csv, index=False)

    def _series_stats(series):
        series = series[np.isfinite(series)]
        if series.empty:
            return {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "pct_positive": np.nan,
                "pct_negative": np.nan,
            }
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "pct_positive": float((series > 0).mean() * 100.0),
            "pct_negative": float((series < 0).mean() * 100.0),
        }

    def _corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 2:
            return np.nan
        return float(np.corrcoef(a[mask], b[mask])[0, 1])

    summary = {
        "fri_vs_wd": _series_stats(df["diff_fri_wd"]),
        "we_vs_wd": _series_stats(df["diff_we_wd"]),
        "we_vs_fri": _series_stats(df["diff_we_fri"]),
        "corr_fri_wd": _corr(df["mean_fri"].values, df["mean_wd"].values),
        "corr_we_wd": _corr(df["mean_we"].values, df["mean_wd"].values),
        "corr_we_fri": _corr(df["mean_we"].values, df["mean_fri"].values),
    }

    print("\n--- Day-Type Spatial Difference Summary ---")
    print(f"Saved per-region stats to '{output_csv}'")
    for key, stats in summary.items():
        if key.startswith("corr_"):
            print(f"{key}: {stats:.3f}" if np.isfinite(stats) else f"{key}: n/a")
            continue
        print(
            f"{key}: mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"std={stats['std']:.3f}, +%={stats['pct_positive']:.1f}, -%={stats['pct_negative']:.1f}"
        )

    if top_n and top_n > 0:
        top_cols = ["absdiff_fri_wd", "absdiff_we_wd", "absdiff_we_fri"]
        for col in top_cols:
            top = df.nlargest(top_n, col)[["region_id", col]]
            print(f"Top {top_n} regions by {col}:")
            print(top.to_string(index=False))

    return df, summary


def generate_pattern_plots(Y, dates, grid_active, regions, max_lag=100, output_dir="."):
    """Generate temporal and spatial diagnostic plots used in the analysis."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_weekly_periodicity(
        Y,
        max_lag=max_lag,
        output_path=str(output_dir / "weekly_periodicity.png"),
    )
    plot_day_of_year(
        dates,
        Y,
        output_path=str(output_dir / "day_of_year_pattern.png"),
    )
    plot_dayofweek_spatial(
        grid_active,
        regions,
        dates,
        Y,
        output_path=str(output_dir / "day_of_week_spatial_distribution.png"),
    )


def plot_spatial_prediction_comparison(
    grid_active,
    y_true,
    y_pred,
    test_idx,
    dates,
    n_regions,
    output_path,
    day_idx=0,
):
    """Plot observed and predicted crime intensity for one test day."""
    k_test = len(test_idx)
    y_true_reshaped = y_true.reshape(k_test, n_regions)
    y_pred_reshaped = y_pred.reshape(k_test, n_regions)

    plot_global_idx = test_idx[day_idx].item()
    date_str = str(dates[plot_global_idx].date())

    plot_df = grid_active.copy()
    plot_df["Actual"] = y_true_reshaped[day_idx]
    plot_df["Predicted"] = y_pred_reshaped[day_idx]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_df.plot(column="Actual", ax=axes[0], legend=True, cmap="OrRd", vmin=0, vmax=20)
    axes[0].set_title(f"Baseline Actual - {date_str}")
    axes[0].axis("off")

    plot_df.plot(column="Predicted", ax=axes[1], legend=True, cmap="OrRd", vmin=0, vmax=20)
    axes[1].set_title(f"Baseline Predicted - {date_str}")
    axes[1].axis("off")

    plt.suptitle("Baseline Spatial Distribution (Test Set, First Day)", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
