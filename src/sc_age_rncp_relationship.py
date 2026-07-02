#!/usr/bin/env python3
"""Sleep-EDF SC age-RNCP subject-level relationships."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def corr_row(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    part = df[[x_col, y_col]].dropna()
    if len(part) < 3:
        return {"metric": y_col, "n": int(len(part)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(part[x_col], part[y_col])
    sr, sp = stats.spearmanr(part[x_col], part[y_col])
    slope, intercept, r_value, p_value, stderr = stats.linregress(part[x_col], part[y_col])
    return {
        "metric": y_col,
        "n": int(len(part)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "linear_slope_per_year": float(slope),
        "linear_intercept": float(intercept),
        "linear_p": float(p_value),
        "linear_slope_stderr": float(stderr),
    }


def load_subject_summary(table_dir: Path, high_quantile: float) -> tuple[pd.DataFrame, float]:
    path = table_dir / "functional_sleep_edf_sc_epoch_anchor_frame.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "rncp_epoch_mean",
        "rncp_epoch_z",
        "age",
        "sex",
        "exit_within_2min",
        "has_2min_followup",
        "n3_bout_cluster",
        "n3_bout_duration_min",
        "time_since_sleep_onset",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    for col in [
        "rncp_epoch_mean",
        "rncp_epoch_z",
        "age",
        "n3_bout_duration_min",
        "time_since_sleep_onset",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    threshold = float(df["rncp_epoch_mean"].quantile(high_quantile))
    df["high_rncp_epoch"] = df["rncp_epoch_mean"] >= threshold
    df["exit_within_2min_bool"] = bool_series(df["exit_within_2min"]) & bool_series(df["has_2min_followup"])

    night = (
        df.groupby(["subject_id", "night_id"], as_index=False)
        .agg(
            age=("age", "first"),
            sex=("sex", "first"),
            n_epochs=("epoch_idx", "nunique"),
            n3_duration_min=("epoch_idx", lambda s: float(s.nunique() * 0.5)),
            mean_rncp_magnitude=("rncp_epoch_mean", "mean"),
            median_rncp_magnitude=("rncp_epoch_mean", "median"),
            mean_rncp_z=("rncp_epoch_z", "mean"),
            sd_rncp_z=("rncp_epoch_z", "std"),
            high_rncp_proportion=("high_rncp_epoch", "mean"),
            exit_within_2min_rate=("exit_within_2min_bool", "mean"),
            n3_bout_count=("n3_bout_cluster", "nunique"),
            median_bout_duration_min=("n3_bout_duration_min", "median"),
            mean_relative_delta_power=("relative_delta_power", "mean"),
            mean_slow_wave_density=("slow_wave_density", "mean"),
            mean_slow_wave_occupancy=("slow_wave_occupancy", "mean"),
        )
    )
    night["n3_fragmentation_per_hour"] = night["n3_bout_count"] / (night["n3_duration_min"] / 60.0)

    subject = (
        night.groupby("subject_id", as_index=False)
        .agg(
            age=("age", "first"),
            sex=("sex", "first"),
            n_nights=("night_id", "nunique"),
            n_epochs=("n_epochs", "sum"),
            n3_duration_min=("n3_duration_min", "sum"),
            mean_rncp_magnitude=("mean_rncp_magnitude", "mean"),
            median_rncp_magnitude=("median_rncp_magnitude", "mean"),
            mean_rncp_z=("mean_rncp_z", "mean"),
            sd_rncp_z=("sd_rncp_z", "mean"),
            high_rncp_proportion=("high_rncp_proportion", "mean"),
            exit_within_2min_rate=("exit_within_2min_rate", "mean"),
            n3_bout_count=("n3_bout_count", "sum"),
            median_bout_duration_min=("median_bout_duration_min", "median"),
            n3_fragmentation_per_hour=("n3_fragmentation_per_hour", "mean"),
            mean_relative_delta_power=("mean_relative_delta_power", "mean"),
            mean_slow_wave_density=("mean_slow_wave_density", "mean"),
            mean_slow_wave_occupancy=("mean_slow_wave_occupancy", "mean"),
        )
    )
    return subject.sort_values("age").reset_index(drop=True), threshold


def write_summary(out_dir: Path, corr: pd.DataFrame, subject: pd.DataFrame, threshold: float, high_quantile: float) -> None:
    lines = [
        "# Sleep-EDF SC Age-RNCP Relationships",
        "",
        f"Subjects: {subject['subject_id'].nunique()}",
        f"Age range: {subject['age'].min():.0f}-{subject['age'].max():.0f} years",
        f"High-RNCP threshold: top {1 - high_quantile:.0%} of SC epoch RNCP magnitude, threshold={threshold:.6g}",
        "",
        "## Age correlations",
        "",
        corr.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        "- `sc_age_age_subject_summary.csv`",
        "- `sc_age_age_correlations.csv`",
    ]
    (out_dir / "sc_age_rncp_relationship_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    parser.add_argument("--out-prefix", default="sc_age")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    subject, threshold = load_subject_summary(table_dir, args.high_quantile)
    metrics = [
        "mean_rncp_magnitude",
        "median_rncp_magnitude",
        "mean_rncp_z",
        "sd_rncp_z",
        "high_rncp_proportion",
        "exit_within_2min_rate",
        "n3_duration_min",
        "n3_fragmentation_per_hour",
        "median_bout_duration_min",
        "mean_relative_delta_power",
        "mean_slow_wave_density",
        "mean_slow_wave_occupancy",
    ]
    corr = pd.DataFrame([corr_row(subject, "age", metric) for metric in metrics])

    prefix = args.out_prefix
    subject.to_csv(table_dir / f"{prefix}_age_subject_summary.csv", index=False)
    corr.to_csv(table_dir / f"{prefix}_age_correlations.csv", index=False)
    write_summary(table_dir, corr, subject, threshold, args.high_quantile)

    print(corr.to_string(index=False))
    print(f"Wrote analysis outputs to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
