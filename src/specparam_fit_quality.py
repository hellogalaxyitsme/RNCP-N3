#!/usr/bin/env python3
"""Specparam fit-quality summaries and threshold exclusion rates."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    display: str
    matrix_path: str
    primary_rncp_rows: int | None = None


SPECS = [
    DatasetSpec(
        dataset="sleep_edf_sc",
        display="Sleep-EDF SC",
        matrix_path="artifact_sleep_edf_sc_n3_matrix_artifact_augmented.csv.gz",
        primary_rncp_rows=23136,
    ),
    DatasetSpec(
        dataset="sleep_edf_st",
        display="Sleep-EDF ST",
        matrix_path="artifact_sleep_edf_st_n3_matrix_artifact_augmented.csv.gz",
        primary_rncp_rows=12215,
    ),
    DatasetSpec(
        dataset="anphy",
        display="ANPHY-Sleep",
        matrix_path="artifact_anphy_n3_matrix_artifact_augmented.csv.gz",
        primary_rncp_rows=350578,
    ),
]

THRESHOLDS = [0.10, 0.12, 0.15]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def summarize_distribution(values: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if vals.empty:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "sd": float(vals.std(ddof=1)),
        "p01": float(vals.quantile(0.01)),
        "p05": float(vals.quantile(0.05)),
        "p25": float(vals.quantile(0.25)),
        "median": float(vals.median()),
        "p75": float(vals.quantile(0.75)),
        "p95": float(vals.quantile(0.95)),
        "p99": float(vals.quantile(0.99)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def denominator_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "analysis_primary_cohort" in df:
        mask &= bool_series(df["analysis_primary_cohort"])
    elif "cohort_usable" in df:
        mask &= bool_series(df["cohort_usable"])
    for col in ["qc_missing_primary_feature", "qc_stage_or_metadata_artifact", "qc_signal_artifact"]:
        if col in df:
            mask &= ~bool_series(df[col])
    return mask


def summarize_dataset(table_dir: Path, spec: DatasetSpec) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    usecols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "channel",
        "analysis_primary_cohort",
        "cohort_usable",
        "analysis_qc_include",
        "qc_missing_primary_feature",
        "qc_stage_or_metadata_artifact",
        "qc_signal_artifact",
        "qc_specparam_error_gt_threshold",
        "qc_specparam_error_threshold",
        "specparam_error",
        "specparam_r_squared",
    ]
    path = table_dir / spec.matrix_path
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    denom = denominator_mask(df)
    fit_df = df[denom].copy()
    fit_df["specparam_error"] = pd.to_numeric(fit_df["specparam_error"], errors="coerce")
    fit_df["specparam_r_squared"] = pd.to_numeric(fit_df["specparam_r_squared"], errors="coerce")

    denom_rows = int(len(fit_df))
    unique_epochs = int(fit_df[["subject_id", "night_id", "epoch_idx"]].drop_duplicates().shape[0])
    missing_fit = int(fit_df["specparam_error"].isna().sum())
    current_threshold = (
        float(pd.to_numeric(fit_df["qc_specparam_error_threshold"], errors="coerce").dropna().iloc[0])
        if "qc_specparam_error_threshold" in fit_df and fit_df["qc_specparam_error_threshold"].notna().any()
        else 0.15
    )
    current_excluded = int((fit_df["specparam_error"] > current_threshold).sum() + missing_fit)

    summary_rows: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        excluded = int((fit_df["specparam_error"] > threshold).sum() + missing_fit)
        summary_rows.append(
            {
                "dataset": spec.dataset,
                "display": spec.display,
                "denominator_rows_before_specparam_qc": denom_rows,
                "recording_level_epochs_before_specparam_qc": unique_epochs,
                "current_threshold": threshold,
                "excluded_rows_at_current_threshold": excluded,
                "excluded_pct_at_current_threshold": 100.0 * excluded / denom_rows if denom_rows else np.nan,
                "included_rows_at_current_threshold": denom_rows - excluded,
                "primary_rncp_rows_reported": spec.primary_rncp_rows if np.isclose(threshold, 0.15) else np.nan,
                "row_difference_vs_reported": ((denom_rows - excluded - spec.primary_rncp_rows) if spec.primary_rncp_rows is not None and np.isclose(threshold, 0.15) else np.nan),
                "missing_specparam_fit_rows": missing_fit,
                "missing_specparam_fit_pct": 100.0 * missing_fit / denom_rows if denom_rows else np.nan,
            }
        )

    dist_rows: list[dict[str, object]] = []
    for metric in ["specparam_error", "specparam_r_squared"]:
        row = {
            "dataset": spec.dataset,
            "display": spec.display,
            "metric": metric,
        }
        row.update(summarize_distribution(fit_df[metric]))
        dist_rows.append(row)
    return summary_rows, dist_rows


def fmt_pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.2f}"


def write_markdown(table_dir: Path, exclusion: pd.DataFrame, dist: pd.DataFrame) -> None:
    current = exclusion[exclusion["current_threshold"].round(2) == 0.15].copy()
    current = current.drop_duplicates(["dataset", "current_threshold"], keep="first")
    lines = [
        "# Specparam Fit Quality and Threshold Exclusions",
        "",
        "## Primary 0.15 fit-error threshold",
        "",
        "| Dataset | Rows before specparam QC | Excluded rows | Excluded % | Included rows | Missing fits % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in current.to_dict("records"):
        lines.append(
            f"| {row['display']} | {int(row['denominator_rows_before_specparam_qc'])} | "
            f"{int(row['excluded_rows_at_current_threshold'])} | {fmt_pct(float(row['excluded_pct_at_current_threshold']))} | "
            f"{int(row['included_rows_at_current_threshold'])} | {fmt_pct(float(row['missing_specparam_fit_pct']))} |"
        )

    lines.extend(
        [
            "",
            "## Fit-quality distributions",
            "",
            "| Dataset | Metric | n | Mean | Median | IQR | 5th-95th percentile |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dist.to_dict("records"):
        lines.append(
            f"| {row['display']} | {row['metric']} | {int(row['n'])} | {float(row['mean']):.4f} | "
            f"{float(row['median']):.4f} | {float(row['p25']):.4f}-{float(row['p75']):.4f} | "
            f"{float(row['p05']):.4f}-{float(row['p95']):.4f} |"
        )

    thresh = exclusion[exclusion["current_threshold"].isin(THRESHOLDS)].copy()
    lines.extend(
        [
            "",
            "## Alternative fit-error thresholds",
            "",
            "| Dataset | Threshold | Excluded % | Included rows |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in thresh.to_dict("records"):
        lines.append(
            f"| {row['display']} | {float(row['current_threshold']):.2f} | "
            f"{fmt_pct(float(row['excluded_pct_at_current_threshold']))} | {int(row['included_rows_at_current_threshold'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The denominator is primary-cohort N3 channel rows after missing-feature, stage/metadata-artifact, and signal-artifact exclusions, but before applying the specparam fit-error threshold. The primary threshold is error <= 0.15.",
            "",
            "SC already has RNCP residual-structure sensitivity results for thresholds 0.10, 0.12, and 0.15; all remained significant relative to block-preserving nulls.",
        ]
    )
    (table_dir / "specparam_fit_quality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"

    all_exclusion = []
    all_dist = []
    for spec in SPECS:
        exclusion_rows, dist_rows = summarize_dataset(table_dir, spec)
        all_exclusion.extend(exclusion_rows)
        all_dist.extend(dist_rows)

    exclusion = pd.DataFrame(all_exclusion)
    dist = pd.DataFrame(all_dist)
    exclusion.to_csv(table_dir / "specparam_threshold_exclusions.csv", index=False)
    dist.to_csv(table_dir / "specparam_fit_quality_distributions.csv", index=False)
    write_markdown(table_dir, exclusion, dist)
    print(table_dir / "specparam_fit_quality_summary.md")
    print(exclusion.to_string(index=False))
    print(dist.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
