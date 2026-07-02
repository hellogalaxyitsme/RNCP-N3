#!/usr/bin/env python3
"""analysis-ready N3 matrix builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["subject_id", "night_id", "epoch_idx", "channel"]
PRIMARY_FEATURES = [
    "lzc",
    "permutation_entropy",
    "spectral_entropy",
    "aperiodic_exponent_specparam",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def read_tables(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(table_dir / "sleep_edf_sc_epoch_metadata.csv.gz", low_memory=False)
    signal = pd.read_csv(table_dir / "sleep_edf_sc_signal_features.csv.gz", low_memory=False)
    complexity = pd.read_csv(table_dir / "sleep_edf_sc_complexity_features.csv.gz", low_memory=False)
    cohort = pd.read_csv(table_dir / "sleep_edf_sc_cohort_table.csv")
    return metadata, signal, complexity, cohort


def build_matrix(table_dir: Path, specparam_error_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata, signal, complexity, cohort = read_tables(table_dir)

    metadata_n3 = metadata[metadata["stage"] == "N3"].copy()
    cohort = cohort.rename(columns={"usable": "cohort_usable"})
    cohort["cohort_usable"] = bool_series(cohort["cohort_usable"])

    meta_cols = [
        "subject_id",
        "night_id",
        "night",
        "epoch_idx",
        "epoch_start_sec",
        "epoch_start_min",
        "stage",
        "stage_original",
        "sleep_onset_min",
        "time_since_sleep_onset",
        "after_sleep_onset",
        "n3_bout_num",
        "position_within_bout",
        "position_within_bout_fraction",
        "n3_bout_duration_epochs",
        "n3_bout_duration_min",
        "stage_is_sleep",
        "stage_is_n3",
        "artifact_flag",
        "artifact_reason",
        "channel",
        "age",
        "sex",
    ]
    signal_cols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "channel",
        "delta_power",
        "total_power_0p5_45",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
        "cumulative_swa",
        "spectral_entropy",
        "aperiodic_exponent",
        "aperiodic_exponent_method",
        "artifact_signal_flag",
        "artifact_signal_reason",
    ]
    complexity_cols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "channel",
        "lzc",
        "permutation_entropy",
        "lzc_method",
        "permutation_entropy_method",
        "aperiodic_exponent_specparam",
        "specparam_r_squared",
        "specparam_error",
        "aperiodic_exponent_method",
    ]
    cohort_cols = [
        "subject_id",
        "night_id",
        "total_n3_min",
        "clean_n3_min",
        "cohort_usable",
        "total_sleep_min",
        "psg_duration_min",
    ]

    matrix = metadata_n3[meta_cols].merge(signal[signal_cols], on=KEYS, how="left", validate="one_to_one")
    matrix = matrix.merge(
        complexity[complexity_cols],
        on=KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_fallback", "_specparam"),
    )
    matrix = matrix.merge(cohort[cohort_cols], on=["subject_id", "night_id"], how="left", validate="many_to_one")

    matrix = matrix.rename(
        columns={
            "aperiodic_exponent": "aperiodic_exponent_fallback_loglog",
            "aperiodic_exponent_method_fallback": "aperiodic_exponent_fallback_method",
            "aperiodic_exponent_method_specparam": "aperiodic_exponent_specparam_method",
        }
    )

    matrix["artifact_flag"] = bool_series(matrix["artifact_flag"])
    matrix["artifact_signal_flag"] = bool_series(matrix["artifact_signal_flag"])
    matrix["after_sleep_onset"] = bool_series(matrix["after_sleep_onset"])
    matrix["stage_is_n3"] = bool_series(matrix["stage_is_n3"])
    matrix["cohort_usable"] = bool_series(matrix["cohort_usable"])

    missing_primary = matrix[PRIMARY_FEATURES].isna().any(axis=1)
    specparam_error_flag = matrix["specparam_error"].gt(specparam_error_threshold).fillna(True)
    matrix["qc_missing_primary_feature"] = missing_primary
    matrix["qc_stage_or_metadata_artifact"] = matrix["artifact_flag"]
    matrix["qc_signal_artifact"] = matrix["artifact_signal_flag"]
    matrix["qc_specparam_error_gt_threshold"] = specparam_error_flag
    matrix["qc_specparam_error_threshold"] = specparam_error_threshold
    matrix["analysis_qc_include"] = (
        matrix["cohort_usable"]
        & matrix["stage_is_n3"]
        & ~matrix["qc_missing_primary_feature"]
        & ~matrix["qc_stage_or_metadata_artifact"]
        & ~matrix["qc_signal_artifact"]
        & ~matrix["qc_specparam_error_gt_threshold"]
    )
    matrix["analysis_primary_cohort"] = matrix["cohort_usable"]

    primary = matrix[matrix["analysis_primary_cohort"]].copy()

    summary = {
        "all_n3_rows": int(len(matrix)),
        "all_n3_recording_level_epochs": int(matrix[["night_id", "epoch_idx"]].drop_duplicates().shape[0]),
        "all_n3_subjects": int(matrix["subject_id"].nunique()),
        "all_n3_recordings": int(matrix["night_id"].nunique()),
        "primary_rows": int(len(primary)),
        "primary_recording_level_epochs": int(primary[["night_id", "epoch_idx"]].drop_duplicates().shape[0]),
        "primary_subjects": int(primary["subject_id"].nunique()),
        "primary_recordings": int(primary["night_id"].nunique()),
        "qc_include_rows": int(matrix["analysis_qc_include"].sum()),
        "primary_qc_include_rows": int(primary["analysis_qc_include"].sum()),
        "missing_primary_feature_rows": int(matrix["qc_missing_primary_feature"].sum()),
        "metadata_artifact_rows": int(matrix["qc_stage_or_metadata_artifact"].sum()),
        "signal_artifact_rows": int(matrix["qc_signal_artifact"].sum()),
        "specparam_error_gt_threshold_rows": int(matrix["qc_specparam_error_gt_threshold"].sum()),
        "specparam_error_threshold": float(specparam_error_threshold),
    }
    return matrix, primary, summary


def write_outputs(table_dir: Path, matrix: pd.DataFrame, primary: pd.DataFrame, summary: dict) -> None:
    matrix_path = table_dir / "sleep_edf_sc_n3_analysis_matrix_all.csv.gz"
    primary_path = table_dir / "sleep_edf_sc_n3_analysis_matrix_primary.csv.gz"
    summary_csv_path = table_dir / "sleep_edf_sc_n3_analysis_matrix_summary.csv"
    summary_md_path = table_dir / "analysis_matrix_summary.md"

    matrix.to_csv(matrix_path, index=False, compression="gzip")
    primary.to_csv(primary_path, index=False, compression="gzip")
    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

    feature_desc = matrix[PRIMARY_FEATURES + ["relative_delta_power", "slow_wave_density", "cumulative_swa"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).T
    channel_counts = matrix.groupby(["channel", "analysis_primary_cohort", "analysis_qc_include"]).size().to_frame("rows")

    lines = [
        "# Analysis-Ready N3 Matrix Summary",
        "",
        "## Outputs",
        "",
        "- `sleep_edf_sc_n3_analysis_matrix_all.csv.gz`",
        "- `sleep_edf_sc_n3_analysis_matrix_primary.csv.gz`",
        "- `sleep_edf_sc_n3_analysis_matrix_summary.csv`",
        "",
        "## Coverage",
        "",
        f"- All N3 channel rows: {summary['all_n3_rows']}",
        f"- All N3 recording-level epochs: {summary['all_n3_recording_level_epochs']}",
        f"- All N3 subjects: {summary['all_n3_subjects']}",
        f"- All N3 recordings: {summary['all_n3_recordings']}",
        f"- Primary usable-cohort channel rows: {summary['primary_rows']}",
        f"- Primary usable-cohort recording-level epochs: {summary['primary_recording_level_epochs']}",
        f"- Primary usable-cohort subjects: {summary['primary_subjects']}",
        f"- Primary usable-cohort recordings: {summary['primary_recordings']}",
        "",
        "## QC Flags",
        "",
        f"- Missing primary feature rows: {summary['missing_primary_feature_rows']}",
        f"- Metadata/stage artifact rows: {summary['metadata_artifact_rows']}",
        f"- Signal artifact rows: {summary['signal_artifact_rows']}",
        f"- Specparam error threshold: {summary['specparam_error_threshold']}",
        f"- Rows above specparam error threshold: {summary['specparam_error_gt_threshold_rows']}",
        f"- All-N3 rows passing QC include flag: {summary['qc_include_rows']}",
        f"- Primary rows passing QC include flag: {summary['primary_qc_include_rows']}",
        "",
        "Rows are not aggressively dropped. `analysis_qc_include` is the conservative include flag for first-pass models.",
        "",
        "## Primary Feature Distributions",
        "",
        feature_desc.to_markdown(floatfmt=".4f"),
        "",
        "## Channel/QC Counts",
        "",
        channel_counts.to_markdown(),
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--specparam-error-threshold", type=float, default=0.15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    matrix, primary, summary = build_matrix(table_dir, args.specparam_error_threshold)
    write_outputs(table_dir, matrix, primary, summary)
    print(f"Wrote all-N3 matrix rows={len(matrix)}")
    print(f"Wrote primary analysis matrix rows={len(primary)}")
    print(f"Primary QC include rows={summary['primary_qc_include_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
