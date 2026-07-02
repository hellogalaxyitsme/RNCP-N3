#!/usr/bin/env python3
"""Sleep-EDF Sleep Telemetry replication pipeline."""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import signal

import sleep_edf_audit as audit
import sleep_edf_epoch_metadata as epoch_meta
import sleep_edf_complexity_features as complexity_features
import sleep_edf_rncp_reproducibility as rncp_reproducibility
import sleep_edf_signal_features as signal_features
import sleep_edf_stage_durations as stage_durations


FEATURES = ["lzc", "permutation_entropy", "spectral_entropy", "aperiodic_exponent_specparam"]
RESIDUAL_COLS = [f"{feature}_rncp_residual_z" for feature in FEATURES]
KEYS = ["subject_id", "night_id", "epoch_idx", "channel"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean()
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - mean) / sd


def normalize_st_subject_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.startswith("ST") and len(text) >= 5:
        return text[:5]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) >= 3 and digits[-3:].startswith("7"):
        return f"ST{digits[-3:]}"
    return f"ST7{int(digits):02d}"


def load_st_subjects(dataset_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls_path = dataset_root / "ST-subjects.xls"
    if not xls_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_excel(xls_path).dropna(how="all").copy()
    raw.columns = [str(col).strip() for col in raw.columns]
    lower = {col.lower(): col for col in raw.columns}
    subject_col = next((col for key, col in lower.items() if "subject" in key or key in {"id", "subj"}), raw.columns[0])
    age_col = next((col for key, col in lower.items() if "age" in key), None)
    sex_col = next((col for key, col in lower.items() if "sex" in key or "gender" in key), None)

    standardized = pd.DataFrame()
    standardized["subject_metadata_value"] = raw[subject_col]
    standardized["subject_id"] = standardized["subject_metadata_value"].map(normalize_st_subject_id)
    standardized["age"] = pd.to_numeric(raw[age_col], errors="coerce") if age_col else pd.NA
    standardized["sex"] = raw[sex_col].astype(str).str.strip() if sex_col else pd.NA

    # Keep any extra columns as metadata hints; ST includes treatment/session fields in some releases.
    for col in raw.columns:
        if col not in {subject_col, age_col, sex_col}:
            clean = re.sub(r"[^A-Za-z0-9]+", "_", col).strip("_").lower()
            if clean:
                standardized[f"st_meta_{clean}"] = raw[col]
    standardized = standardized.dropna(subset=["subject_id"]).drop_duplicates("subject_id")
    return raw, standardized


def build_st_inventory(dataset_root: Path, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = audit.Config(
        dataset_root=dataset_root,
        project_data_root=table_dir.parent,
        cache_root=table_dir.parent / "cache",
        primary_subset="sleep-telemetry",
    )
    file_inventory = audit.build_file_inventory(cfg)
    recordings = audit.build_recording_inventory(file_inventory, "sleep-telemetry")
    recordings = recordings[recordings["cohort"] == "ST"].copy()
    recordings["psg_suffix"] = recordings["psg_suffix"].astype(str)
    recordings["hypnogram_suffix"] = recordings["hypnogram_suffix"].astype(str)
    recordings["st_psg_suffix"] = recordings["psg_suffix"]
    recordings["st_hypnogram_suffix"] = recordings["hypnogram_suffix"]
    recordings["st_treatment_code"] = recordings["hypnogram_suffix"].str[-1].where(
        recordings["hypnogram_suffix"].notna(), recordings["psg_suffix"].str[-1]
    )

    raw_subjects, subjects = load_st_subjects(dataset_root)
    subjects = audit.build_subject_inventory(recordings, subjects)
    if "sex" not in subjects:
        subjects["sex"] = pd.NA
    if "age" not in subjects:
        subjects["age"] = pd.NA

    file_inventory.to_csv(table_dir / "sleep_edf_file_inventory.csv", index=False)
    recordings.to_csv(table_dir / "sleep_edf_st_recording_inventory.csv", index=False)
    subjects.to_csv(table_dir / "sleep_edf_st_subject_inventory.csv", index=False)
    raw_subjects.to_csv(table_dir / "sleep_edf_st_subjects_raw_from_xls.csv", index=False)
    return file_inventory, recordings, subjects


def run_stage_durations(dataset_root: Path, table_dir: Path, recordings: pd.DataFrame, subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [stage_durations.summarize_recording(row, dataset_root) for _, row in recordings.iterrows()]
    durations = pd.DataFrame(rows)
    meta_cols = [col for col in ["subject_id", "age", "sex"] if col in subjects.columns]
    durations = durations.merge(subjects[meta_cols], on="subject_id", how="left")
    cohort = durations[
        [
            "subject_id",
            "night",
            "recording_key",
            "age",
            "sex",
            "n3_min",
            "usable_n3_30min",
            "total_sleep_min",
            "psg_duration_min",
            "status",
            "st_psg_suffix",
            "st_hypnogram_suffix",
            "st_treatment_code",
        ]
    ].copy()
    cohort = cohort.rename(
        columns={"recording_key": "night_id", "n3_min": "total_n3_min", "usable_n3_30min": "usable"}
    )
    cohort["clean_n3_min"] = pd.NA
    cohort = cohort[
        [
            "subject_id",
            "night_id",
            "age",
            "sex",
            "total_n3_min",
            "clean_n3_min",
            "usable",
            "total_sleep_min",
            "psg_duration_min",
            "status",
            "st_psg_suffix",
            "st_hypnogram_suffix",
            "st_treatment_code",
        ]
    ].sort_values(["subject_id", "night_id"])
    durations.to_csv(table_dir / "sleep_edf_st_stage_duration_inventory.csv", index=False)
    cohort.to_csv(table_dir / "sleep_edf_st_cohort_table.csv", index=False)
    return durations, cohort


def run_epoch_metadata(dataset_root: Path, table_dir: Path, recordings: pd.DataFrame, subjects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = [col for col in ["subject_id", "age", "sex"] if col in subjects.columns]
    recordings = recordings.merge(subjects[meta_cols], on="subject_id", how="left")
    all_epochs = []
    summaries = []
    for _, recording in recordings.iterrows():
        try:
            epochs, summary = epoch_meta.build_recording_epochs(recording, dataset_root)
            epochs["age"] = recording.get("age", pd.NA)
            epochs["sex"] = recording.get("sex", pd.NA)
            epochs["st_psg_suffix"] = recording.get("st_psg_suffix", "")
            epochs["st_hypnogram_suffix"] = recording.get("st_hypnogram_suffix", "")
            epochs["st_treatment_code"] = recording.get("st_treatment_code", "")
            all_epochs.append(epochs)
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    "subject_id": recording.get("subject_id"),
                    "night_id": recording.get("recording_key"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    epoch_df = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    epoch_df.to_csv(table_dir / "sleep_edf_st_epoch_metadata.csv.gz", index=False, compression="gzip")
    summary_df.to_csv(table_dir / "sleep_edf_st_epoch_recording_summary.csv", index=False)
    return epoch_df, summary_df


def run_signal_features(dataset_root: Path, table_dir: Path, recordings: pd.DataFrame, metadata: pd.DataFrame, overwrite: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = table_dir.parent / "interim" / "st_features_by_recording"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, recording in recordings.iterrows():
        key = str(recording["recording_key"])
        try:
            rec_key, status, elapsed, n_rows = signal_features.process_recording(
                recording=recording,
                dataset_root=dataset_root,
                metadata=metadata,
                out_dir=out_dir,
                include_complexity=False,
                overwrite=overwrite,
            )
            print(f"ST signal {rec_key}: {status} rows={n_rows} elapsed={elapsed:.1f}s", flush=True)
            rows.append({"recording_key": rec_key, "status": status, "rows": n_rows, "elapsed_sec": elapsed, "error": ""})
        except Exception as exc:
            print(f"ST signal {key}: error {type(exc).__name__}: {exc}", flush=True)
            rows.append({"recording_key": key, "status": "error", "rows": pd.NA, "elapsed_sec": pd.NA, "error": f"{type(exc).__name__}: {exc}"})
    files = sorted(out_dir.glob("*_signal_features.csv.gz"))
    combined = pd.concat([pd.read_csv(path) for path in files], ignore_index=True) if files else pd.DataFrame()
    combined.to_csv(table_dir / "sleep_edf_st_signal_features.csv.gz", index=False, compression="gzip")
    run_df = pd.DataFrame(rows)
    run_df.to_csv(table_dir / "sleep_edf_st_signal_feature_run_summary.csv", index=False)
    return combined, run_df


def run_complexity_features(dataset_root: Path, table_dir: Path, recordings: pd.DataFrame, metadata: pd.DataFrame, overwrite: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = table_dir.parent / "interim" / "st_complexity_by_recording"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, recording in recordings.iterrows():
        key = str(recording["recording_key"])
        try:
            rec_key, status, elapsed, n_rows = complexity_features.process_recording(
                recording=recording,
                dataset_root=dataset_root,
                metadata=metadata,
                out_dir=out_dir,
                stages={"N3"},
                overwrite=overwrite,
                perm_order=5,
                perm_delay=1,
                skip_specparam=False,
                specparam_low=1.0,
                specparam_high=40.0,
                specparam_max_peaks=6,
            )
            print(f"ST complexity {rec_key}: {status} rows={n_rows} elapsed={elapsed:.1f}s", flush=True)
            rows.append({"recording_key": rec_key, "status": status, "rows": n_rows, "elapsed_sec": elapsed, "error": ""})
        except Exception as exc:
            print(f"ST complexity {key}: error {type(exc).__name__}: {exc}", flush=True)
            rows.append({"recording_key": key, "status": "error", "rows": pd.NA, "elapsed_sec": pd.NA, "error": f"{type(exc).__name__}: {exc}"})
    frames = []
    for path in sorted(out_dir.glob("*_complexity_features.csv.gz")):
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(table_dir / "sleep_edf_st_complexity_features.csv.gz", index=False, compression="gzip")
    run_df = pd.DataFrame(rows)
    run_df.to_csv(table_dir / "sleep_edf_st_complexity_feature_run_summary.csv", index=False)
    return combined, run_df


def build_analysis_matrix(table_dir: Path, specparam_error_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = pd.read_csv(table_dir / "sleep_edf_st_epoch_metadata.csv.gz", low_memory=False)
    signal_df = pd.read_csv(table_dir / "sleep_edf_st_signal_features.csv.gz", low_memory=False)
    complexity = pd.read_csv(table_dir / "sleep_edf_st_complexity_features.csv.gz", low_memory=False)
    cohort = pd.read_csv(table_dir / "sleep_edf_st_cohort_table.csv")
    metadata_n3 = metadata[metadata["stage"] == "N3"].copy()
    cohort = cohort.rename(columns={"usable": "cohort_usable"})
    cohort["cohort_usable"] = bool_series(cohort["cohort_usable"])

    meta_cols = [
        "subject_id", "night_id", "night", "epoch_idx", "epoch_start_sec", "epoch_start_min", "stage",
        "stage_original", "sleep_onset_min", "time_since_sleep_onset", "after_sleep_onset", "n3_bout_num",
        "position_within_bout", "position_within_bout_fraction", "n3_bout_duration_epochs",
        "n3_bout_duration_min", "stage_is_sleep", "stage_is_n3", "artifact_flag", "artifact_reason",
        "channel", "age", "sex", "st_psg_suffix", "st_hypnogram_suffix", "st_treatment_code",
    ]
    signal_cols = KEYS + [
        "delta_power", "total_power_0p5_45", "relative_delta_power", "slow_wave_density",
        "slow_wave_occupancy", "cumulative_swa", "spectral_entropy", "aperiodic_exponent",
        "aperiodic_exponent_method", "artifact_signal_flag", "artifact_signal_reason",
    ]
    complexity_cols = KEYS + [
        "lzc", "permutation_entropy", "lzc_method", "permutation_entropy_method",
        "aperiodic_exponent_specparam", "specparam_r_squared", "specparam_error", "aperiodic_exponent_method",
    ]
    cohort_cols = [
        "subject_id", "night_id", "total_n3_min", "clean_n3_min", "cohort_usable",
        "total_sleep_min", "psg_duration_min", "st_psg_suffix", "st_hypnogram_suffix", "st_treatment_code",
    ]

    matrix = metadata_n3[meta_cols].merge(signal_df[signal_cols], on=KEYS, how="left", validate="one_to_one")
    matrix = matrix.merge(complexity[complexity_cols], on=KEYS, how="left", validate="one_to_one", suffixes=("_fallback", "_specparam"))
    matrix = matrix.merge(cohort[cohort_cols], on=["subject_id", "night_id"], how="left", validate="many_to_one", suffixes=("", "_cohort"))
    matrix = matrix.rename(
        columns={
            "aperiodic_exponent": "aperiodic_exponent_fallback_loglog",
            "aperiodic_exponent_method_fallback": "aperiodic_exponent_fallback_method",
            "aperiodic_exponent_method_specparam": "aperiodic_exponent_specparam_method",
        }
    )
    for col in ["artifact_flag", "artifact_signal_flag", "after_sleep_onset", "stage_is_n3", "cohort_usable"]:
        matrix[col] = bool_series(matrix[col])
    missing_primary = matrix[FEATURES].isna().any(axis=1)
    matrix["qc_missing_primary_feature"] = missing_primary
    matrix["qc_stage_or_metadata_artifact"] = matrix["artifact_flag"]
    matrix["qc_signal_artifact"] = matrix["artifact_signal_flag"]
    matrix["qc_specparam_error_gt_threshold"] = matrix["specparam_error"].gt(specparam_error_threshold).fillna(True)
    matrix["qc_specparam_error_threshold"] = specparam_error_threshold
    matrix["analysis_qc_include"] = (
        matrix["cohort_usable"] & matrix["stage_is_n3"] & ~matrix["qc_missing_primary_feature"]
        & ~matrix["qc_stage_or_metadata_artifact"] & ~matrix["qc_signal_artifact"] & ~matrix["qc_specparam_error_gt_threshold"]
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
        "specparam_error_gt_threshold_rows": int(matrix["qc_specparam_error_gt_threshold"].sum()),
    }
    matrix.to_csv(table_dir / "sleep_edf_st_n3_analysis_matrix_all.csv.gz", index=False, compression="gzip")
    primary.to_csv(table_dir / "sleep_edf_st_n3_analysis_matrix_primary.csv.gz", index=False, compression="gzip")
    pd.DataFrame([summary]).to_csv(table_dir / "sleep_edf_st_n3_analysis_matrix_summary.csv", index=False)
    return matrix, primary, summary


def prepare_model_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    df = matrix[bool_series(matrix["analysis_qc_include"])].copy()
    df["sex"] = df["sex"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    df["st_treatment_code"] = df["st_treatment_code"].astype(str)
    df["log_cumulative_swa"] = np.log1p(pd.to_numeric(df["cumulative_swa"], errors="coerce").clip(lower=0))
    continuous = [
        "age", "time_since_sleep_onset", "position_within_bout_fraction", "relative_delta_power",
        "slow_wave_density", "slow_wave_occupancy", "log_cumulative_swa",
    ]
    for col in continuous:
        df[f"{col}_z"] = zscore(df[col])
    df["time_since_sleep_onset_z2"] = df["time_since_sleep_onset_z"] ** 2
    df["time_since_sleep_onset_z3"] = df["time_since_sleep_onset_z"] ** 3
    for feature in FEATURES:
        df[f"{feature}_z"] = zscore(df[feature])
    required = [f"{feature}_z" for feature in FEATURES] + [
        "age_z", "time_since_sleep_onset_z", "position_within_bout_fraction_z",
        "relative_delta_power_z", "slow_wave_density_z", "slow_wave_occupancy_z",
        "log_cumulative_swa_z", "subject_id", "night_id", "channel",
    ]
    return df.dropna(subset=required).copy()


def fixed_formula(feature: str, include_sex: bool, include_treatment: bool, include_channel: bool = True) -> str:
    sex = " + C(sex)" if include_sex else ""
    treatment = " + C(st_treatment_code)" if include_treatment else ""
    channel = " + C(channel)" if include_channel else ""
    return (
        f"{feature}_z ~ age_z{sex}{channel}{treatment} + "
        "time_since_sleep_onset_z + time_since_sleep_onset_z2 + time_since_sleep_onset_z3 + "
        "position_within_bout_fraction_z + relative_delta_power_z + slow_wave_density_z + "
        "slow_wave_occupancy_z + log_cumulative_swa_z"
    )


def fit_st_model(df: pd.DataFrame, feature: str):
    include_sex_default = df["sex"].nunique(dropna=True) > 1
    include_treatment_default = df["st_treatment_code"].nunique(dropna=True) > 1
    include_channel_default = df["channel"].nunique(dropna=True) > 1
    attempts = [
        (include_sex_default, include_treatment_default, include_channel_default, "full_available"),
        (include_sex_default, False, include_channel_default, "drop_treatment"),
        (False, include_treatment_default, include_channel_default, "drop_sex"),
        (False, False, include_channel_default, "drop_sex_treatment"),
        (False, False, False, "drop_sex_treatment_channel"),
    ]
    last_error = None
    last_result = None
    for include_sex, include_treatment, include_channel, label in attempts:
        formula = fixed_formula(feature, include_sex, include_treatment, include_channel)
        for method in ["lbfgs", "powell", "cg"]:
            try:
                model = smf.mixedlm(
                    formula,
                    data=df,
                    groups=df["subject_id"],
                    re_formula="1",
                    vc_formula={"night": "0 + C(night_id)"},
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = model.fit(method=method, reml=True, maxiter=1000, disp=False)
                attempt_label = f"{label}_{method}"
                if bool(getattr(result, "converged", False)):
                    return formula, attempt_label, result, [str(item.message) for item in caught]
                last_result = (formula, attempt_label, result, [str(item.message) for item in caught])
            except Exception as exc:
                last_error = exc
    if last_result is not None:
        # Keep moving if no specification reaches the optimizer convergence flag;
        # this is a secondary replication cohort and the summary records the flag.
        return last_result
    raise RuntimeError(f"All ST mixed-model attempts failed for {feature}: {type(last_error).__name__}: {last_error}")


def run_models(table_dir: Path, matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = prepare_model_frame(matrix)
    variance_rows = []
    fixed_rows = []
    residuals = df[
        ["subject_id", "night_id", "epoch_idx", "channel", "age", "sex", "st_treatment_code",
         "time_since_sleep_onset", "position_within_bout_fraction", "relative_delta_power",
         "slow_wave_density", "slow_wave_occupancy", "cumulative_swa"]
    ].copy()
    for feature in FEATURES:
        formula, model_attempt, result, caught_messages = fit_st_model(df, feature)
        fixed_fitted = np.asarray(result.model.exog @ result.fe_params.to_numpy())
        var_fixed = float(np.nanvar(fixed_fitted, ddof=0))
        var_subject = float(np.asarray(result.cov_re)[0, 0]) if result.cov_re is not None and result.cov_re.size else 0.0
        var_night = float(np.nansum(np.asarray(getattr(result, "vcomp", []), dtype=float)))
        var_resid = float(result.scale)
        total = var_fixed + var_subject + var_night + var_resid
        variance_rows.append(
            {
                "feature": feature,
                "formula": formula,
                "model_attempt": model_attempt,
                "n_rows": int(result.nobs),
                "n_subjects": int(df["subject_id"].nunique()),
                "n_nights": int(df["night_id"].nunique()),
                "converged": bool(getattr(result, "converged", False)),
                "var_fixed": var_fixed,
                "var_subject_intercept": var_subject,
                "var_night_component": var_night,
                "var_residual": var_resid,
                "marginal_r2_approx": var_fixed / total if total > 0 else np.nan,
                "conditional_r2_approx": (var_fixed + var_subject + var_night) / total if total > 0 else np.nan,
                "residual_sd": float(np.nanstd(result.resid, ddof=0)),
                "warnings": " | ".join(dict.fromkeys(caught_messages)),
            }
        )
        params = result.params
        conf = result.conf_int()
        fixed_rows.append(
            pd.DataFrame(
                {
                    "feature": feature,
                    "term": params.index,
                    "estimate": params.values,
                    "std_error": result.bse.reindex(params.index).values,
                    "p_value": result.pvalues.reindex(params.index).values,
                    "ci_low": conf.reindex(params.index)[0].values,
                    "ci_high": conf.reindex(params.index)[1].values,
                }
            )
        )
        residuals[f"{feature}_z"] = df[f"{feature}_z"].to_numpy()
        residuals[f"{feature}_fitted_z"] = np.asarray(result.fittedvalues)
        residuals[f"{feature}_rncp_residual_z"] = np.asarray(result.resid)
        print(f"ST model {feature}: converged={getattr(result, 'converged', False)} attempt={model_attempt}", flush=True)
    variance = pd.DataFrame(variance_rows)
    fixed = pd.concat(fixed_rows, ignore_index=True)
    residuals["rncp_l2_norm"] = np.sqrt(np.square(residuals[RESIDUAL_COLS]).sum(axis=1))
    variance.to_csv(table_dir / "sleep_edf_st_residual_model_variance_summary.csv", index=False)
    fixed.to_csv(table_dir / "sleep_edf_st_residual_model_fixed_effects.csv", index=False)
    residuals.to_csv(table_dir / "sleep_edf_st_n3_rncp_residuals.csv.gz", index=False, compression="gzip")
    return variance, fixed, residuals


def run_reproducibility(table_dir: Path, residuals: pd.DataFrame, n_null: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = residuals.dropna(subset=["subject_id", "night_id", "channel"] + RESIDUAL_COLS).reset_index(drop=True)
    obs_corr = rncp_reproducibility.residual_corr(df)
    null_df, global_summary = rncp_reproducibility.run_global_null(df, n_null, seed)
    n_folds = min(5, max(2, df["subject_id"].nunique()))
    fold_df = rncp_reproducibility.run_fold_reproducibility(df, n_folds, n_null, seed + 1)
    obs_corr.to_csv(table_dir / "sleep_edf_st_rncp_reproducibility_observed_correlation.csv")
    null_df.to_csv(table_dir / "sleep_edf_st_rncp_reproducibility_global_null_iterations.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(table_dir / "sleep_edf_st_rncp_reproducibility_global_null_summary.csv", index=False)
    fold_df.to_csv(table_dir / "sleep_edf_st_rncp_reproducibility_fold_reproducibility.csv", index=False)
    return obs_corr, fold_df, global_summary


def write_summary(table_dir: Path, stage: pd.DataFrame, matrix_summary: dict, variance: pd.DataFrame, obs_corr: pd.DataFrame, fold_df: pd.DataFrame, global_summary: dict) -> None:
    usable = stage[(stage["status"] == "ok") & (stage["usable_n3_30min"])].copy()
    lines = [
        "# Sleep-EDF Telemetry Replication Summary",
        "",
        "## Cohort",
        "",
        f"- ST recordings loaded: {int((stage['status'] == 'ok').sum())} / {len(stage)}",
        f"- ST subjects total: {stage['subject_id'].nunique()}",
        f"- Recordings with >=30 min N3: {len(usable)}",
        f"- Subjects with >=1 usable N3 recording: {usable['subject_id'].nunique()}",
        "",
        "## Analysis Matrix",
        "",
        f"- All N3 rows: {matrix_summary['all_n3_rows']}",
        f"- Primary rows: {matrix_summary['primary_rows']}",
        f"- Primary QC include rows: {matrix_summary['primary_qc_include_rows']}",
        f"- Primary subjects: {matrix_summary['primary_subjects']}",
        f"- Primary recordings: {matrix_summary['primary_recordings']}",
        "",
        "## Mixed Models",
        "",
        variance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## RNCP Residual Correlation",
        "",
        obs_corr.to_markdown(floatfmt=".4f"),
        "",
        "## Null/Reproducibility",
        "",
        f"- Observed mean abs off-diagonal residual correlation: {global_summary['observed_mean_abs_offdiag_corr']:.4f}",
        f"- Null mean: {global_summary['null_mean_abs_offdiag_corr_mean']:.4f}",
        f"- Empirical global p: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}",
        f"- Mean fold train-heldout similarity: {fold_df['observed_train_heldout_vector_similarity'].mean():.4f}",
        "",
        "## Outputs",
        "",
        "- `sleep_edf_st_*` tables in this directory",
    ]
    (table_dir / "sleep_telemetry_replication_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("ST inventory", flush=True)
    _, recordings, subjects = build_st_inventory(dataset_root, table_dir)
    recordings = recordings[recordings["has_complete_pair"]].copy()
    print(f"ST complete pairs: {len(recordings)}", flush=True)

    print("ST stage durations", flush=True)
    stage, cohort = run_stage_durations(dataset_root, table_dir, recordings, subjects)
    print("ST epoch metadata", flush=True)
    metadata, epoch_summary = run_epoch_metadata(dataset_root, table_dir, recordings, subjects)
    print("ST signal features", flush=True)
    signal_df, signal_run = run_signal_features(dataset_root, table_dir, recordings, metadata, args.overwrite)
    print("ST complexity features", flush=True)
    complexity, complexity_run = run_complexity_features(dataset_root, table_dir, recordings, metadata, args.overwrite)
    print("ST analysis matrix", flush=True)
    matrix, primary, matrix_summary = build_analysis_matrix(table_dir, specparam_error_threshold=0.15)
    print("ST models", flush=True)
    variance, fixed, residuals = run_models(table_dir, primary)
    print("ST reproducibility", flush=True)
    obs_corr, fold_df, global_summary = run_reproducibility(table_dir, residuals, args.n_null, args.seed)
    write_summary(table_dir, stage, matrix_summary, variance, obs_corr, fold_df, global_summary)

    print(f"ST analysis complete in {time.time() - started:.1f}s")
    print(f"Primary QC rows: {matrix_summary['primary_qc_include_rows']}")
    print(f"Global p: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
