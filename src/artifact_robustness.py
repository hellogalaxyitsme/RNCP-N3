#!/usr/bin/env python3
"""high-frequency, EMG/EOG, and transition artifact robustness checks."""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy.io
import statsmodels.formula.api as smf
from scipy import signal

from anphy_sleep_common import discover_recordings
from anphy_sleep_n3_features import extract_edf_to_cache, norm_channel
from sleep_edf_rncp_reproducibility import (
    FEATURES,
    RESIDUAL_COLS,
    observed_pair_table,
    run_fold_reproducibility,
    run_global_null,
)


EPOCH_SEC = 30.0
EPS = np.finfo(float).eps
TRANSITION_WINDOW_EPOCHS = 4


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    matrix_path: str
    metadata_path: str
    recording_inventory_path: str | None
    artifact_covariates_path: str
    augmented_matrix_path: str
    residuals_path: str
    summary_prefix: str


SPECS = {
    "sleep_edf_sc": DatasetSpec(
        dataset="sleep_edf_sc",
        matrix_path="sleep_edf_sc_n3_analysis_matrix_primary.csv.gz",
        metadata_path="sleep_edf_sc_epoch_metadata.csv.gz",
        recording_inventory_path="sleep_edf_sc_recording_inventory.csv",
        artifact_covariates_path="artifact_sleep_edf_sc_artifact_covariates.csv.gz",
        augmented_matrix_path="artifact_sleep_edf_sc_n3_matrix_artifact_augmented.csv.gz",
        residuals_path="artifact_sleep_edf_sc_artifact_adjusted_rncp_residuals.csv.gz",
        summary_prefix="artifact_sleep_edf_sc",
    ),
    "sleep_edf_st": DatasetSpec(
        dataset="sleep_edf_st",
        matrix_path="sleep_edf_st_n3_analysis_matrix_primary.csv.gz",
        metadata_path="sleep_edf_st_epoch_metadata.csv.gz",
        recording_inventory_path="sleep_edf_st_recording_inventory.csv",
        artifact_covariates_path="artifact_sleep_edf_st_artifact_covariates.csv.gz",
        augmented_matrix_path="artifact_sleep_edf_st_n3_matrix_artifact_augmented.csv.gz",
        residuals_path="artifact_sleep_edf_st_artifact_adjusted_rncp_residuals.csv.gz",
        summary_prefix="artifact_sleep_edf_st",
    ),
    "anphy": DatasetSpec(
        dataset="anphy",
        matrix_path="anphy_sleep_n3_analysis_matrix_primary.csv.gz",
        metadata_path="anphy_sleep_epoch_metadata.csv.gz",
        recording_inventory_path=None,
        artifact_covariates_path="artifact_anphy_artifact_covariates.csv.gz",
        augmented_matrix_path="artifact_anphy_n3_matrix_artifact_augmented.csv.gz",
        residuals_path="artifact_anphy_artifact_adjusted_rncp_residuals.csv.gz",
        summary_prefix="artifact_anphy",
    ),
}


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


def bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    high = min(high, float(freqs.max()))
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def epoch_bandpowers(epochs_uv: np.ndarray, sfreq: float) -> pd.DataFrame:
    nperseg = min(int(round(4.0 * sfreq)), epochs_uv.shape[1])
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, psd = signal.welch(
        epochs_uv,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        axis=1,
        scaling="density",
    )
    total = bandpower(freqs, psd, 0.5, min(45.0, sfreq / 2.0 - 0.1))
    beta = bandpower(freqs, psd, 20.0, 30.0)
    hf_30_45 = bandpower(freqs, psd, 30.0, min(45.0, sfreq / 2.0 - 0.1))
    hf_35_45 = bandpower(freqs, psd, 35.0, min(45.0, sfreq / 2.0 - 0.1))
    return pd.DataFrame(
        {
            "eeg_beta_20_30_power": beta,
            "eeg_hf_30_45_power": hf_30_45,
            "eeg_hf_35_45_power": hf_35_45,
            "eeg_hf_30_45_ratio": np.divide(hf_30_45, total + EPS),
            "eeg_hf_35_45_ratio": np.divide(hf_35_45, total + EPS),
        }
    )


def channel_epochs(data_uv: np.ndarray, epoch_indices: np.ndarray, samples_per_epoch: int) -> tuple[np.ndarray, np.ndarray]:
    epochs = []
    kept = []
    n = data_uv.size
    for idx in epoch_indices:
        start = int(idx) * samples_per_epoch
        stop = start + samples_per_epoch
        if 0 <= start < stop <= n:
            epochs.append(data_uv[start:stop])
            kept.append(int(idx))
    if not epochs:
        return np.empty((0, samples_per_epoch)), np.asarray([], dtype=int)
    return np.vstack(epochs), np.asarray(kept, dtype=int)


def detect_eog_channels(channel_names: list[str]) -> list[str]:
    tokens = ["eog", "loc", "roc", "eye", "heog", "veog"]
    return [ch for ch in channel_names if any(token in ch.lower() for token in tokens)]


def detect_emg_channels(channel_names: list[str]) -> list[str]:
    tokens = ["emg", "chin", "submental", "mentalis"]
    return [ch for ch in channel_names if any(token in ch.lower() for token in tokens)]


def rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nanmean(np.square(values), axis=1))


def compute_aux_epoch_covariates(raw: mne.io.BaseRaw, epoch_indices: np.ndarray, samples_per_epoch: int) -> pd.DataFrame:
    rows = pd.DataFrame({"epoch_idx": epoch_indices.astype(int)})
    sfreq = float(raw.info["sfreq"])

    for kind, channels in [("eog", detect_eog_channels(raw.ch_names)), ("emg", detect_emg_channels(raw.ch_names))]:
        rms_values = []
        hf_values = []
        for ch in channels:
            data_uv = raw.get_data(picks=[ch], verbose="ERROR")[0] * 1_000_000.0
            epochs_uv, kept = channel_epochs(data_uv, epoch_indices, samples_per_epoch)
            if epochs_uv.size == 0:
                continue
            band = epoch_bandpowers(epochs_uv, sfreq)
            tmp = pd.DataFrame(
                {
                    "epoch_idx": kept,
                    f"{kind}_rms": rms(epochs_uv),
                    f"{kind}_hf_20_45_power": band["eeg_beta_20_30_power"].to_numpy()
                    + band["eeg_hf_30_45_power"].to_numpy(),
                }
            )
            rms_values.append(tmp[["epoch_idx", f"{kind}_rms"]])
            hf_values.append(tmp[["epoch_idx", f"{kind}_hf_20_45_power"]])
        if rms_values:
            rms_df = pd.concat(rms_values).groupby("epoch_idx", as_index=False)[f"{kind}_rms"].mean()
            rows = rows.merge(rms_df, on="epoch_idx", how="left")
        else:
            rows[f"{kind}_rms"] = np.nan
        if hf_values:
            hf_df = pd.concat(hf_values).groupby("epoch_idx", as_index=False)[f"{kind}_hf_20_45_power"].mean()
            rows = rows.merge(hf_df, on="epoch_idx", how="left")
        else:
            rows[f"{kind}_hf_20_45_power"] = np.nan
        rows[f"{kind}_channel_count"] = len(channels)
    return rows


def transition_covariates(metadata: pd.DataFrame) -> pd.DataFrame:
    base = metadata[["subject_id", "night_id", "epoch_idx", "stage"]].drop_duplicates(
        ["subject_id", "night_id", "epoch_idx"]
    )
    rows = []
    for (subject_id, night_id), df in base.groupby(["subject_id", "night_id"], sort=False):
        df = df.sort_values("epoch_idx").reset_index(drop=True)
        epoch_idx = df["epoch_idx"].to_numpy(dtype=int)
        stage = df["stage"].astype(str).to_numpy()
        is_change = np.zeros(len(df), dtype=bool)
        if len(df) > 1:
            changes = np.flatnonzero(stage[1:] != stage[:-1]) + 1
            is_change[changes] = True
            is_change[changes - 1] = True
        wake_like = np.isin(stage, ["W", "N1", "REM", "MOVEMENT", "UNKNOWN"])
        for pos, idx in enumerate(epoch_idx):
            lo = max(0, pos - TRANSITION_WINDOW_EPOCHS)
            hi = min(len(df), pos + TRANSITION_WINDOW_EPOCHS + 1)
            rows.append(
                {
                    "subject_id": subject_id,
                    "night_id": night_id,
                    "epoch_idx": idx,
                    "stage_transition_within_2min": bool(is_change[lo:hi].any()),
                    "wake_n1_rem_within_2min": bool(wake_like[lo:hi].any()),
                }
            )
    return pd.DataFrame(rows)


def read_anphy_artifact_matrix(dataset_root: Path, subject_id: str) -> pd.DataFrame:
    zip_path = dataset_root / "Artifact matrix.zip"
    if not zip_path.exists():
        return pd.DataFrame()
    with zipfile.ZipFile(zip_path) as zf:
        matches = [item for item in zf.infolist() if subject_id.lower() in item.filename.lower() and item.filename.endswith(".mat")]
        if not matches:
            return pd.DataFrame()
        data = zf.read(matches[0])
    arrays = []
    try:
        mat = scipy.io.loadmat(io.BytesIO(data))
        for key, value in mat.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                arrays.append(arr)
    except NotImplementedError:
        try:
            import h5py

            def visit_hdf5(_name: str, obj: object) -> None:
                if isinstance(obj, h5py.Dataset):
                    arr = np.asarray(obj)
                    if arr.ndim == 2 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                        arrays.append(arr)

            with h5py.File(io.BytesIO(data), "r") as h5:
                h5.visititems(visit_hdf5)
        except Exception as exc:
            print(f"{subject_id}: artifact matrix HDF5 read failed {type(exc).__name__}: {exc}", flush=True)
    if not arrays:
        return pd.DataFrame()
    arr = max(arrays, key=lambda x: x.size)
    # ANPHY matrices encode artifact information by channel/epoch, but
    # orientation can vary. Use only recording-level bad-epoch summaries here.
    if arr.shape[0] <= arr.shape[1]:
        by_epoch = np.nanmean(arr != 0, axis=0)
    else:
        by_epoch = np.nanmean(arr != 0, axis=1)
    return pd.DataFrame({"epoch_idx": np.arange(len(by_epoch), dtype=int), "anphy_artifact_fraction_channels": by_epoch})


def build_sleep_edf_recording_map(table_dir: Path, dataset_root: Path, spec: DatasetSpec) -> dict[str, Path]:
    recordings = pd.read_csv(table_dir / str(spec.recording_inventory_path))
    mapping = {}
    for _, row in recordings.iterrows():
        if pd.isna(row.get("recording_key")) or pd.isna(row.get("psg_relative_path")):
            continue
        mapping[str(row["recording_key"])] = dataset_root / str(row["psg_relative_path"])
    return mapping


def build_anphy_recording_map(dataset_root: Path) -> dict[str, object]:
    _, recordings = discover_recordings(dataset_root)
    mapping = {}
    for recording in recordings:
        mapping[recording.night_id] = recording
    return mapping


def cleanup_anphy_cached_edf(recording: object, cache_root: Path) -> None:
    cache_base = (cache_root / "anphy_edf_cache").resolve()
    out_dir = (cache_base / recording.subject_id).resolve()
    if cache_base not in out_dir.parents:
        raise ValueError(f"Refusing to clean unexpected cache directory: {out_dir}")

    out_path = out_dir / Path(recording.edf_member).name
    candidates = [out_path, out_path.with_suffix(out_path.suffix + ".part")]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
    try:
        out_dir.rmdir()
    except OSError:
        pass


def compute_artifacts_for_recording(
    night_id: str,
    raw_path: Path,
    matrix_rec: pd.DataFrame,
    dataset: str,
    dataset_root: Path,
) -> pd.DataFrame:
    started = time.time()
    raw = mne.io.read_raw_edf(raw_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(EPOCH_SEC * sfreq))
    raw_by_key = {norm_channel(ch): ch for ch in raw.ch_names}

    epoch_indices = np.array(sorted(matrix_rec["epoch_idx"].dropna().astype(int).unique()), dtype=int)
    aux = compute_aux_epoch_covariates(raw, epoch_indices, samples_per_epoch)
    if dataset == "anphy":
        subject_id = str(matrix_rec["subject_id"].iloc[0])
        try:
            artifact = read_anphy_artifact_matrix(dataset_root, subject_id)
        except Exception as exc:
            print(f"{subject_id}: artifact matrix skipped {type(exc).__name__}: {exc}", flush=True)
            artifact = pd.DataFrame()
        if not artifact.empty:
            aux = aux.merge(artifact, on="epoch_idx", how="left")
        else:
            aux["anphy_artifact_fraction_channels"] = np.nan

    rows = []
    for channel_key, df_ch in matrix_rec.groupby("channel_key", sort=False):
        raw_ch = raw_by_key.get(str(channel_key))
        if raw_ch is None:
            continue
        ch_epochs = np.array(sorted(df_ch["epoch_idx"].dropna().astype(int).unique()), dtype=int)
        data_uv = raw.get_data(picks=[raw_ch], verbose="ERROR")[0] * 1_000_000.0
        epochs_uv, kept = channel_epochs(data_uv, ch_epochs, samples_per_epoch)
        if epochs_uv.size == 0:
            continue
        band = epoch_bandpowers(epochs_uv, sfreq)
        part = pd.DataFrame(
            {
                "subject_id": df_ch["subject_id"].iloc[0],
                "night_id": night_id,
                "epoch_idx": kept,
                "channel_key": channel_key,
                "channel": raw_ch,
            }
        )
        part = pd.concat([part, band], axis=1)
        rows.append(part)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        out = out.merge(aux, on="epoch_idx", how="left")
    print(
        f"{night_id}: artifact rows={len(out)} channels={out['channel'].nunique() if not out.empty else 0} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return out


def compute_artifact_covariates(cfg: dict, spec: DatasetSpec, max_recordings: int | None = None) -> pd.DataFrame:
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    cache_root = cfg["cache_root"]
    matrix = pd.read_csv(table_dir / spec.matrix_path, low_memory=False)
    matrix = matrix[matrix["stage"] == "N3"].copy()
    matrix["channel_key"] = matrix["channel"].astype(str).map(norm_channel)

    if spec.dataset == "anphy":
        recording_map = build_anphy_recording_map(dataset_root)
    else:
        recording_map = build_sleep_edf_recording_map(table_dir, dataset_root, spec)

    nights = [night for night in matrix["night_id"].dropna().astype(str).unique() if night in recording_map]
    if max_recordings is not None:
        nights = nights[:max_recordings]
    rows = []
    for night_id in nights:
        rec = matrix[matrix["night_id"].astype(str) == night_id].copy()
        anphy_recording = None
        try:
            raw_path = recording_map[night_id]
            if spec.dataset == "anphy":
                anphy_recording = raw_path
                raw_path = extract_edf_to_cache(raw_path, cache_root, overwrite_cache=False)
            rows.append(compute_artifacts_for_recording(night_id, Path(raw_path), rec, spec.dataset, dataset_root))
        except Exception as exc:
            print(f"{night_id}: artifact error {type(exc).__name__}: {exc}", flush=True)
        finally:
            if anphy_recording is not None:
                cleanup_anphy_cached_edf(anphy_recording, cache_root)
    artifacts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    artifacts.to_csv(table_dir / spec.artifact_covariates_path, index=False, compression="gzip")
    return artifacts


def build_augmented_matrix(cfg: dict, spec: DatasetSpec) -> pd.DataFrame:
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / spec.matrix_path, low_memory=False)
    artifacts = pd.read_csv(table_dir / spec.artifact_covariates_path, low_memory=False)
    metadata = pd.read_csv(table_dir / spec.metadata_path, low_memory=False)
    transitions = transition_covariates(metadata)

    matrix["channel_key"] = matrix["channel"].astype(str).map(norm_channel)
    artifacts["channel_key"] = artifacts["channel_key"].astype(str)
    merge_keys = ["subject_id", "night_id", "epoch_idx", "channel_key"]
    augmented = matrix.merge(
        artifacts.drop(columns=["channel"], errors="ignore"),
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )
    augmented = augmented.merge(transitions, on=["subject_id", "night_id", "epoch_idx"], how="left", validate="many_to_one")
    augmented.to_csv(table_dir / spec.augmented_matrix_path, index=False, compression="gzip")
    return augmented


def prepare_artifact_model_frame(matrix: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = matrix[bool_series(matrix["analysis_qc_include"])].copy()
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    df["channel"] = df["channel"].astype(str)

    if scenario == "exclude_hf_top10":
        cutoff = df["eeg_hf_30_45_ratio"].quantile(0.90)
        df = df[df["eeg_hf_30_45_ratio"] <= cutoff].copy()
    elif scenario == "exclude_hf_top20":
        cutoff = df["eeg_hf_30_45_ratio"].quantile(0.80)
        df = df[df["eeg_hf_30_45_ratio"] <= cutoff].copy()
    elif scenario == "exclude_transition2":
        df = df[~bool_series(df["wake_n1_rem_within_2min"])].copy()
    elif scenario == "exclude_hf_top10_transition2":
        cutoff = df["eeg_hf_30_45_ratio"].quantile(0.90)
        df = df[(df["eeg_hf_30_45_ratio"] <= cutoff) & ~bool_series(df["wake_n1_rem_within_2min"])].copy()
    elif scenario == "exclude_emg_top10" and df["emg_rms"].notna().any():
        cutoff = df["emg_rms"].quantile(0.90)
        df = df[df["emg_rms"] <= cutoff].copy()
    elif scenario == "exclude_anphy_artifact" and "anphy_artifact_fraction_channels" in df:
        df = df[df["anphy_artifact_fraction_channels"].fillna(0) == 0].copy()

    df["log_cumulative_swa"] = np.log1p(pd.to_numeric(df["cumulative_swa"], errors="coerce").clip(lower=0))
    continuous = [
        "time_since_sleep_onset",
        "position_within_bout_fraction",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
        "log_cumulative_swa",
        "eeg_beta_20_30_power",
        "eeg_hf_30_45_power",
        "eeg_hf_35_45_power",
        "eeg_hf_30_45_ratio",
        "eeg_hf_35_45_ratio",
        "eog_rms",
        "eog_hf_20_45_power",
        "emg_rms",
        "emg_hf_20_45_power",
        "anphy_artifact_fraction_channels",
    ]
    for col in continuous:
        if col not in df:
            df[col] = np.nan
        values = pd.to_numeric(df[col], errors="coerce")
        if col.endswith("_power") or col.endswith("_rms"):
            values = np.log1p(values.clip(lower=0))
        df[f"{col}_z"] = zscore(values)
    for feature in FEATURES:
        df[f"{feature}_z"] = zscore(df[feature])
    df["time_since_sleep_onset_z2"] = df["time_since_sleep_onset_z"] ** 2
    df["time_since_sleep_onset_z3"] = df["time_since_sleep_onset_z"] ** 3

    required = [f"{feature}_z" for feature in FEATURES] + [
        "time_since_sleep_onset_z",
        "position_within_bout_fraction_z",
        "relative_delta_power_z",
        "slow_wave_density_z",
        "slow_wave_occupancy_z",
        "log_cumulative_swa_z",
        "eeg_hf_30_45_ratio_z",
        "eeg_hf_35_45_ratio_z",
        "eeg_beta_20_30_power_z",
        "subject_id",
        "night_id",
        "channel",
    ]
    optional = []
    for col in ["eog_rms_z", "eog_hf_20_45_power_z", "emg_rms_z", "emg_hf_20_45_power_z", "anphy_artifact_fraction_channels_z"]:
        if df[col].notna().any():
            optional.append(col)
    return df.dropna(subset=required).copy(), optional


def artifact_formula(feature: str, optional_controls: list[str], include_night_fe: bool) -> str:
    fixed_effects = "C(subject_id) + C(channel)"
    if include_night_fe:
        fixed_effects += " + C(night_id)"
    controls = [
        "time_since_sleep_onset_z",
        "time_since_sleep_onset_z2",
        "time_since_sleep_onset_z3",
        "position_within_bout_fraction_z",
        "relative_delta_power_z",
        "slow_wave_density_z",
        "slow_wave_occupancy_z",
        "log_cumulative_swa_z",
        "eeg_beta_20_30_power_z",
        "eeg_hf_30_45_ratio_z",
        "eeg_hf_35_45_ratio_z",
    ] + optional_controls
    return f"{feature}_z ~ " + " + ".join(controls + [fixed_effects])


def fit_artifact_models(df: pd.DataFrame, optional_controls: list[str], include_night_fe: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variance_rows = []
    fixed_rows = []
    residuals = df[
        [
            "subject_id",
            "night_id",
            "epoch_idx",
            "channel",
            "time_since_sleep_onset",
            "position_within_bout_fraction",
            "relative_delta_power",
            "slow_wave_density",
            "slow_wave_occupancy",
            "cumulative_swa",
            "eeg_hf_30_45_ratio",
            "eeg_hf_35_45_ratio",
            "eeg_beta_20_30_power",
            "eog_rms",
            "emg_rms",
            "wake_n1_rem_within_2min",
            "stage_transition_within_2min",
        ]
    ].copy()
    for feature in FEATURES:
        formula = artifact_formula(feature, optional_controls, include_night_fe)
        result = smf.ols(formula, data=df).fit()
        fitted = np.asarray(result.fittedvalues)
        resid = np.asarray(result.resid)
        var_fixed = float(np.nanvar(fitted, ddof=0))
        var_resid = float(np.nanvar(resid, ddof=0))
        total = var_fixed + var_resid
        variance_rows.append(
            {
                "feature": feature,
                "formula": formula,
                "n_rows": int(result.nobs),
                "n_subjects": int(df["subject_id"].nunique()),
                "n_nights": int(df["night_id"].nunique()),
                "model_family": "ols_subject_channel_artifact_controls",
                "var_fixed": var_fixed,
                "var_residual": var_resid,
                "marginal_r2_approx": var_fixed / total if total > 0 else np.nan,
                "residual_sd": float(np.nanstd(resid, ddof=0)),
                "residual_iqr": float(np.nanpercentile(resid, 75) - np.nanpercentile(resid, 25)),
                "aic": float(result.aic),
                "bic": float(result.bic),
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
        residuals[f"{feature}_fitted_z"] = fitted
        residuals[f"{feature}_rncp_residual_z"] = resid
        print(f"model {feature}: rows={int(result.nobs)} resid_sd={np.nanstd(resid, ddof=0):.4f}", flush=True)
    residuals["rncp_l2_norm"] = np.sqrt(np.square(residuals[RESIDUAL_COLS]).sum(axis=1))
    return pd.DataFrame(variance_rows), pd.concat(fixed_rows, ignore_index=True), residuals


def run_artifact_models(cfg: dict, spec: DatasetSpec, scenarios: list[str], n_folds: int, n_null: int, seed: int) -> None:
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / spec.augmented_matrix_path, low_memory=False)
    include_night_fe = spec.dataset.startswith("sleep_edf_")
    summary_rows = []
    all_pairs = []
    for i, scenario in enumerate(scenarios):
        df, optional = prepare_artifact_model_frame(matrix, scenario)
        if len(df) < 100 or df["subject_id"].nunique() < 5:
            print(f"Skipping {scenario}: insufficient rows/subjects", flush=True)
            continue
        print(f"Running scenario {scenario}: rows={len(df)} optional={optional}", flush=True)
        variance, fixed, residuals = fit_artifact_models(df, optional, include_night_fe=include_night_fe)
        out_stem = f"{spec.summary_prefix}_{scenario}"
        variance.to_csv(table_dir / f"{out_stem}_model_variance_summary.csv", index=False)
        fixed.to_csv(table_dir / f"{out_stem}_model_fixed_effects.csv", index=False)
        residuals.to_csv(table_dir / f"{out_stem}_rncp_residuals.csv.gz", index=False, compression="gzip")

        repro_df = residuals.dropna(subset=["subject_id", "night_id", "channel"] + RESIDUAL_COLS).reset_index(drop=True)
        null_df, global_summary = run_global_null(repro_df, n_null, seed + i * 100)
        fold_df = run_fold_reproducibility(repro_df, n_folds, max(100, n_null // 2), seed + i * 100 + 1)
        obs_corr = repro_df[RESIDUAL_COLS].corr()
        pairs = observed_pair_table(obs_corr)
        pairs["scenario"] = scenario
        all_pairs.append(pairs)
        null_df.to_csv(table_dir / f"{out_stem}_global_null_iterations.csv", index=False)
        pd.DataFrame([global_summary]).to_csv(table_dir / f"{out_stem}_global_null_summary.csv", index=False)
        fold_df.to_csv(table_dir / f"{out_stem}_fold_reproducibility.csv", index=False)
        obs_corr.to_csv(table_dir / f"{out_stem}_observed_correlation.csv")
        summary_rows.append(
            {
                "scenario": scenario,
                "rows": len(repro_df),
                "subjects": repro_df["subject_id"].nunique(),
                "nights": repro_df["night_id"].nunique(),
                "observed_mean_abs_offdiag_corr": global_summary["observed_mean_abs_offdiag_corr"],
                "null_mean_abs_offdiag_corr_mean": global_summary["null_mean_abs_offdiag_corr_mean"],
                "global_p_high": global_summary["null_mean_abs_offdiag_corr_p_high"],
                "mean_fold_similarity": float(fold_df["observed_train_heldout_vector_similarity"].mean()),
                "mean_fold_p_high": float(fold_df["similarity_p_high"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(table_dir / f"{spec.summary_prefix}_artifact_robustness_summary.csv", index=False)
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(table_dir / f"{spec.summary_prefix}_artifact_pairwise_correlations.csv", index=False)
    write_artifact_summary(table_dir, spec, summary)


def write_artifact_summary(table_dir: Path, spec: DatasetSpec, summary: pd.DataFrame) -> None:
    lines = [
        f"# Artifact Robustness Summary: {spec.dataset}",
        "",
        "## Design",
        "",
        "Models use the existing N3 primary analysis matrix plus explicit artifact controls:",
        "",
        "- EEG high-frequency power and HF/total ratios: 20-30 Hz, 30-45 Hz, 35-45 Hz.",
        "- EOG RMS and 20-45 Hz power where EOG channels are present.",
        "- EMG RMS and 20-45 Hz power where EMG/chin channels are present.",
        "- Exclusion scenarios for high-frequency outliers and epochs near W/N1/REM transitions.",
        "- Subject and channel fixed effects; Sleep-EDF SC also includes night fixed effects.",
        "",
        "## Scenario Results",
        "",
        summary.to_markdown(index=False, floatfmt=".4f") if not summary.empty else "No completed scenarios.",
        "",
        "## Interpretation",
        "",
        "RNCP is considered artifact-robust only if the observed residual covariance remains non-random and directionally stable after HF/EMG/EOG controls and transition-proximity exclusions.",
    ]
    (table_dir / f"{spec.summary_prefix}_artifact_robustness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--step", choices=["extract", "augment", "models", "all"], default="all")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["artifact_adjusted", "exclude_hf_top10", "exclude_hf_top20", "exclude_transition2", "exclude_hf_top10_transition2", "exclude_emg_top10", "exclude_anphy_artifact"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = SPECS[args.dataset]
    started = time.time()
    if args.step in {"extract", "all"}:
        compute_artifact_covariates(cfg, spec, max_recordings=args.max_recordings)
    if args.step in {"augment", "all"}:
        build_augmented_matrix(cfg, spec)
    if args.step in {"models", "all"}:
        run_artifact_models(cfg, spec, args.scenarios, args.n_folds, args.n_null, args.seed)
    print(f"{args.dataset} {args.step} analysis complete in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
