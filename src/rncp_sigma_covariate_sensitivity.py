#!/usr/bin/env python3
"""Sleep-EDF SC RNCP sensitivity with sigma-band power as an added covariate."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import signal

from sleep_edf_residual_models import (
    FEATURES,
    bool_series,
    fixed_effects_table,
    prepare_model_frame,
    residual_frame,
    variance_summary,
    zscore,
)
from sleep_edf_rncp_reproducibility import (
    observed_pair_table,
    residual_corr,
    run_fold_reproducibility,
    run_global_null,
)
from sleep_edf_signal_features import EPOCH_SEC, EPS, bandpower


KEYS = ["subject_id", "night_id", "epoch_idx", "channel"]
SIGMA_LOW_HZ = 12.0
SIGMA_HIGH_HZ = 15.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def compute_sigma_for_channel(data_uv: np.ndarray, epoch_indices: np.ndarray, sfreq: float) -> pd.DataFrame:
    samples_per_epoch = int(round(EPOCH_SEC * sfreq))
    valid_epoch_indices = []
    epochs = []
    for epoch_idx in epoch_indices:
        start = int(epoch_idx) * samples_per_epoch
        stop = start + samples_per_epoch
        if start < 0 or stop > data_uv.size:
            continue
        valid_epoch_indices.append(int(epoch_idx))
        epochs.append(data_uv[start:stop])

    if not epochs:
        return pd.DataFrame(columns=["epoch_idx", "sigma_power_12_15", "relative_sigma_power_12_15"])

    epochs_uv = np.asarray(epochs, dtype=float)
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
    sigma_power = bandpower(freqs, psd, SIGMA_LOW_HZ, SIGMA_HIGH_HZ)
    total_power = bandpower(freqs, psd, 0.5, min(45.0, sfreq / 2.0 - 0.1))
    relative_sigma = np.divide(sigma_power, total_power + EPS)

    return pd.DataFrame(
        {
            "epoch_idx": valid_epoch_indices,
            "sigma_power_12_15": sigma_power,
            "relative_sigma_power_12_15": relative_sigma,
        }
    )


def compute_sigma_features(
    table_dir: Path,
    dataset_root: Path,
    matrix: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    out_path = table_dir / "sleep_edf_sc_sigma_features.csv.gz"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path, low_memory=False)

    recordings = pd.read_csv(table_dir / "sleep_edf_sc_recording_inventory.csv")
    needed = matrix[KEYS].drop_duplicates().copy()
    needed["night_id"] = needed["night_id"].astype(str)
    needed["channel"] = needed["channel"].astype(str)
    needed["epoch_idx"] = pd.to_numeric(needed["epoch_idx"], errors="coerce").astype("Int64")
    needed = needed.dropna(subset=["epoch_idx"])

    frames = []
    started = time.time()
    for i, rec in recordings.iterrows():
        key = str(rec["recording_key"])
        needed_rec = needed[needed["night_id"] == key]
        if needed_rec.empty:
            continue

        psg_path = dataset_root / rec["psg_relative_path"]
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        lower_to_raw = {ch.lower(): ch for ch in raw.ch_names}

        rec_frames = []
        for channel, needed_ch in needed_rec.groupby("channel", sort=False):
            raw_channel = channel if channel in raw.ch_names else lower_to_raw.get(str(channel).lower())
            if raw_channel is None:
                continue

            data_uv = raw.get_data(picks=[raw_channel], verbose="ERROR")[0] * 1_000_000.0
            epoch_indices = needed_ch["epoch_idx"].dropna().astype(int).unique()
            ch_sigma = compute_sigma_for_channel(data_uv, epoch_indices, sfreq)
            if ch_sigma.empty:
                continue
            ch_sigma["subject_id"] = needed_ch["subject_id"].iloc[0]
            ch_sigma["night_id"] = key
            ch_sigma["channel"] = channel
            rec_frames.append(ch_sigma)

        if rec_frames:
            rec_out = pd.concat(rec_frames, ignore_index=True)
            frames.append(rec_out)
            print(
                f"{i + 1:03d}/{len(recordings)} {key}: sigma rows={len(rec_out)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    sigma = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    sigma = sigma[KEYS + ["sigma_power_12_15", "relative_sigma_power_12_15"]]
    sigma.to_csv(out_path, index=False, compression="gzip")
    return sigma


def prepare_sigma_model_frame(matrix: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    merged = matrix.merge(sigma, on=KEYS, how="left", validate="one_to_one")
    df = prepare_model_frame(merged)
    df["log_sigma_power_12_15"] = np.log1p(
        pd.to_numeric(df["sigma_power_12_15"], errors="coerce").clip(lower=0)
    )
    df["relative_sigma_power_12_15_z"] = zscore(df["relative_sigma_power_12_15"])
    df["log_sigma_power_12_15_z"] = zscore(df["log_sigma_power_12_15"])
    required = ["relative_sigma_power_12_15_z", "log_sigma_power_12_15_z"]
    return df.dropna(subset=required).copy()


def fixed_formula_sigma(outcome: str) -> str:
    return (
        f"{outcome}_z ~ "
        "age_z + C(sex) + "
        "time_since_sleep_onset_z + time_since_sleep_onset_z2 + time_since_sleep_onset_z3 + "
        "position_within_bout_fraction_z + "
        "relative_delta_power_z + slow_wave_density_z + slow_wave_occupancy_z + log_cumulative_swa_z + "
        "relative_sigma_power_12_15_z + "
        "C(channel)"
    )


def fit_sigma_model(df: pd.DataFrame, feature: str):
    formula = fixed_formula_sigma(feature)
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["subject_id"],
        re_formula="1",
        vc_formula={"night": "0 + C(night_id)"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(method="lbfgs", reml=True, maxiter=500, disp=False)
    return formula, result, [str(item.message) for item in caught]


def write_summary(
    table_dir: Path,
    df: pd.DataFrame,
    variance: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    obs_corr: pd.DataFrame,
    global_summary: dict,
    fold_df: pd.DataFrame,
    n_null: int,
    seed: int,
) -> None:
    obs_pairs = observed_pair_table(obs_corr)
    fold_mean = float(fold_df["observed_train_heldout_vector_similarity"].mean())
    lines = [
        "# RNCP Sigma-Band Covariate Sensitivity",
        "",
        "## Design",
        "",
        "Sleep-EDF SC primary rows were re-modeled after adding relative sigma-band power (12-15 Hz / 0.5-45 Hz total power) as an additional fixed-effect covariate.",
        "Sigma power was computed from the same Welch spectra definition used for the signal-feature signal features.",
        "",
        f"- Rows modeled: {len(df)}",
        f"- Subjects: {df['subject_id'].nunique()}",
        f"- Nights: {df['night_id'].nunique()}",
        f"- Null iterations: {n_null}",
        f"- Random seed: {seed}",
        "",
        "## Variance Summary",
        "",
        variance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Sigma Fixed Effect Terms",
        "",
        fixed_effects[fixed_effects["term"] == "relative_sigma_power_12_15_z"].to_markdown(
            index=False, floatfmt=".4f"
        ),
        "",
        "## RNCP Residual Correlation",
        "",
        obs_corr.to_markdown(floatfmt=".4f"),
        "",
        "## Observed Pairwise RNCP Correlations",
        "",
        obs_pairs.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Block-Preserving Null Test",
        "",
        f"- Observed mean absolute off-diagonal correlation: {global_summary['observed_mean_abs_offdiag_corr']:.4f}",
        f"- Null mean: {global_summary['null_mean_abs_offdiag_corr_mean']:.4f}",
        f"- Null SD: {global_summary['null_mean_abs_offdiag_corr_sd']:.4f}",
        f"- Empirical p, observed >= null: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}",
        "",
        "## Subject-Held-Out Reproducibility",
        "",
        f"- Mean fold similarity: {fold_mean:.4f}",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation",
        "",
        "This analysis tests whether the primary SC RNCP structure is reducible to residual sigma/spindle-band power.",
    ]
    (table_dir / "rncp_sigma_covariate_sensitivity_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--overwrite-sigma", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / "sleep_edf_sc_n3_analysis_matrix_primary.csv.gz", low_memory=False)
    matrix = matrix[bool_series(matrix["analysis_qc_include"])].copy()

    sigma = compute_sigma_features(table_dir, cfg["dataset_root"], matrix, overwrite=args.overwrite_sigma)
    df = prepare_sigma_model_frame(matrix, sigma)

    variance_rows = []
    fixed_effect_frames = []
    model_outputs = {}
    for feature in FEATURES:
        print(f"Fitting sigma-adjusted model: {feature}", flush=True)
        formula, result, warning_messages = fit_sigma_model(df, feature)
        model_outputs[feature] = result
        variance_rows.append(variance_summary(result, df, feature, formula, warning_messages))
        fixed_effect_frames.append(fixed_effects_table(result, feature))
        print(
            f"{feature}: converged={getattr(result, 'converged', False)} "
            f"resid_sd={np.nanstd(result.resid, ddof=0):.4f}",
            flush=True,
        )

    variance = pd.DataFrame(variance_rows)
    fixed_effects = pd.concat(fixed_effect_frames, ignore_index=True)
    residuals = residual_frame(df, model_outputs)
    residuals["sigma_power_12_15"] = df["sigma_power_12_15"].to_numpy()
    residuals["relative_sigma_power_12_15"] = df["relative_sigma_power_12_15"].to_numpy()

    obs_corr = residual_corr(residuals)
    null_df, global_summary = run_global_null(residuals, args.n_null, args.seed)
    fold_df = run_fold_reproducibility(residuals, args.n_folds, args.n_null, args.seed + 1)

    variance.to_csv(table_dir / "rncp_sigma_covariate_model_variance_summary.csv", index=False)
    fixed_effects.to_csv(table_dir / "rncp_sigma_covariate_fixed_effects.csv", index=False)
    residuals.to_csv(
        table_dir / "sleep_edf_sc_n3_rncp_sigma_covariate_residuals.csv.gz",
        index=False,
        compression="gzip",
    )
    obs_corr.to_csv(table_dir / "rncp_sigma_covariate_observed_correlation.csv")
    observed_pair_table(obs_corr).to_csv(
        table_dir / "rncp_sigma_covariate_pairwise_correlations.csv", index=False
    )
    null_df.to_csv(table_dir / "rncp_sigma_covariate_global_null_iterations.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(
        table_dir / "rncp_sigma_covariate_global_null_summary.csv", index=False
    )
    fold_df.to_csv(table_dir / "rncp_sigma_covariate_fold_reproducibility.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "Sleep-EDF SC",
                "scenario": "sigma_covariate_12_15hz",
                "rows": int(len(residuals)),
                "subjects": int(residuals["subject_id"].nunique()),
                "observed_mean_abs_offdiag_corr": global_summary["observed_mean_abs_offdiag_corr"],
                "null_mean_abs_offdiag_corr": global_summary["null_mean_abs_offdiag_corr_mean"],
                "null_sd_abs_offdiag_corr": global_summary["null_mean_abs_offdiag_corr_sd"],
                "empirical_p": global_summary["null_mean_abs_offdiag_corr_p_high"],
                "fold_similarity_mean": float(fold_df["observed_train_heldout_vector_similarity"].mean()),
                "fold_similarity_median": float(fold_df["observed_train_heldout_vector_similarity"].median()),
            }
        ]
    ).to_csv(table_dir / "rncp_sigma_covariate_sensitivity_summary.csv", index=False)

    write_summary(
        table_dir=table_dir,
        df=residuals,
        variance=variance,
        fixed_effects=fixed_effects,
        obs_corr=obs_corr,
        global_summary=global_summary,
        fold_df=fold_df,
        n_null=args.n_null,
        seed=args.seed,
    )

    print(f"Rows modeled: {len(residuals)}")
    print(f"Observed mean abs offdiag corr: {global_summary['observed_mean_abs_offdiag_corr']:.4f}")
    print(f"Null mean: {global_summary['null_mean_abs_offdiag_corr_mean']:.4f}")
    print(f"Empirical p: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}")
    print(f"Mean fold similarity: {fold_df['observed_train_heldout_vector_similarity'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
