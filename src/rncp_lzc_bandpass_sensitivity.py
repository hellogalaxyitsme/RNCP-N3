#!/usr/bin/env python3
"""Sleep-EDF SC RNCP sensitivity with LZc computed after 0.5-45 Hz filtering."""

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
import antropy as ant
from scipy import signal

from sleep_edf_residual_models import (
    FEATURES,
    bool_series,
    fixed_effects_table,
    fixed_formula,
    prepare_model_frame,
    residual_frame,
    variance_summary,
)
from sleep_edf_rncp_reproducibility import (
    observed_pair_table,
    residual_corr,
    run_fold_reproducibility,
    run_global_null,
)
from sleep_edf_signal_features import EPOCH_SEC


KEYS = ["subject_id", "night_id", "epoch_idx", "channel"]
LOW_HZ = 0.5
HIGH_HZ = 45.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bandpass_continuous(data_uv: np.ndarray, sfreq: float) -> np.ndarray:
    high = min(HIGH_HZ, sfreq / 2.0 - 0.5)
    sos = signal.butter(4, [LOW_HZ, high], btype="bandpass", fs=sfreq, output="sos")
    try:
        return signal.sosfiltfilt(sos, data_uv)
    except ValueError:
        return signal.sosfilt(sos, data_uv)


def compute_lzc_for_channel(data_uv: np.ndarray, epoch_indices: np.ndarray, sfreq: float) -> pd.DataFrame:
    filtered = bandpass_continuous(data_uv, sfreq)
    samples_per_epoch = int(round(EPOCH_SEC * sfreq))
    rows = []
    for epoch_idx in epoch_indices:
        start = int(epoch_idx) * samples_per_epoch
        stop = start + samples_per_epoch
        if start < 0 or stop > filtered.size:
            continue
        epoch = filtered[start:stop]
        clean = np.nan_to_num(epoch, nan=np.nanmedian(epoch))
        bits = (clean > np.median(clean)).astype(np.uint8)
        n = bits.size
        lzc = float(ant.lziv_complexity(bits, normalize=False) * np.log2(n) / n) if n > 1 else np.nan
        rows.append(
            {
                "epoch_idx": int(epoch_idx),
                "lzc_bandpass_0p5_45": lzc,
            }
        )
    return pd.DataFrame(rows)


def compute_bandpass_lzc_features(
    table_dir: Path,
    dataset_root: Path,
    matrix: pd.DataFrame,
    overwrite: bool,
) -> pd.DataFrame:
    out_path = table_dir / "sleep_edf_sc_lzc_bandpass_0p5_45_features.csv.gz"
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

        raw = mne.io.read_raw_edf(dataset_root / rec["psg_relative_path"], preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        lower_to_raw = {ch.lower(): ch for ch in raw.ch_names}
        rec_frames = []

        for channel, needed_ch in needed_rec.groupby("channel", sort=False):
            raw_channel = channel if channel in raw.ch_names else lower_to_raw.get(str(channel).lower())
            if raw_channel is None:
                continue
            data_uv = raw.get_data(picks=[raw_channel], verbose="ERROR")[0] * 1_000_000.0
            epoch_indices = needed_ch["epoch_idx"].dropna().astype(int).unique()
            ch_lzc = compute_lzc_for_channel(data_uv, epoch_indices, sfreq)
            if ch_lzc.empty:
                continue
            ch_lzc["subject_id"] = needed_ch["subject_id"].iloc[0]
            ch_lzc["night_id"] = key
            ch_lzc["channel"] = channel
            rec_frames.append(ch_lzc)

        if rec_frames:
            rec_out = pd.concat(rec_frames, ignore_index=True)
            frames.append(rec_out)
            print(
                f"{i + 1:03d}/{len(recordings)} {key}: filtered LZc rows={len(rec_out)} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    lzc = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    lzc = lzc[KEYS + ["lzc_bandpass_0p5_45"]]
    lzc.to_csv(out_path, index=False, compression="gzip")
    return lzc


def prepare_lzc_bandpass_model_frame(matrix: pd.DataFrame, lzc_filtered: pd.DataFrame) -> pd.DataFrame:
    merged = matrix.merge(lzc_filtered, on=KEYS, how="left", validate="one_to_one")
    merged["lzc_broadband_original"] = merged["lzc"]
    merged["lzc"] = merged["lzc_bandpass_0p5_45"]
    return prepare_model_frame(merged)


def fit_model(df: pd.DataFrame, feature: str):
    formula = fixed_formula(feature)
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
        "# RNCP Bandpass-Filtered LZc Sensitivity",
        "",
        "## Design",
        "",
        "Sleep-EDF SC primary rows were re-modeled after replacing broadband median-binarized Lempel-Ziv complexity with LZc computed from continuous EEG filtered at 0.5-45 Hz before epoching.",
        "All other primary features and model covariates were unchanged.",
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
        "This analysis tests whether the primary SC RNCP structure depends on broadband LZc sensitivity to high-frequency signal content.",
    ]
    (table_dir / "rncp_lzc_bandpass_sensitivity_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--overwrite-lzc", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / "sleep_edf_sc_n3_analysis_matrix_primary.csv.gz", low_memory=False)
    matrix = matrix[bool_series(matrix["analysis_qc_include"])].copy()

    lzc_filtered = compute_bandpass_lzc_features(
        table_dir=table_dir,
        dataset_root=cfg["dataset_root"],
        matrix=matrix,
        overwrite=args.overwrite_lzc,
    )
    df = prepare_lzc_bandpass_model_frame(matrix, lzc_filtered)

    variance_rows = []
    fixed_effect_frames = []
    model_outputs = {}
    for feature in FEATURES:
        print(f"Fitting bandpass-LZc sensitivity model: {feature}", flush=True)
        formula, result, warning_messages = fit_model(df, feature)
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
    residuals["lzc_bandpass_0p5_45"] = df["lzc"].to_numpy()
    residuals["lzc_broadband_original"] = df["lzc_broadband_original"].to_numpy()

    obs_corr = residual_corr(residuals)
    null_df, global_summary = run_global_null(residuals, args.n_null, args.seed)
    fold_df = run_fold_reproducibility(residuals, args.n_folds, args.n_null, args.seed + 1)

    variance.to_csv(table_dir / "rncp_lzc_bandpass_model_variance_summary.csv", index=False)
    fixed_effects.to_csv(table_dir / "rncp_lzc_bandpass_fixed_effects.csv", index=False)
    residuals.to_csv(
        table_dir / "sleep_edf_sc_n3_rncp_lzc_bandpass_residuals.csv.gz",
        index=False,
        compression="gzip",
    )
    obs_corr.to_csv(table_dir / "rncp_lzc_bandpass_observed_correlation.csv")
    observed_pair_table(obs_corr).to_csv(table_dir / "rncp_lzc_bandpass_pairwise_correlations.csv", index=False)
    null_df.to_csv(table_dir / "rncp_lzc_bandpass_global_null_iterations.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(table_dir / "rncp_lzc_bandpass_global_null_summary.csv", index=False)
    fold_df.to_csv(table_dir / "rncp_lzc_bandpass_fold_reproducibility.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "Sleep-EDF SC",
                "scenario": "lzc_bandpass_0p5_45hz",
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
    ).to_csv(table_dir / "rncp_lzc_bandpass_sensitivity_summary.csv", index=False)

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
