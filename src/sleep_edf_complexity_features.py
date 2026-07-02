#!/usr/bin/env python3
"""optimized N3 complexity and spectral-parameterization features."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import antropy as ant
import mne
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from scipy import signal

try:
    from specparam import SpectralModel
except Exception:  # pragma: no cover - handled at runtime on the lab PC
    SpectralModel = None


EPOCH_SEC = 30.0
EPS = np.finfo(float).eps


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def clean_epoch(epoch_uv: np.ndarray) -> np.ndarray:
    finite = np.isfinite(epoch_uv)
    if not finite.any():
        return np.full(epoch_uv.shape, np.nan)
    median = float(np.nanmedian(epoch_uv))
    return np.nan_to_num(epoch_uv, nan=median, posinf=median, neginf=median)


def lzc_epoch(epoch_uv: np.ndarray) -> float:
    clean = clean_epoch(epoch_uv)
    if np.isnan(clean).all() or float(np.nanstd(clean)) <= EPS:
        return np.nan
    bits = (clean > np.median(clean)).astype(np.uint8)
    return float(ant.lziv_complexity(bits, normalize=True))


def permutation_entropy_epoch(epoch_uv: np.ndarray, order: int, delay: int) -> float:
    clean = clean_epoch(epoch_uv)
    if np.isnan(clean).all() or clean.size < order * delay or float(np.nanstd(clean)) <= EPS:
        return np.nan
    return float(ant.perm_entropy(clean, order=order, delay=delay, normalize=True))


def welch_psd(epochs_uv: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
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
    return freqs, psd


def get_specparam_param(model: object, component: str) -> float:
    try:
        value = model.get_params(component)
    except Exception:
        value = getattr(model, f"{component}_", np.nan)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size == 0:
            return np.nan
        return float(arr[-1])
    try:
        return float(value)
    except Exception:
        return np.nan


def get_specparam_metric(model: object, name: str) -> float:
    results = getattr(model, "results", None)
    if results is not None:
        try:
            value = results.get_metrics(name)
            return float(np.asarray(value, dtype=float).ravel()[-1])
        except Exception:
            pass
    try:
        return float(model.get_params(name))
    except Exception:
        return float(getattr(model, f"{name}_", np.nan))


def specparam_epoch(
    freqs: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float,
    max_n_peaks: int,
) -> tuple[float, float, float]:
    if SpectralModel is None:
        return np.nan, np.nan, np.nan
    mask = (freqs >= low) & (freqs <= high)
    freq_range = [float(low), float(high)]
    if mask.sum() < 8 or not np.isfinite(power[mask]).all():
        return np.nan, np.nan, np.nan
    if float(np.nanmax(power[mask])) <= EPS:
        return np.nan, np.nan, np.nan

    model = SpectralModel(
        peak_width_limits=[0.5, 12.0],
        max_n_peaks=max_n_peaks,
        min_peak_height=0.0,
        aperiodic_mode="fixed",
        verbose=False,
    )
    try:
        model.fit(freqs[mask], power[mask], freq_range)
    except TypeError:
        model.fit(freqs[mask], power[mask])
    except Exception:
        return np.nan, np.nan, np.nan

    exponent = get_specparam_param(model, "aperiodic")
    r_squared = get_specparam_metric(model, "r_squared")
    error = get_specparam_metric(model, "error")
    return exponent, r_squared, error


def compute_complexity_features(
    epochs_uv: np.ndarray,
    sfreq: float,
    meta: pd.DataFrame,
    perm_order: int,
    perm_delay: int,
    skip_specparam: bool,
    specparam_low: float,
    specparam_high: float,
    specparam_max_peaks: int,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "subject_id": meta["subject_id"].to_numpy(),
            "night_id": meta["night_id"].to_numpy(),
            "night": meta["night"].to_numpy(),
            "epoch_idx": meta["epoch_idx"].to_numpy(),
            "epoch_start_sec": meta["epoch_start_sec"].to_numpy(),
            "stage": meta["stage"].to_numpy(),
            "channel": meta["channel"].to_numpy(),
            "age": meta["age"].to_numpy(),
            "sex": meta["sex"].to_numpy(),
            "n3_bout_num": meta["n3_bout_num"].to_numpy(),
            "position_within_bout_fraction": meta["position_within_bout_fraction"].to_numpy(),
        }
    )
    out["lzc"] = [lzc_epoch(epoch) for epoch in epochs_uv]
    out["permutation_entropy"] = [
        permutation_entropy_epoch(epoch, order=perm_order, delay=perm_delay) for epoch in epochs_uv
    ]
    out["lzc_method"] = "antropy_lziv_median_binarized_normalized"
    out["permutation_entropy_method"] = f"antropy_perm_entropy_order{perm_order}_delay{perm_delay}_normalized"

    if skip_specparam:
        out["aperiodic_exponent_specparam"] = np.nan
        out["specparam_r_squared"] = np.nan
        out["specparam_error"] = np.nan
        out["aperiodic_exponent_method"] = "not_computed"
        return out

    freqs, psd = welch_psd(epochs_uv, sfreq)
    specs = [
        specparam_epoch(freqs, row, specparam_low, specparam_high, specparam_max_peaks)
        for row in psd
    ]
    out["aperiodic_exponent_specparam"] = [item[0] for item in specs]
    out["specparam_r_squared"] = [item[1] for item in specs]
    out["specparam_error"] = [item[2] for item in specs]
    out["aperiodic_exponent_method"] = f"specparam_fixed_{specparam_low:g}_{specparam_high:g}hz"
    return out


def process_recording(
    recording: pd.Series,
    dataset_root: Path,
    metadata: pd.DataFrame,
    out_dir: Path,
    stages: set[str],
    overwrite: bool,
    perm_order: int,
    perm_delay: int,
    skip_specparam: bool,
    specparam_low: float,
    specparam_high: float,
    specparam_max_peaks: int,
) -> tuple[str, str, float, int]:
    key = str(recording["recording_key"])
    out_path = out_dir / f"{key}_complexity_features.csv.gz"
    if out_path.exists() and not overwrite:
        return key, "skipped", 0.0, -1

    started = time.time()
    psg_path = dataset_root / recording["psg_relative_path"]
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(EPOCH_SEC * sfreq))

    meta_rec = metadata[(metadata["night_id"] == key) & (metadata["stage"].isin(stages))].copy()
    results = []
    for channel in sorted(meta_rec["channel"].dropna().unique()):
        if channel not in raw.ch_names:
            continue
        meta_ch = meta_rec[meta_rec["channel"] == channel].sort_values("epoch_idx").reset_index(drop=True)
        if meta_ch.empty:
            continue
        data = raw.get_data(picks=[channel], verbose="ERROR")[0] * 1_000_000.0
        epochs = []
        kept_rows = []
        for row_idx, row in meta_ch.iterrows():
            start = int(row["epoch_idx"]) * samples_per_epoch
            stop = start + samples_per_epoch
            if stop <= data.size:
                epochs.append(data[start:stop])
                kept_rows.append(row_idx)
        if not epochs:
            continue
        epochs_uv = np.vstack(epochs)
        kept_meta = meta_ch.loc[kept_rows].reset_index(drop=True)
        results.append(
            compute_complexity_features(
                epochs_uv=epochs_uv,
                sfreq=sfreq,
                meta=kept_meta,
                perm_order=perm_order,
                perm_delay=perm_delay,
                skip_specparam=skip_specparam,
                specparam_low=specparam_low,
                specparam_high=specparam_high,
                specparam_max_peaks=specparam_max_peaks,
            )
        )

    if not results:
        empty = pd.DataFrame()
        empty.to_csv(out_path, index=False, compression="gzip")
        return key, "ok_empty", time.time() - started, 0

    features = pd.concat(results, ignore_index=True)
    features.to_csv(out_path, index=False, compression="gzip")
    return key, "ok", time.time() - started, len(features)


def combine_outputs(feature_dir: Path, table_dir: Path) -> pd.DataFrame:
    files = sorted(feature_dir.glob("*_complexity_features.csv.gz"))
    frames = []
    for path in files:
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(table_dir / "sleep_edf_sc_complexity_features.csv.gz", index=False, compression="gzip")
    return combined


def write_summary(
    table_dir: Path,
    run_rows: list[dict],
    combined: pd.DataFrame,
    stages: set[str],
    skip_specparam: bool,
    perm_order: int,
    perm_delay: int,
) -> None:
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(table_dir / "sleep_edf_sc_complexity_feature_run_summary.csv", index=False)
    lines = [
        "# Complexity Feature Summary",
        "",
        f"- Stages included: {', '.join(sorted(stages))}",
        f"- Recordings considered: {len(run_df)}",
        f"- Recordings newly processed: {int(run_df['status'].isin(['ok', 'ok_empty']).sum())}",
        f"- Recordings reused from checkpoint: {int((run_df['status'] == 'skipped').sum())}",
        f"- Failed recordings: {int((run_df['status'] == 'error').sum())}",
        f"- Combined rows: {len(combined)}",
        f"- LZc implementation: antropy {package_version('antropy')}",
        f"- Permutation entropy implementation: antropy {package_version('antropy')}, order={perm_order}, delay={perm_delay}",
        f"- Specparam implementation: {'not computed' if skip_specparam else 'specparam ' + package_version('specparam')}",
        f"- NumPy: {package_version('numpy')}",
        f"- Numba: {package_version('numba')}",
    ]
    if not combined.empty:
        lines += [
            "",
            "## Non-null Feature Rows",
            "",
            f"- `lzc`: {int(combined['lzc'].notna().sum())}",
            f"- `permutation_entropy`: {int(combined['permutation_entropy'].notna().sum())}",
            f"- `aperiodic_exponent_specparam`: {int(combined['aperiodic_exponent_specparam'].notna().sum())}",
        ]
    processed_sec = run_df.loc[pd.to_numeric(run_df["elapsed_sec"], errors="coerce").notna(), "elapsed_sec"].sum()
    lines += ["", f"- Processing seconds for newly processed recordings: {float(processed_sec):.1f}"]
    if (run_df["status"] == "error").any():
        lines += ["", "## Errors", ""]
        for _, row in run_df[run_df["status"] == "error"].iterrows():
            lines.append(f"- {row['recording_key']}: {row['error']}")
    (table_dir / "complexity_feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stages", nargs="+", default=["N3"])
    parser.add_argument("--perm-order", type=int, default=5)
    parser.add_argument("--perm-delay", type=int, default=1)
    parser.add_argument("--skip-specparam", action="store_true")
    parser.add_argument("--specparam-low", type=float, default=1.0)
    parser.add_argument("--specparam-high", type=float, default=40.0)
    parser.add_argument("--specparam-max-peaks", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    project_root = cfg["project_data_root"]
    table_dir = project_root / "tables"
    feature_dir = project_root / "interim" / "complexity_by_recording"
    feature_dir.mkdir(parents=True, exist_ok=True)

    stages = {stage.upper() for stage in args.stages}
    recordings = pd.read_csv(table_dir / "sleep_edf_sc_recording_inventory.csv")
    metadata = pd.read_csv(table_dir / "sleep_edf_sc_epoch_metadata.csv.gz", low_memory=False)
    if args.recording_limit is not None:
        recordings = recordings.head(args.recording_limit).copy()

    run_rows = []
    for _, recording in recordings.iterrows():
        key = str(recording["recording_key"])
        try:
            rec_key, status, elapsed, rows = process_recording(
                recording=recording,
                dataset_root=dataset_root,
                metadata=metadata,
                out_dir=feature_dir,
                stages=stages,
                overwrite=args.overwrite,
                perm_order=args.perm_order,
                perm_delay=args.perm_delay,
                skip_specparam=args.skip_specparam,
                specparam_low=args.specparam_low,
                specparam_high=args.specparam_high,
                specparam_max_peaks=args.specparam_max_peaks,
            )
            print(f"{rec_key}: {status} rows={rows} elapsed={elapsed:.1f}s", flush=True)
            run_rows.append(
                {
                    "recording_key": rec_key,
                    "status": status,
                    "rows": rows,
                    "elapsed_sec": elapsed,
                    "error": "",
                }
            )
        except Exception as exc:
            print(f"{key}: error {type(exc).__name__}: {exc}", flush=True)
            run_rows.append(
                {
                    "recording_key": key,
                    "status": "error",
                    "rows": pd.NA,
                    "elapsed_sec": pd.NA,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    combined = combine_outputs(feature_dir, table_dir)
    write_summary(
        table_dir=table_dir,
        run_rows=run_rows,
        combined=combined,
        stages=stages,
        skip_specparam=args.skip_specparam,
        perm_order=args.perm_order,
        perm_delay=args.perm_delay,
    )
    print(f"Wrote combined complexity table: {table_dir / 'sleep_edf_sc_complexity_features.csv.gz'}")
    print(f"Combined rows: {len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
