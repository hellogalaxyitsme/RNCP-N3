#!/usr/bin/env python3
"""signal-derived controls and primary EEG features."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy import signal
from numpy.lib.stride_tricks import sliding_window_view


EPOCH_SEC = 30.0
EPS = np.finfo(float).eps
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def eeg_channels(channel_names: list[str]) -> list[str]:
    eeg = [ch for ch in channel_names if "eeg" in ch.lower()]
    if eeg:
        return eeg
    return [ch for ch in channel_names if any(token in ch.lower() for token in ["fpz", "pz", "cz", "oz"])]


def bandpower(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        return np.full(psd.shape[0], np.nan)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, low: float = 0.5, high: float = 45.0) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    values = psd[:, mask]
    denom = values.sum(axis=1, keepdims=True)
    probs = np.divide(values, denom + EPS)
    ent = -(probs * np.log(probs + EPS)).sum(axis=1)
    return ent / np.log(values.shape[1])


def aperiodic_exponent_fallback(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    # Fallback when specparam is unavailable: fit 1/f slope outside dominant slow-wave band.
    mask = (freqs >= 2.0) & (freqs <= 40.0)
    x = np.log10(freqs[mask] + EPS)
    y = np.log10(psd[:, mask] + EPS)
    x_centered = x - x.mean()
    denom = np.sum(x_centered**2)
    slopes = ((y - y.mean(axis=1, keepdims=True)) * x_centered).sum(axis=1) / denom
    return -slopes


def lz76_complexity_binary(bits: np.ndarray) -> int:
    n = int(bits.size)
    if n == 0:
        return 0
    i = 0
    k = 1
    ell = 1
    c = 1
    k_max = 1
    while True:
        if ell + k > n:
            c += 1
            break
        if bits[i + k - 1] == bits[ell + k - 1]:
            k += 1
            if ell + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == ell:
                c += 1
                ell += k_max
                if ell >= n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return c


def normalized_lzc(epoch: np.ndarray) -> float:
    clean = np.nan_to_num(epoch, nan=np.nanmedian(epoch))
    bits = (clean > np.median(clean)).astype(np.uint8)
    n = bits.size
    if n <= 1:
        return np.nan
    return float(lz76_complexity_binary(bits) * math.log2(n) / n)


def permutation_entropy_epoch(epoch: np.ndarray, order: int = 5, delay: int = 1) -> float:
    clean = np.nan_to_num(epoch, nan=np.nanmedian(epoch))
    if clean.size < order * delay:
        return np.nan
    if delay == 1:
        windows = sliding_window_view(clean, order)
    else:
        windows = np.stack([clean[i : clean.size - (order - 1) * delay + i : delay] for i in range(order)], axis=1)
    patterns = np.argsort(windows, axis=1, kind="mergesort")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / counts.sum()
    ent = -(probs * np.log(probs + EPS)).sum()
    return float(ent / np.log(math.factorial(order)))


def slow_wave_metrics(epochs_uv: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    sos = signal.butter(4, [0.5, 2.0], btype="bandpass", fs=sfreq, output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, epochs_uv, axis=1)
    except ValueError:
        filtered = signal.sosfilt(sos, epochs_uv, axis=1)

    min_distance = max(1, int(0.25 * sfreq))
    local_half_window = max(1, int(0.5 * sfreq))
    density = np.zeros(epochs_uv.shape[0], dtype=float)
    occupancy = np.zeros(epochs_uv.shape[0], dtype=float)

    for i, epoch in enumerate(filtered):
        troughs, props = signal.find_peaks(-epoch, height=37.5, distance=min_distance)
        valid = 0
        occupied_samples = 0
        for trough in troughs:
            lo = max(0, trough - local_half_window)
            hi = min(epoch.size, trough + local_half_window + 1)
            segment = epoch[lo:hi]
            if segment.size == 0:
                continue
            if float(segment.max() - segment.min()) >= 75.0:
                valid += 1
                occupied_samples += hi - lo
        density[i] = valid / EPOCH_SEC
        occupancy[i] = min(1.0, occupied_samples / (epoch.size + EPS))
    return density, occupancy


def signal_artifacts(epochs_uv: np.ndarray) -> tuple[np.ndarray, list[str]]:
    max_abs = np.nanmax(np.abs(epochs_uv), axis=1)
    ptp = np.nanmax(epochs_uv, axis=1) - np.nanmin(epochs_uv, axis=1)
    std = np.nanstd(epochs_uv, axis=1)
    has_nan = np.isnan(epochs_uv).any(axis=1)

    flags = (max_abs > 500.0) | (ptp > 1000.0) | (std < 0.1) | has_nan
    reasons = []
    for a, p, s, n in zip(max_abs, ptp, std, has_nan):
        parts = []
        if a > 500.0:
            parts.append("abs_gt_500uv")
        if p > 1000.0:
            parts.append("ptp_gt_1000uv")
        if s < 0.1:
            parts.append("flat_or_low_variance")
        if n:
            parts.append("nan")
        reasons.append("|".join(parts))
    return flags, reasons


def compute_channel_features(
    epochs_uv: np.ndarray,
    sfreq: float,
    meta: pd.DataFrame,
    include_complexity: bool,
) -> pd.DataFrame:
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

    delta = bandpower(freqs, psd, 0.5, 4.0)
    total = bandpower(freqs, psd, 0.5, min(45.0, sfreq / 2.0 - 0.1))
    relative_delta = np.divide(delta, total + EPS)
    spec_ent = spectral_entropy(freqs, psd, 0.5, min(45.0, sfreq / 2.0 - 0.1))
    aperiodic = aperiodic_exponent_fallback(freqs, psd)
    sw_density, sw_occupancy = slow_wave_metrics(epochs_uv, sfreq)
    artifact_flags, artifact_reasons = signal_artifacts(epochs_uv)

    after_sleep = meta["after_sleep_onset"].fillna(False).astype(bool).to_numpy()
    cumulative_swa = np.cumsum(np.where(after_sleep, delta, 0.0))
    cumulative_swa = np.where(after_sleep, cumulative_swa, 0.0)

    out = pd.DataFrame(
        {
            "subject_id": meta["subject_id"].to_numpy(),
            "night_id": meta["night_id"].to_numpy(),
            "epoch_idx": meta["epoch_idx"].to_numpy(),
            "channel": meta["channel"].to_numpy(),
            "delta_power": delta,
            "total_power_0p5_45": total,
            "relative_delta_power": relative_delta,
            "slow_wave_density": sw_density,
            "slow_wave_occupancy": sw_occupancy,
            "cumulative_swa": cumulative_swa,
            "spectral_entropy": spec_ent,
            "aperiodic_exponent": aperiodic,
            "aperiodic_exponent_method": "fallback_loglog_2_40hz",
            "artifact_signal_flag": artifact_flags,
            "artifact_signal_reason": artifact_reasons,
        }
    )

    if include_complexity:
        out["lzc"] = [normalized_lzc(epoch) for epoch in epochs_uv]
        out["permutation_entropy"] = [permutation_entropy_epoch(epoch, order=5, delay=1) for epoch in epochs_uv]
    else:
        out["lzc"] = np.nan
        out["permutation_entropy"] = np.nan

    return out


def process_recording(
    recording: pd.Series,
    dataset_root: Path,
    metadata: pd.DataFrame,
    out_dir: Path,
    include_complexity: bool,
    overwrite: bool,
) -> tuple[str, str, float, int]:
    key = str(recording["recording_key"])
    out_path = out_dir / f"{key}_signal_features.csv.gz"
    if out_path.exists() and not overwrite:
        return key, "skipped", 0.0, -1

    started = time.time()
    psg_path = dataset_root / recording["psg_relative_path"]
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    channels = eeg_channels(raw.ch_names)
    sfreq = float(raw.info["sfreq"])
    samples_per_epoch = int(round(EPOCH_SEC * sfreq))

    meta_rec = metadata[metadata["night_id"] == key].copy()
    results = []
    for channel in channels:
        meta_ch = meta_rec[meta_rec["channel"] == channel].sort_values("epoch_idx").reset_index(drop=True)
        if meta_ch.empty:
            continue
        data = raw.get_data(picks=[channel], verbose="ERROR")[0] * 1_000_000.0
        n_epochs = min(len(meta_ch), data.size // samples_per_epoch)
        trimmed = data[: n_epochs * samples_per_epoch]
        epochs_uv = trimmed.reshape(n_epochs, samples_per_epoch)
        meta_ch = meta_ch.iloc[:n_epochs].reset_index(drop=True)
        results.append(compute_channel_features(epochs_uv, sfreq, meta_ch, include_complexity))

    if not results:
        raise RuntimeError(f"No EEG channels matched metadata for {key}")

    features = pd.concat(results, ignore_index=True)
    features.to_csv(out_path, index=False, compression="gzip")
    elapsed = time.time() - started
    return key, "ok", elapsed, len(features)


def combine_outputs(feature_dir: Path, table_dir: Path) -> pd.DataFrame:
    files = sorted(feature_dir.glob("*_signal_features.csv.gz"))
    frames = [pd.read_csv(path) for path in files]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(table_dir / "sleep_edf_sc_signal_features.csv.gz", index=False, compression="gzip")
    return combined


def write_summary(table_dir: Path, run_rows: list[dict], combined: pd.DataFrame, include_complexity: bool) -> None:
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(table_dir / "sleep_edf_sc_signal_feature_run_summary.csv", index=False)
    ok = run_df[run_df["status"].isin(["ok", "skipped"])]
    lines = [
        "# Signal Feature Summary",
        "",
        f"- Recordings considered: {len(run_df)}",
        f"- Recordings newly processed: {int((run_df['status'] == 'ok').sum())}",
        f"- Recordings reused from checkpoint: {int((run_df['status'] == 'skipped').sum())}",
        f"- Failed recordings: {int((run_df['status'] == 'error').sum())}",
        f"- Combined rows: {len(combined)}",
        f"- Include LZc and permutation entropy: {include_complexity}",
        f"- Signal artifact flagged rows: {int(combined['artifact_signal_flag'].sum()) if not combined.empty else 0}",
        "",
        "## Feature Columns",
        "",
        "- `relative_delta_power`",
        "- `delta_power`",
        "- `total_power_0p5_45`",
        "- `slow_wave_density`",
        "- `slow_wave_occupancy`",
        "- `cumulative_swa`",
        "- `spectral_entropy`",
        "- `aperiodic_exponent`",
        "- `lzc`",
        "- `permutation_entropy`",
        "",
        "Aperiodic exponent currently uses fallback log-log regression over 2-40 Hz because `specparam` is not installed.",
    ]
    if not ok.empty:
        processed_sec = run_df.loc[run_df["elapsed_sec"].notna(), "elapsed_sec"].sum()
        lines += ["", f"- Processing seconds for newly processed recordings: {processed_sec:.1f}"]
    if (run_df["status"] == "error").any():
        lines += ["", "## Errors", ""]
        for _, row in run_df[run_df["status"] == "error"].iterrows():
            lines.append(f"- {row['recording_key']}: {row['error']}")
    (table_dir / "signal_feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recording-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-complexity", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    project_root = cfg["project_data_root"]
    table_dir = project_root / "tables"
    feature_dir = project_root / "interim" / "features_by_recording"
    feature_dir.mkdir(parents=True, exist_ok=True)

    recordings = pd.read_csv(table_dir / "sleep_edf_sc_recording_inventory.csv")
    metadata = pd.read_csv(table_dir / "sleep_edf_sc_epoch_metadata.csv.gz")
    if args.recording_limit is not None:
        recordings = recordings.head(args.recording_limit).copy()

    include_complexity = not args.skip_complexity
    run_rows = []
    for i, recording in recordings.iterrows():
        key = str(recording["recording_key"])
        try:
            rec_key, status, elapsed, rows = process_recording(
                recording=recording,
                dataset_root=dataset_root,
                metadata=metadata,
                out_dir=feature_dir,
                include_complexity=include_complexity,
                overwrite=args.overwrite,
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
    write_summary(table_dir, run_rows, combined, include_complexity)
    print(f"Wrote combined feature table: {table_dir / 'sleep_edf_sc_signal_features.csv.gz'}")
    print(f"Combined rows: {len(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
