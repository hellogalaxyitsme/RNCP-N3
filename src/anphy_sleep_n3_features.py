#!/usr/bin/env python3
"""ANPHY-Sleep N3 signal, complexity, and specparam feature pass."""

from __future__ import annotations

import argparse
import os
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from anphy_sleep_common import EPOCH_SEC, discover_recordings, load_config, load_position_channels
from sleep_edf_complexity_features import compute_complexity_features
from sleep_edf_signal_features import compute_channel_features


# Avoid oversubscribing BLAS/OpenMP threads when channel-level workers are used.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def norm_channel(name: str) -> str:
    text = name.lower().strip()
    for token in ["eeg ", "eeg-", "-ref", "_ref", " ref", " "]:
        text = text.replace(token, "")
    return text


def match_eeg_channels(raw_names: list[str], position_channels: list[str]) -> list[str]:
    raw_by_norm = {norm_channel(ch): ch for ch in raw_names}
    matched = []
    for ch in position_channels:
        raw = raw_by_norm.get(norm_channel(ch))
        if raw is not None:
            matched.append(raw)
    if matched:
        return matched

    exclude_tokens = ["eog", "emg", "ecg", "ekg", "airflow", "snore", "thor", "abdo", "spo2", "pulse"]
    return [ch for ch in raw_names if not any(token in ch.lower() for token in exclude_tokens)]


def compute_channel_task(payload: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    channel = payload["channel"]
    epochs_uv = payload["epochs_uv"]
    sfreq = payload["sfreq"]
    meta = payload["meta"]
    signal_df = compute_channel_features(
        epochs_uv=epochs_uv,
        sfreq=sfreq,
        meta=meta,
        include_complexity=False,
    )
    complexity_df = compute_complexity_features(
        epochs_uv=epochs_uv,
        sfreq=sfreq,
        meta=meta,
        perm_order=payload["perm_order"],
        perm_delay=payload["perm_delay"],
        skip_specparam=payload["skip_specparam"],
        specparam_low=payload["specparam_low"],
        specparam_high=payload["specparam_high"],
        specparam_max_peaks=payload["specparam_max_peaks"],
    )
    return channel, signal_df, complexity_df


def extract_edf_to_cache(recording, cache_root: Path, overwrite_cache: bool) -> Path:
    out_dir = cache_root / "anphy_edf_cache" / recording.subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(recording.edf_member).name

    if out_path.exists() and out_path.stat().st_size == recording.edf_size_bytes and not overwrite_cache:
        return out_path
    if out_path.exists():
        out_path.unlink()

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()
    with zipfile.ZipFile(recording.zip_path) as zf:
        with zf.open(recording.edf_member) as src, tmp_path.open("wb") as dst:
            while True:
                chunk = src.read(16 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    if tmp_path.stat().st_size != recording.edf_size_bytes:
        raise IOError(
            f"Extracted EDF size mismatch for {recording.subject_id}: "
            f"{tmp_path.stat().st_size} != {recording.edf_size_bytes}"
        )
    tmp_path.replace(out_path)
    return out_path


def process_recording(
    recording,
    metadata: pd.DataFrame,
    position_channels: list[str],
    cache_root: Path,
    signal_out_dir: Path,
    complexity_out_dir: Path,
    stages: set[str],
    overwrite: bool,
    overwrite_cache: bool,
    keep_extracted: bool,
    skip_specparam: bool,
    perm_order: int,
    perm_delay: int,
    specparam_low: float,
    specparam_high: float,
    specparam_max_peaks: int,
    channel_workers: int,
) -> dict:
    signal_path = signal_out_dir / f"{recording.night_id}_signal_features.csv.gz"
    complexity_path = complexity_out_dir / f"{recording.night_id}_complexity_features.csv.gz"
    if signal_path.exists() and complexity_path.exists() and not overwrite:
        return {
            "subject_id": recording.subject_id,
            "night_id": recording.night_id,
            "status": "skipped",
            "rows": -1,
            "seconds": 0.0,
            "error": "",
        }

    started = time.time()
    edf_path = extract_edf_to_cache(recording, cache_root, overwrite_cache=overwrite_cache)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        sfreq = float(raw.info["sfreq"])
        samples_per_epoch = int(round(EPOCH_SEC * sfreq))
        channels = match_eeg_channels(raw.ch_names, position_channels)
        print(
            f"{recording.night_id}: start channels={len(channels)} sfreq={sfreq:g}Hz",
            flush=True,
        )

        meta_rec = metadata[(metadata["night_id"] == recording.night_id) & (metadata["stage"].isin(stages))].copy()
        tasks = []
        for channel in channels:
            meta_ch = meta_rec[meta_rec["channel"].map(norm_channel) == norm_channel(channel)].copy()
            if meta_ch.empty:
                continue
            meta_ch = meta_ch.sort_values("epoch_idx").reset_index(drop=True)
            data_uv = raw.get_data(picks=[channel], verbose="ERROR")[0] * 1_000_000.0
            epochs = []
            kept = []
            for row_idx, row in meta_ch.iterrows():
                start = int(row["epoch_idx"]) * samples_per_epoch
                stop = start + samples_per_epoch
                if stop <= data_uv.size:
                    epochs.append(data_uv[start:stop])
                    kept.append(row_idx)
            if not epochs:
                continue

            epochs_uv = np.vstack(epochs)
            kept_meta = meta_ch.loc[kept].reset_index(drop=True)
            kept_meta["channel"] = channel
            tasks.append(
                {
                    "channel": channel,
                    "epochs_uv": epochs_uv,
                    "sfreq": sfreq,
                    "meta": kept_meta,
                    "perm_order": perm_order,
                    "perm_delay": perm_delay,
                    "skip_specparam": skip_specparam,
                    "specparam_low": specparam_low,
                    "specparam_high": specparam_high,
                    "specparam_max_peaks": specparam_max_peaks,
                }
            )

        signal_parts = []
        complexity_parts = []
        if channel_workers <= 1:
            for idx, task in enumerate(tasks, start=1):
                channel, signal_df, complexity_df = compute_channel_task(task)
                signal_parts.append(signal_df)
                complexity_parts.append(complexity_df)
                print(f"{recording.night_id}: channel {idx}/{len(tasks)} done {channel}", flush=True)
        else:
            workers = min(channel_workers, len(tasks))
            print(f"{recording.night_id}: using channel_workers={workers}", flush=True)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_channel = {executor.submit(compute_channel_task, task): task["channel"] for task in tasks}
                for idx, future in enumerate(as_completed(future_to_channel), start=1):
                    channel, signal_df, complexity_df = future.result()
                    signal_parts.append(signal_df)
                    complexity_parts.append(complexity_df)
                    print(f"{recording.night_id}: channel {idx}/{len(tasks)} done {channel}", flush=True)

        signal = pd.concat(signal_parts, ignore_index=True) if signal_parts else pd.DataFrame()
        complexity = pd.concat(complexity_parts, ignore_index=True) if complexity_parts else pd.DataFrame()
        signal.to_csv(signal_path, index=False, compression="gzip")
        complexity.to_csv(complexity_path, index=False, compression="gzip")
        rows = max(len(signal), len(complexity))
        return {
            "subject_id": recording.subject_id,
            "night_id": recording.night_id,
            "status": "ok",
            "rows": rows,
            "seconds": time.time() - started,
            "sfreq_hz": sfreq,
            "raw_channel_count": len(raw.ch_names),
            "matched_eeg_channel_count": len(channels),
            "error": "",
        }
    finally:
        if not keep_extracted and edf_path.exists():
            edf_path.unlink()


def combine_outputs(out_dir: Path, pattern: str, combined_path: Path) -> int:
    paths = sorted(out_dir.glob(pattern))
    frames = [pd.read_csv(path, low_memory=False) for path in paths if path.stat().st_size > 0]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(combined_path, index=False, compression="gzip")
    return len(combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["N3"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--skip-specparam", action="store_true")
    parser.add_argument("--max-recordings", type=int, default=None)
    parser.add_argument("--perm-order", type=int, default=5)
    parser.add_argument("--perm-delay", type=int, default=1)
    parser.add_argument("--specparam-low", type=float, default=2.0)
    parser.add_argument("--specparam-high", type=float, default=40.0)
    parser.add_argument("--specparam-max-peaks", type=int, default=6)
    parser.add_argument("--channel-workers", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    cache_root = cfg["cache_root"]
    cache_root.mkdir(parents=True, exist_ok=True)

    _, recordings = discover_recordings(dataset_root)
    cohort = pd.read_csv(table_dir / "anphy_sleep_cohort_table.csv")
    usable = set(cohort.loc[cohort["usable"].astype(str).str.lower().isin(["true", "1"]), "night_id"])
    recordings = [recording for recording in recordings if recording.night_id in usable]
    if args.max_recordings is not None:
        recordings = recordings[: args.max_recordings]

    metadata = pd.read_csv(table_dir / "anphy_sleep_epoch_metadata.csv.gz", low_memory=False)
    position_channels = load_position_channels(dataset_root)
    signal_out_dir = table_dir / "anphy_recording_signal_features"
    complexity_out_dir = table_dir / "anphy_recording_complexity_features"
    signal_out_dir.mkdir(parents=True, exist_ok=True)
    complexity_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for recording in recordings:
        try:
            result = process_recording(
                recording=recording,
                metadata=metadata,
                position_channels=position_channels,
                cache_root=cache_root,
                signal_out_dir=signal_out_dir,
                complexity_out_dir=complexity_out_dir,
                stages=set(args.stages),
                overwrite=args.overwrite,
                overwrite_cache=args.overwrite_cache,
                keep_extracted=args.keep_extracted,
                skip_specparam=args.skip_specparam,
                perm_order=args.perm_order,
                perm_delay=args.perm_delay,
                specparam_low=args.specparam_low,
                specparam_high=args.specparam_high,
                specparam_max_peaks=args.specparam_max_peaks,
                channel_workers=max(1, args.channel_workers),
            )
        except Exception as exc:
            result = {
                "subject_id": recording.subject_id,
                "night_id": recording.night_id,
                "status": "error",
                "rows": 0,
                "seconds": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(f"{result['night_id']}: {result['status']} rows={result['rows']} seconds={result['seconds']:.1f}", flush=True)
        rows.append(result)

    run_summary = pd.DataFrame(rows)
    run_summary.to_csv(table_dir / "anphy_sleep_n3_feature_run_summary.csv", index=False)
    signal_rows = combine_outputs(
        signal_out_dir,
        "*_signal_features.csv.gz",
        table_dir / "anphy_sleep_signal_features.csv.gz",
    )
    complexity_rows = combine_outputs(
        complexity_out_dir,
        "*_complexity_features.csv.gz",
        table_dir / "anphy_sleep_complexity_features.csv.gz",
    )

    lines = [
        "# ANPHY-Sleep N3 Feature Summary",
        "",
        f"- Recordings attempted: {len(run_summary)}",
        f"- OK recordings: {int((run_summary['status'] == 'ok').sum()) if not run_summary.empty else 0}",
        f"- Skipped recordings: {int((run_summary['status'] == 'skipped').sum()) if not run_summary.empty else 0}",
        f"- Error recordings: {int((run_summary['status'] == 'error').sum()) if not run_summary.empty else 0}",
        f"- Combined signal rows: {signal_rows}",
        f"- Combined complexity rows: {complexity_rows}",
        f"- Stages: {'|'.join(args.stages)}",
        f"- Specparam skipped: {args.skip_specparam}",
    ]
    (table_dir / "anphy_feature_n3_feature_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Combined signal rows={signal_rows}")
    print(f"Combined complexity rows={complexity_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
