#!/usr/bin/env python3
"""Sleep-EDF 30-second epoch metadata builder."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_right
from pathlib import Path

import mne
import pandas as pd


EPOCH_SEC = 30.0
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def normalize_stage(description: str) -> tuple[str, str]:
    desc = str(description).strip()
    low = desc.lower()
    if low in {"sleep stage w", "stage w", "wake", "w"}:
        return "W", "W"
    if low in {"sleep stage 1", "stage 1", "n1", "1"}:
        return "N1", "N1"
    if low in {"sleep stage 2", "stage 2", "n2", "2"}:
        return "N2", "N2"
    if low in {"sleep stage 3", "stage 3", "3"}:
        return "N3", "S3"
    if low in {"sleep stage 4", "stage 4", "4"}:
        return "N3", "S4"
    if low in {"sleep stage r", "stage r", "rem", "r"}:
        return "REM", "REM"
    if "movement" in low:
        return "MOVEMENT", "MOVEMENT"
    return "UNKNOWN", desc


def eeg_channels(channel_names: list[str]) -> list[str]:
    eeg = [ch for ch in channel_names if "eeg" in ch.lower()]
    if eeg:
        return eeg
    return [ch for ch in channel_names if any(token in ch.lower() for token in ["fpz", "pz", "cz", "oz"])]


def stage_at_midpoint(midpoint: float, intervals: list[tuple[float, float, str, str]], starts: list[float]) -> tuple[str, str]:
    idx = bisect_right(starts, midpoint) - 1
    if idx < 0:
        return "UNKNOWN", "UNKNOWN"
    start, end, stage, original = intervals[idx]
    if start <= midpoint < end:
        return stage, original
    return "UNKNOWN", "UNKNOWN"


def build_intervals(annotations: mne.Annotations, raw_duration_sec: float) -> list[tuple[float, float, str, str]]:
    intervals = []
    for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
        start = max(float(onset), 0.0)
        end = min(float(onset) + float(duration), raw_duration_sec)
        if end <= start:
            continue
        stage, original = normalize_stage(str(desc))
        intervals.append((start, end, stage, original))
    intervals.sort(key=lambda x: x[0])
    return intervals


def assign_n3_bouts(epoch_df: pd.DataFrame) -> pd.DataFrame:
    epoch_df = epoch_df.copy()
    n = len(epoch_df)
    epoch_df["n3_bout_num"] = pd.NA
    epoch_df["position_within_bout"] = pd.NA
    epoch_df["position_within_bout_fraction"] = pd.NA
    epoch_df["n3_bout_duration_epochs"] = pd.NA
    epoch_df["n3_bout_duration_min"] = pd.NA

    bout_num = 0
    i = 0
    stages = epoch_df["stage"].tolist()
    while i < n:
        if stages[i] != "N3":
            i += 1
            continue
        bout_num += 1
        start = i
        while i < n and stages[i] == "N3":
            i += 1
        end = i
        length = end - start
        idx = epoch_df.index[start:end]
        epoch_df.loc[idx, "n3_bout_num"] = bout_num
        epoch_df.loc[idx, "position_within_bout"] = list(range(1, length + 1))
        epoch_df.loc[idx, "position_within_bout_fraction"] = [pos / length for pos in range(1, length + 1)]
        epoch_df.loc[idx, "n3_bout_duration_epochs"] = length
        epoch_df.loc[idx, "n3_bout_duration_min"] = length * EPOCH_SEC / 60.0

    return epoch_df


def build_recording_epochs(recording: pd.Series, dataset_root: Path) -> tuple[pd.DataFrame, dict]:
    psg_path = dataset_root / recording["psg_relative_path"]
    hyp_path = dataset_root / recording["hypnogram_relative_path"]

    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
    annotations = mne.read_annotations(hyp_path)

    raw_duration_sec = float(raw.n_times) / float(raw.info["sfreq"])
    n_epochs = int(raw_duration_sec // EPOCH_SEC)
    channels = eeg_channels(raw.ch_names)
    intervals = build_intervals(annotations, raw_duration_sec)
    starts = [item[0] for item in intervals]

    base_rows = []
    sleep_onset_sec = None
    for epoch_idx in range(n_epochs):
        epoch_start_sec = epoch_idx * EPOCH_SEC
        midpoint = epoch_start_sec + (EPOCH_SEC / 2.0)
        stage, original = stage_at_midpoint(midpoint, intervals, starts)
        if sleep_onset_sec is None and stage in SLEEP_STAGES:
            sleep_onset_sec = epoch_start_sec
        base_rows.append(
            {
                "subject_id": recording["subject_id"],
                "night_id": recording["recording_key"],
                "night": int(recording["night"]),
                "epoch_idx": epoch_idx,
                "epoch_start_sec": epoch_start_sec,
                "epoch_start_min": epoch_start_sec / 60.0,
                "stage": stage,
                "stage_original": original,
            }
        )

    base = pd.DataFrame(base_rows)
    if sleep_onset_sec is None:
        base["sleep_onset_min"] = pd.NA
        base["time_since_sleep_onset"] = pd.NA
        base["after_sleep_onset"] = False
    else:
        base["sleep_onset_min"] = sleep_onset_sec / 60.0
        base["time_since_sleep_onset"] = (base["epoch_start_sec"] - sleep_onset_sec) / 60.0
        base["after_sleep_onset"] = base["epoch_start_sec"] >= sleep_onset_sec

    base = assign_n3_bouts(base)
    base["stage_is_sleep"] = base["stage"].isin(SLEEP_STAGES)
    base["stage_is_n3"] = base["stage"] == "N3"
    base["artifact_flag"] = base["stage"].isin(["MOVEMENT", "UNKNOWN"])
    base["artifact_reason"] = base["stage"].map({"MOVEMENT": "movement_annotation", "UNKNOWN": "unknown_stage"}).fillna("")
    base["relative_delta_power"] = pd.NA
    base["slow_wave_density"] = pd.NA
    base["cumulative_swa"] = pd.NA

    repeated = []
    for channel in channels:
        part = base.copy()
        part["channel"] = channel
        repeated.append(part)
    epochs = pd.concat(repeated, ignore_index=True) if repeated else base.assign(channel=pd.NA)

    summary = {
        "subject_id": recording["subject_id"],
        "night_id": recording["recording_key"],
        "n_epochs": n_epochs,
        "n_channels": len(channels),
        "channels": "|".join(channels),
        "sleep_onset_min": pd.NA if sleep_onset_sec is None else sleep_onset_sec / 60.0,
        "n3_epochs": int((base["stage"] == "N3").sum()),
        "n3_min": float((base["stage"] == "N3").sum() * EPOCH_SEC / 60.0),
        "n3_bouts": int(base["n3_bout_num"].dropna().nunique()),
        "movement_epochs": int((base["stage"] == "MOVEMENT").sum()),
        "unknown_epochs": int((base["stage"] == "UNKNOWN").sum()),
        "raw_duration_min": raw_duration_sec / 60.0,
        "status": "ok",
        "error": "",
    }
    return epochs, summary


def write_summary(table_dir: Path, recording_summary: pd.DataFrame, epoch_rows: int) -> None:
    ok = recording_summary[recording_summary["status"] == "ok"].copy()
    lines = [
        "# Sleep-EDF SC Epoch Metadata Summary",
        "",
        f"- Successfully processed recordings: {len(ok)}",
        f"- Failed recordings: {int((recording_summary['status'] != 'ok').sum())}",
        f"- Epoch metadata rows, channel-expanded: {epoch_rows}",
        f"- Recording-level 30-second epochs: {int(ok['n_epochs'].sum())}",
        f"- EEG channels per recording, median: {ok['n_channels'].median():.0f}",
        f"- N3 recording-level epochs: {int(ok['n3_epochs'].sum())}",
        f"- N3 recording-level minutes: {ok['n3_min'].sum():.1f}",
        f"- N3 bouts total: {int(ok['n3_bouts'].sum())}",
        "",
        "## Outputs",
        "",
        "- `sleep_edf_sc_epoch_metadata.csv.gz`",
        "- `sleep_edf_sc_epoch_recording_summary.csv`",
        "- `epoch_metadata_summary.md`",
        "",
        "Signal-derived fields are placeholders in this analysis:",
        "",
        "- `relative_delta_power`",
        "- `slow_wave_density`",
        "- `cumulative_swa`",
    ]
    if (recording_summary["status"] != "ok").any():
        lines += ["", "## Errors", ""]
        for _, row in recording_summary[recording_summary["status"] != "ok"].iterrows():
            lines.append(f"- {row['night_id']}: {row['error']}")
    (table_dir / "epoch_metadata_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only-usable-n3", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    recordings = pd.read_csv(table_dir / "sleep_edf_sc_recording_inventory.csv")
    subjects = pd.read_csv(table_dir / "sleep_edf_sc_subject_inventory.csv")
    cohort = pd.read_csv(table_dir / "sleep_edf_sc_cohort_table.csv")

    if args.only_usable_n3:
        usable_keys = set(cohort.loc[cohort["usable"], "night_id"])
        recordings = recordings[recordings["recording_key"].isin(usable_keys)].copy()

    subject_cols = ["subject_id", "age", "sex"]
    recordings = recordings.merge(subjects[subject_cols], on="subject_id", how="left")

    all_epochs = []
    summaries = []
    for _, recording in recordings.iterrows():
        try:
            epochs, summary = build_recording_epochs(recording, dataset_root)
            epochs["age"] = recording["age"]
            epochs["sex"] = recording["sex"]
            all_epochs.append(epochs)
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    "subject_id": recording.get("subject_id"),
                    "night_id": recording.get("recording_key"),
                    "n_epochs": pd.NA,
                    "n_channels": pd.NA,
                    "channels": "",
                    "sleep_onset_min": pd.NA,
                    "n3_epochs": pd.NA,
                    "n3_min": pd.NA,
                    "n3_bouts": pd.NA,
                    "movement_epochs": pd.NA,
                    "unknown_epochs": pd.NA,
                    "raw_duration_min": pd.NA,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    epoch_metadata = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    recording_summary = pd.DataFrame(summaries)

    out_name = "sleep_edf_sc_epoch_metadata.csv.gz"
    if args.only_usable_n3:
        out_name = "sleep_edf_sc_epoch_metadata_usable_n3.csv.gz"

    epoch_metadata.to_csv(table_dir / out_name, index=False, compression="gzip")
    recording_summary.to_csv(table_dir / "sleep_edf_sc_epoch_recording_summary.csv", index=False)
    write_summary(table_dir, recording_summary, len(epoch_metadata))

    print(f"Wrote epoch metadata to {table_dir / out_name}")
    print(f"Rows: {len(epoch_metadata)}")
    print(f"Processed recordings: {int((recording_summary['status'] == 'ok').sum())}/{len(recording_summary)}")
    print(f"N3 minutes: {recording_summary['n3_min'].sum():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
