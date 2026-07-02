#!/usr/bin/env python3
"""ANPHY-Sleep channel-expanded 30-second epoch metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from anphy_sleep_common import (
    EPOCH_SEC,
    SLEEP_STAGES,
    assign_n3_bouts,
    discover_recordings,
    load_config,
    load_position_channels,
    read_recording_annotations,
)


def build_base_epochs(recording, annotations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid = annotations.dropna(subset=["onset_sec", "duration_sec"]).copy()
    valid = valid[valid["duration_sec"] > 0].sort_values("onset_sec").reset_index(drop=True)
    end_sec = float((valid["onset_sec"] + valid["duration_sec"]).max()) if not valid.empty else 0.0
    n_epochs = int(end_sec // EPOCH_SEC)

    stage_by_epoch = {}
    original_by_epoch = {}
    for _, row in valid.iterrows():
        start_epoch = int(float(row["onset_sec"]) // EPOCH_SEC)
        n = int(round(float(row["duration_sec"]) / EPOCH_SEC))
        for offset in range(max(0, n)):
            idx = start_epoch + offset
            stage_by_epoch[idx] = row["stage"]
            original_by_epoch[idx] = row["stage_original"]

    sleep_onset_sec = None
    rows = []
    for epoch_idx in range(n_epochs):
        start_sec = epoch_idx * EPOCH_SEC
        stage = stage_by_epoch.get(epoch_idx, "UNKNOWN")
        if sleep_onset_sec is None and stage in SLEEP_STAGES:
            sleep_onset_sec = start_sec
        rows.append(
            {
                "subject_id": recording.subject_id,
                "night_id": recording.night_id,
                "night": recording.night,
                "epoch_idx": epoch_idx,
                "epoch_start_sec": start_sec,
                "epoch_start_min": start_sec / 60.0,
                "stage": stage,
                "stage_original": original_by_epoch.get(epoch_idx, "UNKNOWN"),
            }
        )

    base = pd.DataFrame(rows)
    if base.empty:
        base = pd.DataFrame(
            columns=[
                "subject_id",
                "night_id",
                "night",
                "epoch_idx",
                "epoch_start_sec",
                "epoch_start_min",
                "stage",
                "stage_original",
            ]
        )

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

    summary = {
        "subject_id": recording.subject_id,
        "night_id": recording.night_id,
        "n_epochs": n_epochs,
        "sleep_onset_min": pd.NA if sleep_onset_sec is None else sleep_onset_sec / 60.0,
        "n3_epochs": int((base["stage"] == "N3").sum()),
        "n3_min": float((base["stage"] == "N3").sum() * EPOCH_SEC / 60.0),
        "n3_bouts": int(base["n3_bout_num"].dropna().nunique()) if not base.empty else 0,
        "unknown_epochs": int((base["stage"] == "UNKNOWN").sum()) if not base.empty else 0,
        "annotation_duration_min": end_sec / 60.0,
        "status": "ok",
        "error": "",
    }
    return base, summary


def write_summary(table_dir: Path, recording_summary: pd.DataFrame, epoch_rows: int, n_channels: int) -> None:
    ok = recording_summary[recording_summary["status"] == "ok"].copy()
    lines = [
        "# ANPHY-Sleep Epoch Metadata Summary",
        "",
        f"- Successfully processed recordings: {len(ok)}",
        f"- Failed recordings: {int((recording_summary['status'] != 'ok').sum())}",
        f"- Position-derived EEG channels per recording: {n_channels}",
        f"- Epoch metadata rows, channel-expanded: {epoch_rows}",
        f"- Recording-level 30-second epochs: {int(ok['n_epochs'].sum())}",
        f"- N3 recording-level epochs: {int(ok['n3_epochs'].sum())}",
        f"- N3 recording-level minutes: {ok['n3_min'].sum():.1f}",
        f"- N3 bouts total: {int(ok['n3_bouts'].sum())}",
        "",
        "## Outputs",
        "",
        "- `anphy_sleep_epoch_metadata.csv.gz`",
        "- `anphy_sleep_epoch_recording_summary.csv`",
        "- `anphy_epoch_metadata_summary.md`",
    ]
    if (recording_summary["status"] != "ok").any():
        lines += ["", "## Errors", ""]
        for _, row in recording_summary[recording_summary["status"] != "ok"].iterrows():
            lines.append(f"- {row['night_id']}: {row['error']}")
    (table_dir / "anphy_epoch_metadata_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only-usable-n3", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    _, recordings = discover_recordings(dataset_root)
    channels = load_position_channels(dataset_root)
    if not channels:
        raise RuntimeError("No ANPHY position channels found; cannot build channel-expanded metadata.")

    subjects = pd.read_csv(table_dir / "anphy_sleep_subject_inventory.csv")
    cohort = pd.read_csv(table_dir / "anphy_sleep_cohort_table.csv")
    if args.only_usable_n3:
        usable = set(cohort.loc[cohort["usable"].astype(str).str.lower().isin(["true", "1"]), "night_id"])
        recordings = [recording for recording in recordings if recording.night_id in usable]

    subject_meta = subjects[["subject_id", "age", "sex"]].drop_duplicates("subject_id")
    all_epochs = []
    summaries = []
    for recording in recordings:
        try:
            annotations = read_recording_annotations(recording)
            base, summary = build_base_epochs(recording, annotations)
            meta = subject_meta[subject_meta["subject_id"] == recording.subject_id]
            age = meta["age"].iloc[0] if not meta.empty else pd.NA
            sex = meta["sex"].iloc[0] if not meta.empty else pd.NA
            base["age"] = age
            base["sex"] = sex
            repeated = []
            for channel in channels:
                part = base.copy()
                part["channel"] = channel
                repeated.append(part)
            expanded = pd.concat(repeated, ignore_index=True) if repeated else base.assign(channel=pd.NA)
            all_epochs.append(expanded)
            summary["n_channels"] = len(channels)
            summary["channels"] = "|".join(channels)
            summaries.append(summary)
        except Exception as exc:
            summaries.append(
                {
                    "subject_id": recording.subject_id,
                    "night_id": recording.night_id,
                    "n_epochs": pd.NA,
                    "n_channels": len(channels),
                    "channels": "|".join(channels),
                    "sleep_onset_min": pd.NA,
                    "n3_epochs": pd.NA,
                    "n3_min": pd.NA,
                    "n3_bouts": pd.NA,
                    "unknown_epochs": pd.NA,
                    "annotation_duration_min": pd.NA,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    metadata = pd.concat(all_epochs, ignore_index=True) if all_epochs else pd.DataFrame()
    recording_summary = pd.DataFrame(summaries).sort_values("subject_id").reset_index(drop=True)
    metadata.to_csv(table_dir / "anphy_sleep_epoch_metadata.csv.gz", index=False, compression="gzip")
    recording_summary.to_csv(table_dir / "anphy_sleep_epoch_recording_summary.csv", index=False)
    write_summary(table_dir, recording_summary, len(metadata), len(channels))

    print(f"Wrote ANPHY epoch metadata rows={len(metadata)} to {table_dir}")
    print(f"Recordings processed: {int((recording_summary['status'] == 'ok').sum())}/{len(recording_summary)}")
    print(f"N3 recording-level epochs: {int(recording_summary['n3_epochs'].fillna(0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
