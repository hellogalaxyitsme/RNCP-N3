#!/usr/bin/env python3
"""ANPHY-Sleep ZIP audit and N3 stage-duration cohort table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from anphy_sleep_common import (
    EPOCH_SEC,
    SLEEP_STAGES,
    discover_recordings,
    load_config,
    load_demographics,
    load_position_channels,
    read_recording_annotations,
)


def stage_duration_row(recording, ann: pd.DataFrame) -> dict:
    valid = ann.dropna(subset=["onset_sec", "duration_sec"]).copy()
    valid = valid[valid["duration_sec"] > 0]
    duration_min = valid.groupby("stage")["duration_sec"].sum() / 60.0
    total_sleep_min = float(duration_min.reindex(sorted(SLEEP_STAGES)).fillna(0.0).sum())
    n3_min = float(duration_min.get("N3", 0.0))
    sleep_rows = valid[valid["stage"].isin(SLEEP_STAGES)]
    sleep_onset_min = pd.NA if sleep_rows.empty else float(sleep_rows["onset_sec"].min() / 60.0)
    end_sec = (valid["onset_sec"] + valid["duration_sec"]).max() if not valid.empty else pd.NA
    return {
        "subject_id": recording.subject_id,
        "night": recording.night,
        "night_id": recording.night_id,
        "recording_key": recording.night_id,
        "zip_relative_path": recording.zip_relative_path,
        "edf_member": recording.edf_member,
        "annotation_member": recording.annotation_member,
        "annotation_rows": len(ann),
        "annotation_parse_error_rows": int((ann["parse_error"].fillna("") != "").sum()),
        "annotation_unique_stages": "|".join(sorted(ann["stage"].dropna().unique())),
        "wake_min": float(duration_min.get("W", 0.0)),
        "light_min": float(duration_min.get("LIGHT", 0.0)),
        "n1_min": float(duration_min.get("N1", 0.0)),
        "n2_min": float(duration_min.get("N2", 0.0)),
        "n3_min": n3_min,
        "rem_min": float(duration_min.get("REM", 0.0)),
        "movement_min": float(duration_min.get("MOVEMENT", 0.0)),
        "unknown_min": float(duration_min.get("UNKNOWN", 0.0)),
        "total_sleep_min": total_sleep_min,
        "sleep_onset_min": sleep_onset_min,
        "annotation_duration_min": pd.NA if pd.isna(end_sec) else float(end_sec / 60.0),
        "usable_n3_30min": n3_min >= 30.0,
        "status": "ok",
        "error": "",
    }


def write_summary(table_dir: Path, inventory: pd.DataFrame, durations: pd.DataFrame, subjects: pd.DataFrame) -> None:
    usable = durations[durations["usable_n3_30min"]].copy()
    has_zip = subjects["has_zip"].fillna(False).astype(bool)
    missing_zip = subjects.loc[~has_zip, "subject_id"].dropna().tolist()
    lines = [
        "# ANPHY-Sleep Summary",
        "",
        "## Coverage",
        "",
        f"- ZIP recordings found: {len(inventory)}",
        f"- Complete EDF + annotation pairs: {int(inventory['has_complete_pair'].sum())}",
        f"- Demographic subjects: {subjects['subject_id'].nunique()}",
        f"- Demographic subjects without ZIP: {len(missing_zip)}",
        f"- Recordings with >=30 min N3: {len(usable)}",
        f"- Subjects with >=1 usable N3 recording: {usable['subject_id'].nunique()}",
        "",
        "## N3 Minutes",
        "",
        f"- Median per recording: {durations['n3_min'].median():.2f}",
        f"- Mean per recording: {durations['n3_min'].mean():.2f}",
        f"- Min per recording: {durations['n3_min'].min():.2f}",
        f"- Max per recording: {durations['n3_min'].max():.2f}",
        "",
        "## Outputs",
        "",
        "- `anphy_sleep_recording_inventory.csv`",
        "- `anphy_sleep_subject_inventory.csv`",
        "- `anphy_sleep_stage_duration_inventory.csv`",
        "- `anphy_sleep_cohort_table.csv`",
    ]
    if missing_zip:
        lines += ["", "## Missing ZIPs From Demographics", ""]
        lines.extend(f"- {subject_id}" for subject_id in missing_zip)
    (table_dir / "anphy_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    inventory, recordings = discover_recordings(dataset_root)
    demographics = load_demographics(dataset_root)
    position_channels = load_position_channels(dataset_root)

    duration_rows = []
    for recording in recordings:
        try:
            ann = read_recording_annotations(recording)
            duration_rows.append(stage_duration_row(recording, ann))
        except Exception as exc:
            duration_rows.append(
                {
                    "subject_id": recording.subject_id,
                    "night": recording.night,
                    "night_id": recording.night_id,
                    "recording_key": recording.night_id,
                    "zip_relative_path": recording.zip_relative_path,
                    "edf_member": recording.edf_member,
                    "annotation_member": recording.annotation_member,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    durations = pd.DataFrame(duration_rows).sort_values("subject_id").reset_index(drop=True)
    subjects = demographics.merge(
        inventory[["subject_id", "zip_relative_path", "has_zip", "has_edf", "has_annotation", "has_complete_pair"]],
        on="subject_id",
        how="outer",
    )
    subjects["has_zip"] = subjects["has_zip"].fillna(False)
    subjects["has_edf"] = subjects["has_edf"].fillna(False)
    subjects["has_annotation"] = subjects["has_annotation"].fillna(False)
    subjects["has_complete_pair"] = subjects["has_complete_pair"].fillna(False)
    subjects = subjects.sort_values("subject_id").reset_index(drop=True)

    cohort = durations.merge(subjects[["subject_id", "age", "sex"]], on="subject_id", how="left")
    cohort = cohort.rename(columns={"recording_key": "night_id_raw", "n3_min": "total_n3_min"})
    cohort["night_id"] = cohort["subject_id"]
    cohort["clean_n3_min"] = pd.NA
    cohort["usable"] = cohort["usable_n3_30min"].fillna(False)
    cohort["eeg_position_channel_count"] = len(position_channels)
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
            "annotation_duration_min",
            "eeg_position_channel_count",
            "status",
        ]
    ].sort_values("subject_id")

    inventory.to_csv(table_dir / "anphy_sleep_recording_inventory.csv", index=False)
    subjects.to_csv(table_dir / "anphy_sleep_subject_inventory.csv", index=False)
    durations.to_csv(table_dir / "anphy_sleep_stage_duration_inventory.csv", index=False)
    cohort.to_csv(table_dir / "anphy_sleep_cohort_table.csv", index=False)
    pd.DataFrame({"channel": position_channels}).to_csv(table_dir / "anphy_sleep_position_channels.csv", index=False)
    write_summary(table_dir, inventory, durations, subjects)

    print(f"Wrote ANPHY audit tables to {table_dir}")
    print(f"Complete pairs: {int(inventory['has_complete_pair'].sum())}/{len(inventory)}")
    print(f"Usable N3 recordings: {int(cohort['usable'].sum())}")
    print(f"Position EEG channels: {len(position_channels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
