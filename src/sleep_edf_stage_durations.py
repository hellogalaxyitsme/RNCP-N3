#!/usr/bin/env python3
"""lightweight Sleep-EDF stage-duration audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mne
import pandas as pd


STAGE_COLUMNS = [
    "wake_min",
    "n1_min",
    "n2_min",
    "n3_stage3_min",
    "n3_stage4_min",
    "n3_min",
    "rem_min",
    "movement_min",
    "unknown_min",
    "total_sleep_min",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in cfg.items()}


def stage_key(description: str) -> str:
    desc = description.strip().lower()
    if desc in {"sleep stage w", "stage w", "wake", "w"}:
        return "wake"
    if desc in {"sleep stage 1", "stage 1", "n1", "1"}:
        return "n1"
    if desc in {"sleep stage 2", "stage 2", "n2", "2"}:
        return "n2"
    if desc in {"sleep stage 3", "stage 3", "n3", "3"}:
        return "n3_stage3"
    if desc in {"sleep stage 4", "stage 4", "n4", "4"}:
        return "n3_stage4"
    if desc in {"sleep stage r", "stage r", "rem", "r"}:
        return "rem"
    if "movement" in desc:
        return "movement"
    return "unknown"


def overlap_seconds(start: float, duration: float, lo: float, hi: float) -> float:
    end = start + duration
    return max(0.0, min(end, hi) - max(start, lo))


def summarize_recording(row: pd.Series, dataset_root: Path) -> dict:
    psg_path = dataset_root / row["psg_relative_path"]
    hyp_path = dataset_root / row["hypnogram_relative_path"]

    out = row.to_dict()
    out.update({c: pd.NA for c in STAGE_COLUMNS})
    out.update(
        {
            "status": "ok",
            "error": "",
            "sfreq_hz": pd.NA,
            "n_channels": pd.NA,
            "channel_names": "",
            "psg_duration_min": pd.NA,
            "annotation_count": pd.NA,
            "annotation_descriptions": "",
            "sleep_onset_min": pd.NA,
            "usable_n3_30min": False,
        }
    )

    try:
        raw = mne.io.read_raw_edf(psg_path, preload=False, verbose="ERROR")
        annotations = mne.read_annotations(hyp_path)

        sfreq = float(raw.info["sfreq"])
        raw_duration_sec = float(raw.n_times) / sfreq
        out["sfreq_hz"] = sfreq
        out["n_channels"] = len(raw.ch_names)
        out["channel_names"] = "|".join(raw.ch_names)
        out["psg_duration_min"] = raw_duration_sec / 60.0
        out["annotation_count"] = len(annotations)
        out["annotation_descriptions"] = "|".join(sorted(set(map(str, annotations.description))))

        durations = defaultdict(float)
        sleep_onset = None
        sleep_keys = {"n1", "n2", "n3_stage3", "n3_stage4", "rem"}

        for onset, duration, desc in zip(annotations.onset, annotations.duration, annotations.description):
            clipped = overlap_seconds(float(onset), float(duration), 0.0, raw_duration_sec)
            if clipped <= 0:
                continue
            key = stage_key(str(desc))
            durations[key] += clipped
            if sleep_onset is None and key in sleep_keys:
                sleep_onset = max(float(onset), 0.0)

        values_min = {k: v / 60.0 for k, v in durations.items()}
        n3_min = values_min.get("n3_stage3", 0.0) + values_min.get("n3_stage4", 0.0)
        total_sleep_min = (
            values_min.get("n1", 0.0)
            + values_min.get("n2", 0.0)
            + n3_min
            + values_min.get("rem", 0.0)
        )

        out.update(
            {
                "wake_min": values_min.get("wake", 0.0),
                "n1_min": values_min.get("n1", 0.0),
                "n2_min": values_min.get("n2", 0.0),
                "n3_stage3_min": values_min.get("n3_stage3", 0.0),
                "n3_stage4_min": values_min.get("n3_stage4", 0.0),
                "n3_min": n3_min,
                "rem_min": values_min.get("rem", 0.0),
                "movement_min": values_min.get("movement", 0.0),
                "unknown_min": values_min.get("unknown", 0.0),
                "total_sleep_min": total_sleep_min,
                "sleep_onset_min": pd.NA if sleep_onset is None else sleep_onset / 60.0,
                "usable_n3_30min": n3_min >= 30.0,
            }
        )
    except Exception as exc:
        out["status"] = "error"
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out


def write_summary(table_dir: Path, durations: pd.DataFrame, cohort: pd.DataFrame) -> None:
    ok = durations[durations["status"] == "ok"].copy()
    usable = ok[ok["usable_n3_30min"]].copy()

    lines = [
        "# Sleep-EDF SC Stage-Duration Summary",
        "",
        f"- Recording rows: {len(durations)}",
        f"- Successfully loaded pairs: {len(ok)}",
        f"- Failed pairs: {int((durations['status'] != 'ok').sum())}",
        f"- Subjects total: {durations['subject_id'].nunique()}",
        f"- Recordings with >=30 min N3: {len(usable)}",
        f"- Subjects with >=1 usable N3 recording: {usable['subject_id'].nunique()}",
        "",
        "## N3 Minutes",
        "",
        f"- Median per recording: {ok['n3_min'].median():.2f}",
        f"- Mean per recording: {ok['n3_min'].mean():.2f}",
        f"- Min per recording: {ok['n3_min'].min():.2f}",
        f"- Max per recording: {ok['n3_min'].max():.2f}",
        "",
        "## Outputs",
        "",
        "- `sleep_edf_sc_stage_duration_inventory.csv`",
        "- `sleep_edf_sc_cohort_table.csv`",
    ]
    if (durations["status"] != "ok").any():
        lines += ["", "## Errors", ""]
        for _, row in durations[durations["status"] != "ok"].iterrows():
            lines.append(f"- {row['recording_key']}: {row['error']}")

    (table_dir / "stage_duration_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    project_data_root = cfg["project_data_root"]
    table_dir = project_data_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    recording_path = table_dir / "sleep_edf_sc_recording_inventory.csv"
    subject_path = table_dir / "sleep_edf_sc_subject_inventory.csv"
    recordings = pd.read_csv(recording_path)
    subjects = pd.read_csv(subject_path)

    rows = [summarize_recording(row, dataset_root) for _, row in recordings.iterrows()]
    durations = pd.DataFrame(rows)
    durations = durations.merge(subjects[["subject_id", "age", "sex"]], on="subject_id", how="left")

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
        ]
    ].copy()
    cohort = cohort.rename(
        columns={
            "recording_key": "night_id",
            "n3_min": "total_n3_min",
            "usable_n3_30min": "usable",
        }
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
        ]
    ].sort_values(["subject_id", "night_id"])

    durations.to_csv(table_dir / "sleep_edf_sc_stage_duration_inventory.csv", index=False)
    cohort.to_csv(table_dir / "sleep_edf_sc_cohort_table.csv", index=False)
    write_summary(table_dir, durations, cohort)

    print(f"Wrote stage-duration tables to {table_dir}")
    print(f"Loaded pairs: {int((durations['status'] == 'ok').sum())}/{len(durations)}")
    print(f"Recordings with >=30 min N3: {int(durations['usable_n3_30min'].sum())}")
    print(f"Subjects with >=1 usable N3 recording: {durations.loc[durations['usable_n3_30min'], 'subject_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
