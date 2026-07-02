#!/usr/bin/env python3
"""Shared ANPHY-Sleep dataset helpers."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EPOCH_SEC = 30.0
SUBJECT_RE = re.compile(r"(EPCTL\d{2})", re.IGNORECASE)
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


@dataclass(frozen=True)
class AnphyRecording:
    subject_id: str
    zip_path: Path
    zip_relative_path: str
    edf_member: str
    annotation_member: str
    edf_size_bytes: int
    annotation_size_bytes: int

    @property
    def night(self) -> int:
        return 1

    @property
    def night_id(self) -> str:
        return self.subject_id


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def normalize_stage(value: object) -> tuple[str, str]:
    original = str(value).strip()
    stage = original.upper()
    if stage in {"W", "WAKE"}:
        return "W", original
    if stage in {"N1", "S1", "1"}:
        return "N1", original
    if stage in {"N2", "S2", "2"}:
        return "N2", original
    if stage in {"N3", "S3", "S4", "3", "4"}:
        return "N3", original
    if stage in {"R", "REM"}:
        return "REM", original
    if stage == "L":
        return "LIGHT", original
    if "MOVEMENT" in stage or stage in {"MT", "M"}:
        return "MOVEMENT", original
    return "UNKNOWN", original


def subject_id_from_name(name: str) -> str | None:
    match = SUBJECT_RE.search(name)
    if not match:
        return None
    return match.group(1).upper()


def anphy_zip_paths(dataset_root: Path) -> list[Path]:
    return sorted(dataset_root.glob("EPCTL*.zip"), key=lambda p: subject_id_from_name(p.name) or p.name)


def select_member(members: list[zipfile.ZipInfo], suffix: str) -> zipfile.ZipInfo | None:
    candidates = [
        item
        for item in members
        if not item.is_dir() and item.filename.lower().endswith(suffix.lower())
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (0 if subject_id_from_name(item.filename) else 1, item.filename.lower()))
    return candidates[0]


def discover_recordings(dataset_root: Path) -> tuple[pd.DataFrame, list[AnphyRecording]]:
    rows = []
    recordings: list[AnphyRecording] = []
    for zip_path in anphy_zip_paths(dataset_root):
        subject_id = subject_id_from_name(zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.infolist()
            edf = select_member(members, ".edf")
            annot = select_member(members, ".txt")

        row = {
            "subject_id": subject_id,
            "night": 1,
            "night_id": subject_id,
            "recording_key": subject_id,
            "zip_relative_path": zip_path.relative_to(dataset_root).as_posix(),
            "zip_file_name": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "edf_member": None if edf is None else edf.filename,
            "edf_size_bytes": None if edf is None else edf.file_size,
            "annotation_member": None if annot is None else annot.filename,
            "annotation_size_bytes": None if annot is None else annot.file_size,
            "has_zip": True,
            "has_edf": edf is not None,
            "has_annotation": annot is not None,
            "has_complete_pair": edf is not None and annot is not None,
        }
        rows.append(row)
        if subject_id and edf is not None and annot is not None:
            recordings.append(
                AnphyRecording(
                    subject_id=subject_id,
                    zip_path=zip_path,
                    zip_relative_path=row["zip_relative_path"],
                    edf_member=edf.filename,
                    annotation_member=annot.filename,
                    edf_size_bytes=int(edf.file_size),
                    annotation_size_bytes=int(annot.file_size),
                )
            )

    inventory = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    return inventory, recordings


def read_annotation_text(zip_path: Path, annotation_member: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(annotation_member) as f:
            return f.read().decode("utf-8-sig", errors="replace")


def parse_annotation_text(text: str) -> pd.DataFrame:
    rows = []
    for line_num, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = re.split(r"[\s,;]+", stripped)
        if len(parts) < 3:
            rows.append(
                {
                    "line_num": line_num,
                    "stage": "UNKNOWN",
                    "stage_original": stripped,
                    "onset_sec": pd.NA,
                    "duration_sec": pd.NA,
                    "parse_error": "expected_stage_onset_duration",
                }
            )
            continue
        stage, original = normalize_stage(parts[0])
        rows.append(
            {
                "line_num": line_num,
                "stage": stage,
                "stage_original": original,
                "onset_sec": pd.to_numeric(parts[1], errors="coerce"),
                "duration_sec": pd.to_numeric(parts[2], errors="coerce"),
                "parse_error": "",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["line_num", "stage", "stage_original", "onset_sec", "duration_sec", "parse_error"]
        )
    out["onset_sec"] = pd.to_numeric(out["onset_sec"], errors="coerce")
    out["duration_sec"] = pd.to_numeric(out["duration_sec"], errors="coerce")
    return out


def read_recording_annotations(recording: AnphyRecording) -> pd.DataFrame:
    text = read_annotation_text(recording.zip_path, recording.annotation_member)
    ann = parse_annotation_text(text)
    ann["subject_id"] = recording.subject_id
    ann["night_id"] = recording.night_id
    ann["night"] = recording.night
    return ann


def load_demographics(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "Details information for healthy subjects.csv"
    if not path.exists():
        return pd.DataFrame(columns=["subject_id", "sex", "age"])
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    subject_col = next((c for c in raw.columns if c.lower().startswith("subjects")), raw.columns[0])
    out = raw.copy()
    out["subject_id"] = out[subject_col].map(subject_id_from_name)
    out = out.rename(columns={"Sex": "sex", "Age": "age"})
    if "sex" not in out.columns:
        out["sex"] = pd.NA
    if "age" not in out.columns:
        out["age"] = pd.NA
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


def load_position_channels(dataset_root: Path) -> list[str]:
    path = dataset_root / "Co-registered average positions.pos"
    if not path.exists():
        return []
    channels = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                channels.append(parts[1])
    return channels


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
