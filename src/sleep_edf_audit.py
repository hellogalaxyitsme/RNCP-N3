#!/usr/bin/env python3
"""audit for the Sleep-EDF Expanded dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SC_RE = re.compile(r"^(SC)(\d{3})([12])([A-Z0-9]{2})-(PSG|Hypnogram)\.edf$", re.IGNORECASE)
ST_RE = re.compile(r"^(ST)(\d{3})([12])([A-Z0-9]{2})-(PSG|Hypnogram)\.edf$", re.IGNORECASE)


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    project_data_root: Path
    cache_root: Path
    primary_subset: str

    @property
    def table_dir(self) -> Path:
        return self.project_data_root / "tables"

    @property
    def log_dir(self) -> Path:
        return self.project_data_root / "logs"


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return Config(
        dataset_root=Path(raw["dataset_root"]),
        project_data_root=Path(raw["project_data_root"]),
        cache_root=Path(raw["cache_root"]),
        primary_subset=raw.get("primary_subset", "sleep-cassette"),
    )


def ensure_output_dirs(cfg: Config) -> None:
    cfg.table_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_root.mkdir(parents=True, exist_ok=True)


def parse_edf_name(path: Path, dataset_root: Path) -> dict:
    rel = path.relative_to(dataset_root).as_posix()
    subset = path.parent.name
    name = path.name
    match = SC_RE.match(name) or ST_RE.match(name)

    row = {
        "relative_path": rel,
        "file_name": name,
        "subset": subset,
        "size_bytes": path.stat().st_size,
        "is_edf": path.suffix.lower() == ".edf",
        "cohort": None,
        "subject_id": None,
        "subject_num": None,
        "night": None,
        "recording_key": None,
        "file_kind": None,
    }

    if match:
        cohort, subject_num, night, suffix, file_kind = match.groups()
        cohort = cohort.upper()
        subject_num = subject_num
        row.update(
            {
                "cohort": cohort,
                "subject_num": subject_num,
                "subject_id": f"{cohort}{subject_num}",
                "night": int(night),
                "recording_key": f"{cohort}{subject_num}{night}",
                "file_kind": file_kind.lower(),
                "recording_suffix": suffix,
            }
        )
    else:
        row["recording_suffix"] = None

    return row


def build_file_inventory(cfg: Config) -> pd.DataFrame:
    files = sorted(p for p in cfg.dataset_root.rglob("*") if p.is_file())
    rows = [parse_edf_name(p, cfg.dataset_root) for p in files]
    return pd.DataFrame(rows)


def build_recording_inventory(file_inventory: pd.DataFrame, subset: str) -> pd.DataFrame:
    edf = file_inventory[
        (file_inventory["subset"] == subset)
        & (file_inventory["is_edf"])
        & (file_inventory["recording_key"].notna())
    ].copy()

    psg = edf[edf["file_kind"] == "psg"].copy()
    hyp = edf[edf["file_kind"] == "hypnogram"].copy()

    psg = psg.rename(
        columns={
            "relative_path": "psg_relative_path",
            "file_name": "psg_file_name",
            "size_bytes": "psg_size_bytes",
            "recording_suffix": "psg_suffix",
        }
    )
    hyp = hyp.rename(
        columns={
            "relative_path": "hypnogram_relative_path",
            "file_name": "hypnogram_file_name",
            "size_bytes": "hypnogram_size_bytes",
            "recording_suffix": "hypnogram_suffix",
        }
    )

    keep_common = ["cohort", "subject_num", "subject_id", "night", "recording_key"]
    psg_keep = keep_common + ["psg_relative_path", "psg_file_name", "psg_size_bytes", "psg_suffix"]
    hyp_keep = keep_common + [
        "hypnogram_relative_path",
        "hypnogram_file_name",
        "hypnogram_size_bytes",
        "hypnogram_suffix",
    ]

    merged = pd.merge(psg[psg_keep], hyp[hyp_keep], on=keep_common, how="outer")
    merged["has_psg"] = merged["psg_relative_path"].notna()
    merged["has_hypnogram"] = merged["hypnogram_relative_path"].notna()
    merged["has_complete_pair"] = merged["has_psg"] & merged["has_hypnogram"]
    return merged.sort_values(["subject_id", "night"]).reset_index(drop=True)


def read_xls_via_libreoffice(xls_path: Path, cache_root: Path) -> pd.DataFrame | None:
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not libreoffice:
        return None

    out_dir = cache_root / "excel_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_csv = out_dir / f"{xls_path.stem}.csv"
    if expected_csv.exists():
        expected_csv.unlink()

    cmd = [
        libreoffice,
        "--headless",
        "--convert-to",
        "csv",
        "--outdir",
        str(out_dir),
        str(xls_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if not expected_csv.exists():
        candidates = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        expected_csv = candidates[0]

    return pd.read_csv(expected_csv)


def load_sc_subjects(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    xls_path = cfg.dataset_root / "SC-subjects.xls"
    if not xls_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    raw = None
    try:
        raw = pd.read_excel(xls_path)
    except Exception:
        raw = read_xls_via_libreoffice(xls_path, cfg.cache_root)

    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw = raw.dropna(how="all").copy()
    raw.columns = [str(c).strip() for c in raw.columns]

    lower_to_col = {c.lower().strip(): c for c in raw.columns}
    subject_col = next(
        (c for key, c in lower_to_col.items() if "subject" in key or key in {"id", "subj"}),
        raw.columns[0],
    )
    age_col = next((c for key, c in lower_to_col.items() if "age" in key), None)
    sex_col = next((c for key, c in lower_to_col.items() if "sex" in key or "gender" in key), None)

    standardized = pd.DataFrame()
    standardized["subject_metadata_value"] = raw[subject_col]
    standardized["subject_id"] = standardized["subject_metadata_value"].map(normalize_sc_subject_id)
    if age_col:
        standardized["age"] = pd.to_numeric(raw[age_col], errors="coerce")
    else:
        standardized["age"] = pd.NA
    if sex_col:
        standardized["sex"] = raw[sex_col].astype(str).str.strip()
    else:
        standardized["sex"] = pd.NA

    standardized = standardized.dropna(subset=["subject_id"]).drop_duplicates("subject_id")
    return raw, standardized


def normalize_sc_subject_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if upper.startswith("SC") and len(upper) >= 5:
        return upper[:5]

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    if len(digits) >= 3 and digits[-3:].startswith("4"):
        return f"SC{digits[-3:]}"
    return f"SC4{int(digits):02d}"


def build_subject_inventory(recordings: pd.DataFrame, subject_meta: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        recordings.groupby(["cohort", "subject_num", "subject_id"], dropna=False)
        .agg(
            n_recordings=("recording_key", "count"),
            n_complete_pairs=("has_complete_pair", "sum"),
            n_psg=("has_psg", "sum"),
            n_hypnogram=("has_hypnogram", "sum"),
            nights=("night", lambda x: ",".join(str(int(v)) for v in sorted(x.dropna().unique()))),
        )
        .reset_index()
    )
    if not subject_meta.empty:
        grouped = grouped.merge(subject_meta[["subject_id", "age", "sex"]], on="subject_id", how="left")
    else:
        grouped["age"] = pd.NA
        grouped["sex"] = pd.NA
    return grouped.sort_values("subject_id").reset_index(drop=True)


def write_summary(cfg: Config, file_inventory: pd.DataFrame, recordings: pd.DataFrame, subjects: pd.DataFrame) -> None:
    lines = [
        "# Sleep-EDF Audit Summary",
        "",
        f"Dataset root: `{cfg.dataset_root}`",
        f"Project data root: `{cfg.project_data_root}`",
        f"Primary subset: `{cfg.primary_subset}`",
        "",
        "## File Counts",
        "",
        f"- Total files: {len(file_inventory)}",
        f"- EDF files: {int(file_inventory['is_edf'].sum())}",
        f"- SC PSG files: {int(((file_inventory['cohort'] == 'SC') & (file_inventory['file_kind'] == 'psg')).sum())}",
        f"- SC hypnograms: {int(((file_inventory['cohort'] == 'SC') & (file_inventory['file_kind'] == 'hypnogram')).sum())}",
        f"- ST PSG files: {int(((file_inventory['cohort'] == 'ST') & (file_inventory['file_kind'] == 'psg')).sum())}",
        f"- ST hypnograms: {int(((file_inventory['cohort'] == 'ST') & (file_inventory['file_kind'] == 'hypnogram')).sum())}",
        "",
        "## Primary Subset",
        "",
        f"- SC recording rows: {len(recordings)}",
        f"- SC complete PSG/hypnogram pairs: {int(recordings['has_complete_pair'].sum())}",
        f"- SC subjects: {subjects['subject_id'].nunique()}",
        f"- Subjects with age metadata: {int(subjects['age'].notna().sum())}",
        f"- Subjects with sex metadata: {int(subjects['sex'].notna().sum())}",
    ]
    (cfg.table_dir / "sleep_edf_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    if not cfg.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {cfg.dataset_root}")

    file_inventory = build_file_inventory(cfg)
    recordings = build_recording_inventory(file_inventory, cfg.primary_subset)
    raw_subjects, subject_meta = load_sc_subjects(cfg)
    subjects = build_subject_inventory(recordings, subject_meta)

    file_inventory.to_csv(cfg.table_dir / "sleep_edf_file_inventory.csv", index=False)
    recordings.to_csv(cfg.table_dir / "sleep_edf_sc_recording_inventory.csv", index=False)
    subjects.to_csv(cfg.table_dir / "sleep_edf_sc_subject_inventory.csv", index=False)
    if not raw_subjects.empty:
        raw_subjects.to_csv(cfg.table_dir / "sleep_edf_sc_subjects_raw_from_xls.csv", index=False)
    write_summary(cfg, file_inventory, recordings, subjects)

    print(f"Wrote audit tables to {cfg.table_dir}")
    print(f"SC complete pairs: {int(recordings['has_complete_pair'].sum())}")
    print(f"SC subjects: {subjects['subject_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
