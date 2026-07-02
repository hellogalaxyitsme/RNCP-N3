#!/usr/bin/env python3
"""Summarize N3 complexity feature outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FEATURES = [
    "lzc",
    "permutation_entropy",
    "aperiodic_exponent_specparam",
    "specparam_r_squared",
    "specparam_error",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def md_table(frame: pd.DataFrame, float_fmt: str = ".4f") -> str:
    return frame.to_markdown(floatfmt=float_fmt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    path = table_dir / "sleep_edf_sc_complexity_features.csv.gz"
    out_path = table_dir / "complexity_snapshot.md"

    df = pd.read_csv(path)
    desc = df[FEATURES].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    corr = df[["lzc", "permutation_entropy", "aperiodic_exponent_specparam"]].corr()
    per_recording = df.groupby("night_id").size().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    per_subject = df.groupby("subject_id").size().describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    by_channel = df.groupby("channel")[["lzc", "permutation_entropy", "aperiodic_exponent_specparam"]].agg(
        ["count", "mean", "std", "median"]
    )

    low_fit = int((df["specparam_r_squared"] < 0.90).sum())
    very_low_fit = int((df["specparam_r_squared"] < 0.80).sum())
    high_error = int((df["specparam_error"] > 0.10).sum())

    lines = [
        "# Complexity Snapshot",
        "",
        "## Coverage",
        "",
        f"- Rows: {len(df)}",
        f"- Subjects: {df['subject_id'].nunique()}",
        f"- Recordings with N3 rows: {df['night_id'].nunique()}",
        f"- Channels: {df['channel'].nunique()}",
        f"- Stages: {', '.join(sorted(df['stage'].dropna().unique()))}",
        f"- Missing LZc rows: {int(df['lzc'].isna().sum())}",
        f"- Missing permutation entropy rows: {int(df['permutation_entropy'].isna().sum())}",
        f"- Missing specparam exponent rows: {int(df['aperiodic_exponent_specparam'].isna().sum())}",
        "",
        "## Feature Distributions",
        "",
        md_table(desc),
        "",
        "## Feature Correlations",
        "",
        md_table(corr),
        "",
        "## N3 Rows Per Recording",
        "",
        md_table(per_recording.to_frame("rows")),
        "",
        "## N3 Rows Per Subject",
        "",
        md_table(per_subject.to_frame("rows")),
        "",
        "## Channel Summary",
        "",
        md_table(by_channel),
        "",
        "## Specparam Fit Screen",
        "",
        f"- Rows with R^2 < 0.90: {low_fit}",
        f"- Rows with R^2 < 0.80: {very_low_fit}",
        f"- Rows with error > 0.10: {high_error}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
