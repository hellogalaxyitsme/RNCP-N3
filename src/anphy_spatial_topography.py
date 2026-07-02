#!/usr/bin/env python3
"""ANPHY high-RNCP spatial topography analysis.

this analysis asks whether high-RNCP N3 epochs show spatially organized residual
complexity expression across ANPHY's high-density EEG montage. High and low
RNCP epochs are defined within subject from the epoch-level channel-averaged
RNCP magnitude, then channel and regional deltas are summarized at the subject
level to avoid treating channel rows as independent subjects.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RESIDUAL_COLS = [
    "lzc_rncp_residual_z",
    "permutation_entropy_rncp_residual_z",
    "spectral_entropy_rncp_residual_z",
    "aperiodic_exponent_specparam_rncp_residual_z",
]
COMPONENT_LABELS = {
    "lzc_rncp_residual_z": "LZc",
    "permutation_entropy_rncp_residual_z": "PE",
    "spectral_entropy_rncp_residual_z": "SE",
    "aperiodic_exponent_specparam_rncp_residual_z": "AE",
}
TOPO_COLS = ["rncp_l2_norm"] + RESIDUAL_COLS
TOPO_LABELS = {
    "rncp_l2_norm": "RNCP magnitude",
    **COMPONENT_LABELS,
}


@dataclass(frozen=True)
class RegionRule:
    region: str
    contrast_group: str
    prefixes: tuple[str, ...]


REGION_RULES = [
    RegionRule("Frontal", "frontoparietal", ("FP", "AF", "F")),
    RegionRule("Central", "frontoparietal", ("FC", "C")),
    RegionRule("Parietal", "frontoparietal", ("CP", "P")),
    RegionRule("Occipital", "occipital", ("PO", "O", "I")),
    RegionRule("Temporal", "temporal", ("FT", "T", "TP")),
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def norm_channel(name: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


def channel_region(channel: object) -> tuple[str, str]:
    key = norm_channel(channel)
    prefix_rules = [
        (prefix, rule.region, rule.contrast_group)
        for rule in REGION_RULES
        for prefix in rule.prefixes
    ]
    for prefix, region, contrast_group in sorted(prefix_rules, key=lambda item: len(item[0]), reverse=True):
        if key.startswith(prefix):
            return region, contrast_group
    for rule in REGION_RULES:
        if any(key.startswith(prefix) for prefix in rule.prefixes):
            return rule.region, rule.contrast_group
    return "Other", "other"


def load_positions(dataset_root: Path, include_peripheral: bool = False) -> pd.DataFrame:
    pos_path = dataset_root / "Co-registered average positions.pos"
    if not pos_path.exists():
        raise FileNotFoundError(f"ANPHY position file not found: {pos_path}")

    rows = []
    with pos_path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        expected = int(first) if first.isdigit() else None
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "channel_index": int(parts[0]),
                    "channel": parts[1],
                    "x_anterior": float(parts[2]),
                    "y_left": float(parts[3]),
                    "z_superior": float(parts[4]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No channel positions parsed from {pos_path}")
    if expected is not None and expected != len(out):
        print(f"Warning: position file expected {expected} channels, parsed {len(out)}", flush=True)

    out["channel_key"] = out["channel"].map(norm_channel)
    out["is_peripheral"] = (
        out["channel_key"].str.startswith(("SO", "ZY"))
        | (out["z_superior"] < 0)
    )
    if not include_peripheral:
        out = out[~out["is_peripheral"]].copy()
    out[["region", "contrast_group"]] = out["channel"].apply(lambda ch: pd.Series(channel_region(ch)))

    # Topomap orientation: y_left gives left/right and x_anterior gives anterior/posterior.
    # Store right-positive x coordinates while preserving left-positive source coordinates in y_left.
    out["topo_x"] = -out["y_left"]
    out["topo_y"] = out["x_anterior"]
    scale = float(np.nanmax(np.sqrt(out["topo_x"] ** 2 + out["topo_y"] ** 2)))
    out["topo_x"] = out["topo_x"] / scale
    out["topo_y"] = out["topo_y"] / scale
    return out


def read_residuals(table_dir: Path, residuals_file: str) -> pd.DataFrame:
    path = table_dir / residuals_file
    if not path.exists():
        raise FileNotFoundError(f"Residual table not found: {path}")
    usecols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "channel",
        "rncp_l2_norm",
        *RESIDUAL_COLS,
    ]
    df = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
    missing = sorted(set(usecols) - set(df.columns))
    if missing:
        raise ValueError(f"Residual table is missing required columns: {missing}")
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    df["channel_key"] = df["channel"].map(norm_channel)
    df["epoch_idx"] = pd.to_numeric(df["epoch_idx"], errors="coerce").astype("Int64")
    return df.dropna(subset=["epoch_idx", "rncp_l2_norm"]).copy()


def assign_epoch_groups(df: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    epoch = (
        df.groupby(["subject_id", "night_id", "epoch_idx"], as_index=False)
        .agg(epoch_rncp_l2=("rncp_l2_norm", "mean"), rncp_channel_count=("channel", "nunique"))
    )

    quantiles = (
        epoch.groupby("subject_id")["epoch_rncp_l2"]
        .quantile([low_q, high_q])
        .unstack()
        .rename(columns={low_q: "low_cut", high_q: "high_cut"})
        .reset_index()
    )
    epoch = epoch.merge(quantiles, on="subject_id", how="left")
    epoch["rncp_group"] = np.where(
        epoch["epoch_rncp_l2"] <= epoch["low_cut"],
        "low",
        np.where(epoch["epoch_rncp_l2"] >= epoch["high_cut"], "high", "middle"),
    )
    return epoch


def one_sample_stats(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return {"n_subjects": int(len(clean)), "mean": np.nan, "sd": np.nan, "sem": np.nan, "t": np.nan, "p": np.nan}
    t_stat, p_val = stats.ttest_1samp(clean.to_numpy(), popmean=0.0, nan_policy="omit")
    return {
        "n_subjects": int(len(clean)),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)),
        "sem": float(clean.sem(ddof=1)),
        "t": float(t_stat),
        "p": float(p_val),
    }


def high_low_delta(table: pd.DataFrame, index_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = table.groupby(index_cols + ["rncp_group"], as_index=False)[value_cols].mean()
    high = grouped[grouped["rncp_group"] == "high"].drop(columns="rncp_group")
    low = grouped[grouped["rncp_group"] == "low"].drop(columns="rncp_group")
    merged = high.merge(low, on=index_cols, suffixes=("_high", "_low"), how="inner")
    for col in value_cols:
        merged[f"{col}_delta_high_minus_low"] = merged[f"{col}_high"] - merged[f"{col}_low"]
    return merged


def summarize_channel_deltas(channel_subject: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [f"{col}_delta_high_minus_low" for col in TOPO_COLS]
    rows = []
    for channel_key, df_ch in channel_subject.groupby("channel_key"):
        row = {"channel_key": channel_key}
        for col in delta_cols:
            stats_row = one_sample_stats(df_ch[col])
            prefix = col.replace("_delta_high_minus_low", "")
            for key, value in stats_row.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    out = pd.DataFrame(rows)
    return positions.merge(out, on="channel_key", how="inner")


def summarize_region_deltas(region_subject: pd.DataFrame) -> pd.DataFrame:
    delta_cols = [f"{col}_delta_high_minus_low" for col in TOPO_COLS]
    rows = []
    for region, df_region in region_subject.groupby("region"):
        row = {"region": region}
        for col in delta_cols:
            stats_row = one_sample_stats(df_region[col])
            prefix = col.replace("_delta_high_minus_low", "")
            for key, value in stats_row.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    order = ["Frontal", "Central", "Parietal", "Occipital", "Temporal", "Other"]
    out = pd.DataFrame(rows)
    out["region"] = pd.Categorical(out["region"], categories=order, ordered=True)
    return out.sort_values("region").reset_index(drop=True)


def frontoparietal_occipital_contrast(region_subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in TOPO_COLS:
        delta_col = f"{col}_delta_high_minus_low"
        tmp = region_subject[region_subject["contrast_group"].isin(["frontoparietal", "occipital"])].copy()
        by_group = tmp.groupby(["subject_id", "contrast_group"], as_index=False)[delta_col].mean()
        wide = by_group.pivot(index="subject_id", columns="contrast_group", values=delta_col).dropna()
        wide["frontoparietal_minus_occipital"] = wide["frontoparietal"] - wide["occipital"]
        stats_row = one_sample_stats(wide["frontoparietal_minus_occipital"])
        rows.append(
            {
                "metric": col,
                "metric_label": TOPO_LABELS[col],
                **{f"contrast_{key}": value for key, value in stats_row.items()},
                "frontoparietal_mean_delta": float(wide["frontoparietal"].mean()) if len(wide) else np.nan,
                "occipital_mean_delta": float(wide["occipital"].mean()) if len(wide) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    out_dir: Path,
    prefix: str,
    residuals_file: str,
    epoch_groups: pd.DataFrame,
    channel_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    contrast: pd.DataFrame,
    low_q: float,
    high_q: float,
) -> None:
    n_subjects = int(epoch_groups["subject_id"].nunique())
    group_counts = epoch_groups["rncp_group"].value_counts().to_dict()
    rncp_region = region_summary[["region", "rncp_l2_norm_n_subjects", "rncp_l2_norm_mean", "rncp_l2_norm_sem", "rncp_l2_norm_t", "rncp_l2_norm_p"]].copy()
    rncp_contrast = contrast[contrast["metric"] == "rncp_l2_norm"].iloc[0]

    lines = [
        "# ANPHY Spatial Topography Summary",
        "",
        f"Residual source: `{residuals_file}`",
        f"Channels retained for scalp topography: {len(channel_summary)}",
        f"High/low RNCP epochs defined within subject using bottom {low_q:.0%} and top {1 - high_q:.0%} of channel-averaged epoch RNCP magnitude.",
        f"Subjects: {n_subjects}",
        f"Epoch groups: {group_counts}",
        "",
        "All channel and region statistics are based on subject-level high-minus-low deltas.",
        "",
        "## Regional RNCP magnitude deltas",
        "",
        rncp_region.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Frontoparietal minus occipital contrast",
        "",
        pd.DataFrame([rncp_contrast]).to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        f"- `{prefix}_epoch_rncp_groups.csv.gz`",
        f"- `{prefix}_channel_subject_deltas.csv`",
        f"- `{prefix}_channel_topography_summary.csv`",
        f"- `{prefix}_region_subject_deltas.csv`",
        f"- `{prefix}_region_topography_summary.csv`",
        f"- `{prefix}_frontoparietal_occipital_contrast.csv`",
    ]
    (out_dir / f"{prefix}_spatial_topography_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--residuals-file", default="anphy_sleep_n3_rncp_residuals.csv.gz")
    parser.add_argument("--low-quantile", type=float, default=0.20)
    parser.add_argument("--high-quantile", type=float, default=0.80)
    parser.add_argument("--out-prefix", default="anphy_spatial")
    parser.add_argument(
        "--include-peripheral",
        action="store_true",
        help="Include periocular/inferior electrodes in topographic and regional summaries.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = cfg["dataset_root"]
    table_dir = cfg["project_data_root"] / "tables"
    out_dir = table_dir

    positions = load_positions(dataset_root, include_peripheral=args.include_peripheral)
    residuals = read_residuals(table_dir, args.residuals_file)
    retained_channels = set(positions["channel_key"])
    residuals = residuals[residuals["channel_key"].isin(retained_channels)].copy()
    epoch_groups = assign_epoch_groups(residuals, args.low_quantile, args.high_quantile)

    labels = epoch_groups[["subject_id", "night_id", "epoch_idx", "rncp_group", "epoch_rncp_l2", "rncp_channel_count"]]
    work = residuals.merge(labels, on=["subject_id", "night_id", "epoch_idx"], how="inner")
    work = work[work["rncp_group"].isin(["low", "high"])].merge(
        positions[["channel_key", "region", "contrast_group"]], on="channel_key", how="inner"
    )

    channel_subject = high_low_delta(work, ["subject_id", "channel_key"], TOPO_COLS)
    channel_summary = summarize_channel_deltas(channel_subject, positions)

    region_work = work[work["region"] != "Other"].copy()
    region_subject = high_low_delta(region_work, ["subject_id", "region", "contrast_group"], TOPO_COLS)
    region_summary = summarize_region_deltas(region_subject)
    contrast = frontoparietal_occipital_contrast(region_subject)

    prefix = args.out_prefix
    epoch_groups.to_csv(out_dir / f"{prefix}_epoch_rncp_groups.csv.gz", index=False, compression="gzip")
    channel_subject.to_csv(out_dir / f"{prefix}_channel_subject_deltas.csv", index=False)
    channel_summary.to_csv(out_dir / f"{prefix}_channel_topography_summary.csv", index=False)
    region_subject.to_csv(out_dir / f"{prefix}_region_subject_deltas.csv", index=False)
    region_summary.to_csv(out_dir / f"{prefix}_region_topography_summary.csv", index=False)
    contrast.to_csv(out_dir / f"{prefix}_frontoparietal_occipital_contrast.csv", index=False)
    write_summary(
        out_dir,
        prefix,
        args.residuals_file,
        epoch_groups,
        channel_summary,
        region_summary,
        contrast,
        args.low_quantile,
        args.high_quantile,
    )

    main_contrast = contrast[contrast["metric"] == "rncp_l2_norm"].iloc[0]
    print(f"Subjects: {epoch_groups['subject_id'].nunique()}")
    print(f"High/low channel rows: {len(work)}")
    print(
        "RNCP frontoparietal-minus-occipital contrast: "
        f"mean={main_contrast['contrast_mean']:.4f}, "
        f"t={main_contrast['contrast_t']:.4f}, p={main_contrast['contrast_p']:.4g}"
    )
    print(f"Wrote analysis outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
