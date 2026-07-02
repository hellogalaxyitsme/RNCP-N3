#!/usr/bin/env python3
"""Within-bout RNCP temporal dynamics.

this analysis uses epoch-level RNCP anchor frames to summarize how RNCP
magnitude evolves across normalized N3 bout time. The unit of averaging is the
N3 bout: epoch values are first binned within each bout, then bout-bin means
are averaged across bouts and subjects.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    anchor_file: str


SPECS = [
    DatasetSpec("sleep_edf_sc", "Sleep-EDF SC", "functional_sleep_edf_sc_epoch_anchor_frame.csv.gz"),
    DatasetSpec("sleep_edf_st", "Sleep-EDF ST", "functional_sleep_edf_st_epoch_anchor_frame.csv.gz"),
    DatasetSpec("anphy", "ANPHY", "functional_anphy_epoch_anchor_frame.csv.gz"),
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def terminal_category(stage: object) -> str:
    value = str(stage).strip().upper()
    if value in {"N1", "N2", "W", "WAKE"}:
        return "lighter_or_wake"
    if value == "REM":
        return "rem"
    if value in {"", "NAN", "NONE"}:
        return "unknown"
    return value.lower()


def one_sample(values: pd.Series) -> dict:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return {"n": int(len(clean)), "mean": np.nan, "sd": np.nan, "sem": np.nan, "t": np.nan, "p": np.nan}
    t_stat, p_val = stats.ttest_1samp(clean.to_numpy(), popmean=0.0, nan_policy="omit")
    return {
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "sd": float(clean.std(ddof=1)),
        "sem": float(clean.sem(ddof=1)),
        "t": float(t_stat),
        "p": float(p_val),
    }


def load_anchor(table_dir: Path, spec: DatasetSpec, min_bout_min: float) -> pd.DataFrame:
    path = table_dir / spec.anchor_file
    if not path.exists():
        raise FileNotFoundError(f"Missing anchor frame: {path}")
    usecols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "rncp_epoch_mean",
        "rncp_epoch_z",
        "next_stage",
        "time_to_n3_exit_min",
        "n3_bout_num",
        "n3_bout_cluster",
        "n3_bout_duration_min",
        "position_within_bout_fraction",
    ]
    df = pd.read_csv(path, usecols=lambda col: col in usecols, low_memory=False)
    df["dataset"] = spec.key
    df["dataset_label"] = spec.label
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    for col in ["epoch_idx", "rncp_epoch_mean", "rncp_epoch_z", "time_to_n3_exit_min", "n3_bout_num", "n3_bout_duration_min", "position_within_bout_fraction"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "n3_bout_cluster" not in df or df["n3_bout_cluster"].isna().all():
        bout = df["n3_bout_num"].round().astype("Int64").astype(str)
        df["n3_bout_cluster"] = df["subject_id"] + ":" + df["night_id"] + ":bout" + bout
    df = df.dropna(subset=["rncp_epoch_z", "position_within_bout_fraction", "n3_bout_duration_min", "n3_bout_cluster"]).copy()
    df = df[df["n3_bout_duration_min"] >= min_bout_min].copy()
    df["position_within_bout_fraction"] = df["position_within_bout_fraction"].clip(0, 1)
    return df


def add_bout_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["n3_bout_cluster", "epoch_idx"]).copy()
    last_rows = ordered.groupby("n3_bout_cluster", as_index=False).tail(1)
    bout_meta = last_rows[
        [
            "dataset",
            "dataset_label",
            "subject_id",
            "night_id",
            "n3_bout_cluster",
            "n3_bout_duration_min",
            "next_stage",
        ]
    ].copy()
    bout_meta = bout_meta.rename(columns={"next_stage": "terminal_stage"})
    bout_meta["terminal_category"] = bout_meta["terminal_stage"].map(terminal_category)
    bout_meta["terminal_category_label"] = bout_meta["terminal_category"].map(
        {
            "lighter_or_wake": "N1/N2/W",
            "rem": "REM",
            "unknown": "Unknown",
        }
    ).fillna(bout_meta["terminal_category"])
    df = df.merge(
        bout_meta[["n3_bout_cluster", "terminal_stage", "terminal_category", "terminal_category_label"]],
        on="n3_bout_cluster",
        how="left",
        validate="many_to_one",
    )
    return df, bout_meta


def bin_trajectories(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    labels = list(range(1, n_bins + 1))
    df = df.copy()
    df["position_bin"] = pd.cut(
        df["position_within_bout_fraction"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype("Int64")
    df["position_bin_mid"] = df["position_bin"].astype(float).map(lambda b: (b - 0.5) / n_bins)

    bout_bin = (
        df.dropna(subset=["position_bin"])
        .groupby(
            [
                "dataset",
                "dataset_label",
                "subject_id",
                "night_id",
                "n3_bout_cluster",
                "n3_bout_duration_min",
                "terminal_stage",
                "terminal_category",
                "terminal_category_label",
                "position_bin",
                "position_bin_mid",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            rncp_epoch_z_mean=("rncp_epoch_z", "mean"),
            rncp_epoch_mean=("rncp_epoch_mean", "mean"),
            time_to_n3_exit_min_mean=("time_to_n3_exit_min", "mean"),
            epoch_count=("epoch_idx", "count"),
        )
    )
    return bout_bin


def summarize_trajectory(bout_bin: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "dataset_label", "terminal_category", "terminal_category_label", "position_bin", "position_bin_mid"]
    for keys, part in bout_bin.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        vals = pd.to_numeric(part["rncp_epoch_z_mean"], errors="coerce").dropna()
        row.update(
            {
                "n_bouts": int(vals.shape[0]),
                "n_subjects": int(part["subject_id"].nunique()),
                "rncp_epoch_z_mean": float(vals.mean()) if len(vals) else np.nan,
                "rncp_epoch_z_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "rncp_epoch_z_sem": float(vals.sem(ddof=1)) if len(vals) > 1 else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def ramp_stats(bout_bin: pd.DataFrame) -> pd.DataFrame:
    early = bout_bin[bout_bin["position_bin"].astype(int).between(1, 2)]
    mid = bout_bin[bout_bin["position_bin"].astype(int).between(5, 6)]
    late = bout_bin[bout_bin["position_bin"].astype(int).between(9, 10)]
    pre_exit = bout_bin[bout_bin["position_bin"].astype(int).between(7, 8)]
    windows = {
        "early_0_20": early,
        "middle_40_60": mid,
        "pre_exit_60_80": pre_exit,
        "late_80_100": late,
    }
    wide = None
    keys = ["dataset", "dataset_label", "subject_id", "night_id", "n3_bout_cluster", "terminal_category", "terminal_category_label"]
    for name, part in windows.items():
        agg = part.groupby(keys, as_index=False, observed=True)["rncp_epoch_z_mean"].mean().rename(columns={"rncp_epoch_z_mean": name})
        wide = agg if wide is None else wide.merge(agg, on=keys, how="outer")
    wide["late_minus_middle"] = wide["late_80_100"] - wide["middle_40_60"]
    wide["late_minus_early"] = wide["late_80_100"] - wide["early_0_20"]
    wide["pre_exit_minus_middle"] = wide["pre_exit_60_80"] - wide["middle_40_60"]

    rows = []
    for keys_group, part in wide.groupby(["dataset", "dataset_label", "terminal_category", "terminal_category_label"], observed=True):
        row_base = dict(zip(["dataset", "dataset_label", "terminal_category", "terminal_category_label"], keys_group))
        for contrast in ["pre_exit_minus_middle", "late_minus_middle", "late_minus_early"]:
            stat = one_sample(part[contrast])
            rows.append({**row_base, "contrast": contrast, **stat})
    return wide, pd.DataFrame(rows)


def bout_level_summary(bout_bin: pd.DataFrame) -> pd.DataFrame:
    bout = (
        bout_bin.groupby(
            ["dataset", "dataset_label", "subject_id", "night_id", "n3_bout_cluster", "terminal_category", "terminal_category_label"],
            as_index=False,
            observed=True,
        )
        .agg(
            bout_rncp_mean_z=("rncp_epoch_z_mean", "mean"),
            bout_rncp_sd_z=("rncp_epoch_z_mean", "std"),
            n3_bout_duration_min=("n3_bout_duration_min", "first"),
            covered_bins=("position_bin", "nunique"),
        )
    )
    rows = []
    for keys, part in bout.groupby(["dataset", "dataset_label", "terminal_category", "terminal_category_label"], observed=True):
        vals = part["bout_rncp_mean_z"].dropna()
        rows.append(
            {
                "dataset": keys[0],
                "dataset_label": keys[1],
                "terminal_category": keys[2],
                "terminal_category_label": keys[3],
                "n_bouts": int(len(vals)),
                "n_subjects": int(part["subject_id"].nunique()),
                "bout_rncp_mean_z_mean": float(vals.mean()) if len(vals) else np.nan,
                "bout_rncp_mean_z_sd": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "bout_duration_min_median": float(part["n3_bout_duration_min"].median()) if len(part) else np.nan,
            }
        )
    return bout, pd.DataFrame(rows)


def write_summary(out_dir: Path, trajectory: pd.DataFrame, ramp_summary: pd.DataFrame, bout_summary: pd.DataFrame, min_bout_min: float) -> None:
    lines = [
        "# Within-Bout RNCP Dynamics",
        "",
        f"Minimum N3 bout duration: {min_bout_min:g} min",
        "",
        "## Bout counts by dataset and terminal category",
        "",
        bout_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Ramp statistics",
        "",
        ramp_summary.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        "- `within_bout_bout_bin_values.csv.gz`",
        "- `within_bout_trajectory_summary.csv`",
        "- `within_bout_ramp_by_bout.csv`",
        "- `within_bout_ramp_statistics.csv`",
        "- `within_bout_bout_level_values.csv`",
        "- `within_bout_bout_level_summary.csv`",
    ]
    (out_dir / "within_bout_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--min-bout-min", type=float, default=5.0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--out-prefix", default="within_bout")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    frames = []
    bout_meta_frames = []
    for spec in SPECS:
        df = load_anchor(table_dir, spec, args.min_bout_min)
        df, bout_meta = add_bout_metadata(df)
        frames.append(df)
        bout_meta_frames.append(bout_meta)
        print(f"{spec.label}: epochs={len(df)} bouts={df['n3_bout_cluster'].nunique()}", flush=True)
    all_epochs = pd.concat(frames, ignore_index=True)

    bout_bin = bin_trajectories(all_epochs, args.n_bins)
    trajectory = summarize_trajectory(bout_bin)
    ramp_by_bout, ramp_summary = ramp_stats(bout_bin)
    bout_values, bout_summary = bout_level_summary(bout_bin)

    prefix = args.out_prefix
    bout_bin.to_csv(table_dir / f"{prefix}_bout_bin_values.csv.gz", index=False, compression="gzip")
    trajectory.to_csv(table_dir / f"{prefix}_trajectory_summary.csv", index=False)
    ramp_by_bout.to_csv(table_dir / f"{prefix}_ramp_by_bout.csv", index=False)
    ramp_summary.to_csv(table_dir / f"{prefix}_ramp_statistics.csv", index=False)
    bout_values.to_csv(table_dir / f"{prefix}_bout_level_values.csv", index=False)
    bout_summary.to_csv(table_dir / f"{prefix}_bout_level_summary.csv", index=False)
    write_summary(table_dir, trajectory, ramp_summary, bout_summary, args.min_bout_min)

    print(ramp_summary.to_string(index=False))
    print(f"Wrote analysis outputs to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
