#!/usr/bin/env python3
"""Sleep-EDF SC night-to-night RNCP stability.

this analysis uses repeated Sleep Cassette recordings to quantify whether RNCP
magnitude and the six-edge RNCP residual-correlation profile are stable from
night 1 to night 2 within the same subject.
"""

from __future__ import annotations

import argparse
import json
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
FEATURE_LABELS = ["LZc", "PE", "SE", "AE"]
EDGE_LABELS = [
    "LZc-PE",
    "LZc-SE",
    "LZc-AE",
    "PE-SE",
    "PE-AE",
    "SE-AE",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def pearson_summary(x: pd.Series, y: pd.Series) -> dict:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return {"n": int(len(frame)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    pr, pp = stats.pearsonr(frame["x"], frame["y"])
    sr, sp = stats.spearmanr(frame["x"], frame["y"])
    return {
        "n": int(len(frame)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def icc_two_way(values: pd.DataFrame) -> dict:
    """Return ICC(2,1), ICC(3,1), and mean-measure variants for two nights."""
    mat = values.dropna().to_numpy(dtype=float)
    n, k = mat.shape if mat.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"icc2_1": np.nan, "icc3_1": np.nan, "icc2_k": np.nan, "icc3_k": np.nan}

    grand = mat.mean()
    subj_mean = mat.mean(axis=1)
    night_mean = mat.mean(axis=0)
    ss_subject = k * np.sum((subj_mean - grand) ** 2)
    ss_night = n * np.sum((night_mean - grand) ** 2)
    ss_error = np.sum((mat - subj_mean[:, None] - night_mean[None, :] + grand) ** 2)
    ms_subject = ss_subject / (n - 1)
    ms_night = ss_night / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc2_1 = (ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error + k * (ms_night - ms_error) / n)
    icc3_1 = (ms_subject - ms_error) / (ms_subject + (k - 1) * ms_error)
    icc2_k = (ms_subject - ms_error) / (ms_subject + (ms_night - ms_error) / n)
    icc3_k = (ms_subject - ms_error) / ms_subject
    return {
        "icc2_1_absolute": float(icc2_1),
        "icc3_1_consistency": float(icc3_1),
        "icc2_k_absolute": float(icc2_k),
        "icc3_k_consistency": float(icc3_k),
    }


def load_night_magnitude(table_dir: Path, min_epochs: int) -> pd.DataFrame:
    path = table_dir / "functional_sleep_edf_sc_epoch_anchor_frame.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    cols = ["subject_id", "night_id", "epoch_idx", "rncp_epoch_mean", "rncp_epoch_z", "age", "sex"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    summary = (
        df.groupby(["subject_id", "night_id"], as_index=False)
        .agg(
            n_epochs=("epoch_idx", "nunique"),
            mean_rncp_magnitude=("rncp_epoch_mean", "mean"),
            median_rncp_magnitude=("rncp_epoch_mean", "median"),
            mean_rncp_z=("rncp_epoch_z", "mean"),
            sd_rncp_z=("rncp_epoch_z", "std"),
            age=("age", "first"),
            sex=("sex", "first"),
        )
    )
    summary = summary[summary["n_epochs"] >= min_epochs].copy()
    summary = summary.sort_values(["subject_id", "night_id"]).reset_index(drop=True)
    summary["night_order"] = summary.groupby("subject_id").cumcount() + 1
    counts = summary.groupby("subject_id")["night_id"].transform("nunique")
    return summary[counts >= 2].copy()


def edge_vector(corr: pd.DataFrame) -> list[float]:
    vals = []
    for i in range(len(RESIDUAL_COLS)):
        for j in range(i + 1, len(RESIDUAL_COLS)):
            vals.append(float(corr.iloc[i, j]))
    return vals


def load_night_corr_vectors(table_dir: Path, paired_subjects: set[str], min_rows: int) -> pd.DataFrame:
    path = table_dir / "sleep_edf_sc_n3_rncp_residuals.csv.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["subject_id", "night_id", "epoch_idx", "channel", *RESIDUAL_COLS]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    df = df[df["subject_id"].isin(paired_subjects)].copy()

    rows = []
    for (subject_id, night_id), part in df.groupby(["subject_id", "night_id"], sort=True):
        clean = part.dropna(subset=RESIDUAL_COLS)
        if len(clean) < min_rows:
            continue
        corr = clean[RESIDUAL_COLS].corr()
        row = {
            "subject_id": subject_id,
            "night_id": night_id,
            "n_rows": int(len(clean)),
            "n_epochs": int(clean["epoch_idx"].nunique()),
            "n_channels": int(clean["channel"].nunique()),
        }
        for label, value in zip(EDGE_LABELS, edge_vector(corr)):
            row[f"corr_{label}"] = value
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["subject_id", "night_id"]).reset_index(drop=True)
    out["night_order"] = out.groupby("subject_id").cumcount() + 1
    counts = out.groupby("subject_id")["night_id"].transform("nunique")
    return out[counts >= 2].copy()


def paired_wide(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    keep = df[df["night_order"].isin([1, 2])].copy()
    wide_parts = []
    for night_order in [1, 2]:
        part = keep[keep["night_order"] == night_order][["subject_id", "night_id", *value_cols]].copy()
        part = part.rename(columns={col: f"{col}_night{night_order}" for col in ["night_id", *value_cols]})
        wide_parts.append(part)
    return wide_parts[0].merge(wide_parts[1], on="subject_id", how="inner")


def magnitude_results(night_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = ["mean_rncp_magnitude", "median_rncp_magnitude", "mean_rncp_z", "sd_rncp_z", "n_epochs"]
    wide = paired_wide(night_summary, value_cols)
    rows = []
    for metric in ["mean_rncp_magnitude", "median_rncp_magnitude", "mean_rncp_z", "sd_rncp_z"]:
        x = wide[f"{metric}_night1"]
        y = wide[f"{metric}_night2"]
        row = {"metric": metric}
        row.update(pearson_summary(x, y))
        row.update(icc_two_way(wide[[f"{metric}_night1", f"{metric}_night2"]]))
        row["night1_mean"] = float(pd.to_numeric(x, errors="coerce").mean())
        row["night2_mean"] = float(pd.to_numeric(y, errors="coerce").mean())
        row["mean_difference_night2_minus_night1"] = float((pd.to_numeric(y, errors="coerce") - pd.to_numeric(x, errors="coerce")).mean())
        rows.append(row)
    return wide, pd.DataFrame(rows)


def corr_vector_results(corr_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edge_cols = [f"corr_{label}" for label in EDGE_LABELS]
    wide = paired_wide(corr_summary, edge_cols + ["n_rows", "n_epochs"])

    sim_rows = []
    for _, row in wide.iterrows():
        v1 = row[[f"{col}_night1" for col in edge_cols]].astype(float).to_numpy()
        v2 = row[[f"{col}_night2" for col in edge_cols]].astype(float).to_numpy()
        pearson = stats.pearsonr(v1, v2)[0] if np.isfinite(v1).all() and np.isfinite(v2).all() else np.nan
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))) if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0 else np.nan
        sim_rows.append(
            {
                "subject_id": row["subject_id"],
                "night_id_night1": row["night_id_night1"],
                "night_id_night2": row["night_id_night2"],
                "profile_pearson_r": float(pearson),
                "profile_cosine_similarity": cosine,
                "profile_euclidean_distance": float(np.linalg.norm(v2 - v1)),
            }
        )
    sim = pd.DataFrame(sim_rows)

    edge_rows = []
    for label, col in zip(EDGE_LABELS, edge_cols):
        x = wide[f"{col}_night1"]
        y = wide[f"{col}_night2"]
        row = {"edge": label, "metric": col}
        row.update(pearson_summary(x, y))
        row.update(icc_two_way(wide[[f"{col}_night1", f"{col}_night2"]]))
        row["night1_mean"] = float(pd.to_numeric(x, errors="coerce").mean())
        row["night2_mean"] = float(pd.to_numeric(y, errors="coerce").mean())
        edge_rows.append(row)
    edge_stats = pd.DataFrame(edge_rows)

    mean_v1 = np.array([wide[f"{col}_night1"].mean() for col in edge_cols], dtype=float)
    mean_v2 = np.array([wide[f"{col}_night2"].mean() for col in edge_cols], dtype=float)
    group_r, group_p = stats.pearsonr(mean_v1, mean_v2)
    group = pd.DataFrame(
        [
            {
                "n_subjects": int(len(wide)),
                "group_mean_vector_pearson_r": float(group_r),
                "group_mean_vector_pearson_p": float(group_p),
                "within_subject_profile_pearson_mean": float(sim["profile_pearson_r"].mean()),
                "within_subject_profile_pearson_median": float(sim["profile_pearson_r"].median()),
                "within_subject_profile_cosine_mean": float(sim["profile_cosine_similarity"].mean()),
                "within_subject_profile_cosine_median": float(sim["profile_cosine_similarity"].median()),
            }
        ]
    )
    return wide, sim, edge_stats, group


def write_summary(
    out_dir: Path,
    mag_stats: pd.DataFrame,
    edge_stats: pd.DataFrame,
    group_profile: pd.DataFrame,
    n_subjects: int,
) -> None:
    lines = [
        "# Sleep-EDF SC Night-to-Night RNCP Stability",
        "",
        f"Paired subjects: {n_subjects}",
        "",
        "## Mean RNCP magnitude test-retest",
        "",
        mag_stats.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Residual-correlation edge test-retest",
        "",
        edge_stats.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Residual-correlation profile similarity",
        "",
        group_profile.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        "- `sc_night_night_subject_summary.csv`",
        "- `sc_night_mean_rncp_paired.csv`",
        "- `sc_night_mean_rncp_testretest.csv`",
        "- `sc_night_residual_corr_vectors.csv`",
        "- `sc_night_residual_corr_paired.csv`",
        "- `sc_night_residual_corr_profile_similarity.csv`",
        "- `sc_night_residual_corr_edge_testretest.csv`",
        "- `sc_night_residual_corr_group_profile.csv`",
    ]
    (out_dir / "sc_night_to_night_stability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--min-residual-rows", type=int, default=60)
    parser.add_argument("--out-prefix", default="sc_night")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    prefix = args.out_prefix

    night_summary = load_night_magnitude(table_dir, args.min_epochs)
    paired_subjects = set(night_summary["subject_id"].unique())
    corr_summary = load_night_corr_vectors(table_dir, paired_subjects, args.min_residual_rows)

    mag_wide, mag_stats = magnitude_results(night_summary)
    corr_wide, profile_similarity, edge_stats, group_profile = corr_vector_results(corr_summary)

    night_summary.to_csv(table_dir / f"{prefix}_night_subject_summary.csv", index=False)
    mag_wide.to_csv(table_dir / f"{prefix}_mean_rncp_paired.csv", index=False)
    mag_stats.to_csv(table_dir / f"{prefix}_mean_rncp_testretest.csv", index=False)
    corr_summary.to_csv(table_dir / f"{prefix}_residual_corr_vectors.csv", index=False)
    corr_wide.to_csv(table_dir / f"{prefix}_residual_corr_paired.csv", index=False)
    profile_similarity.to_csv(table_dir / f"{prefix}_residual_corr_profile_similarity.csv", index=False)
    edge_stats.to_csv(table_dir / f"{prefix}_residual_corr_edge_testretest.csv", index=False)
    group_profile.to_csv(table_dir / f"{prefix}_residual_corr_group_profile.csv", index=False)
    write_summary(table_dir, mag_stats, edge_stats, group_profile, len(mag_wide))

    print(f"Paired subjects with mean RNCP: {len(mag_wide)}")
    print(f"Paired subjects with residual-correlation profiles: {len(corr_wide)}")
    print(mag_stats.to_string(index=False))
    print(group_profile.to_string(index=False))
    print(f"Wrote analysis outputs to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
