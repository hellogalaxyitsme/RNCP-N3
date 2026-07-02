#!/usr/bin/env python3
"""ANPHY held-out fold quality-control analysis."""

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
EDGE_LABELS = ["LZc-PE", "LZc-SE", "LZc-AE", "PE-SE", "PE-AE", "SE-AE"]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def upper_edges(df: pd.DataFrame) -> list[float]:
    corr = df[RESIDUAL_COLS].corr()
    out = []
    for i, left in enumerate(RESIDUAL_COLS):
        for right in RESIDUAL_COLS[i + 1 :]:
            out.append(float(corr.loc[left, right]))
    return out


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    sd = vals.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (vals - vals.mean()) / sd


def load_fold_table(table_dir: Path, scenario: str) -> pd.DataFrame:
    candidates = [
        table_dir / "artifact_anphy_artifact_adjusted_fold_reproducibility.csv",
        table_dir / "anphy_rncp_reproducibility_fold_reproducibility.csv",
    ]
    if scenario == "baseline":
        candidates = [table_dir / "anphy_rncp_reproducibility_fold_reproducibility.csv"]
    for path in candidates:
        if path.exists():
            fold = pd.read_csv(path)
            if "observed_similarity" in fold and "observed_train_heldout_vector_similarity" not in fold:
                fold = fold.rename(columns={"observed_similarity": "observed_train_heldout_vector_similarity"})
            return fold
    raise FileNotFoundError("No ANPHY fold reproducibility table found.")


def load_residuals(table_dir: Path, scenario: str) -> pd.DataFrame:
    filename = "artifact_anphy_artifact_adjusted_rncp_residuals.csv.gz" if scenario == "artifact_adjusted" else "anphy_sleep_n3_rncp_residuals.csv.gz"
    path = table_dir / filename
    if not path.exists():
        raise FileNotFoundError(path)
    usecols = ["subject_id", "night_id", "epoch_idx", "channel", "rncp_l2_norm", *RESIDUAL_COLS]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    return df.dropna(subset=RESIDUAL_COLS).copy()


def per_subject_residual_summary(residuals: pd.DataFrame, full_vec: np.ndarray) -> pd.DataFrame:
    rows = []
    for subject_id, part in residuals.groupby("subject_id", sort=True):
        edges = np.asarray(upper_edges(part), dtype=float)
        profile_similarity = stats.pearsonr(edges, full_vec)[0] if np.isfinite(edges).all() else np.nan
        row = {
            "subject_id": subject_id,
            "residual_rows": int(len(part)),
            "n_epochs_residual": int(part["epoch_idx"].nunique()),
            "n_channels_residual": int(part["channel"].nunique()),
            "mean_rncp_l2_norm": float(pd.to_numeric(part["rncp_l2_norm"], errors="coerce").mean()),
            "profile_similarity_to_full_anphy": float(profile_similarity),
        }
        for label, value in zip(EDGE_LABELS, edges):
            row[f"corr_{label}"] = value
            row[f"abs_dev_full_{label}"] = abs(value - full_vec[EDGE_LABELS.index(label)])
        row["mean_abs_edge_deviation_from_full"] = float(np.nanmean([row[f"abs_dev_full_{label}"] for label in EDGE_LABELS]))
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_summary(table_dir: Path) -> pd.DataFrame:
    cohort = pd.read_csv(table_dir / "anphy_sleep_cohort_table.csv")
    cohort["subject_id"] = cohort["subject_id"].astype(str)
    numeric = [
        "age",
        "total_n3_min",
        "total_sleep_min",
        "annotation_duration_min",
        "eeg_position_channel_count",
    ]
    for col in numeric:
        if col in cohort:
            cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
    return cohort


def metadata_summary(table_dir: Path) -> pd.DataFrame:
    path = table_dir / "anphy_sleep_epoch_metadata.csv.gz"
    usecols = [
        "subject_id",
        "night_id",
        "epoch_idx",
        "stage",
        "after_sleep_onset",
        "stage_is_sleep",
        "stage_is_n3",
        "n3_bout_num",
        "n3_bout_duration_min",
        "relative_delta_power",
        "slow_wave_density",
    ]
    meta = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["stage"] = meta["stage"].astype(str)
    for col in ["n3_bout_num", "n3_bout_duration_min", "relative_delta_power", "slow_wave_density"]:
        if col in meta:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")
    meta["stage_is_n3_bool"] = meta["stage"].str.upper().eq("N3")
    n3 = meta[meta["stage_is_n3_bool"]].copy()
    n3_bouts = n3.groupby("subject_id")["n3_bout_num"].nunique().rename("n3_bout_count")
    summary = (
        n3.groupby("subject_id", as_index=False)
        .agg(
            n3_epochs_metadata=("epoch_idx", "nunique"),
            median_n3_bout_duration_min=("n3_bout_duration_min", "median"),
            mean_relative_delta_power=("relative_delta_power", "mean"),
            mean_slow_wave_density=("slow_wave_density", "mean"),
        )
        .merge(n3_bouts.reset_index(), on="subject_id", how="left")
    )
    summary["n3_duration_min_metadata"] = summary["n3_epochs_metadata"] * 0.5
    summary["n3_fragmentation_per_hour"] = summary["n3_bout_count"] / (summary["n3_duration_min_metadata"] / 60.0)
    return summary


def artifact_summary(table_dir: Path) -> pd.DataFrame:
    path = table_dir / "artifact_anphy_artifact_covariates.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=["subject_id"])
    usecols = [
        "subject_id",
        "epoch_idx",
        "channel_key",
        "eeg_hf_30_45_ratio",
        "eeg_hf_35_45_ratio",
        "eeg_beta_20_30_power",
        "anphy_artifact_fraction_channels",
        "eog_rms",
        "emg_rms",
    ]
    art = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    art["subject_id"] = art["subject_id"].astype(str)
    for col in usecols[3:]:
        if col in art:
            art[col] = pd.to_numeric(art[col], errors="coerce")
    return art.groupby("subject_id", as_index=False).agg(
        artifact_rows=("epoch_idx", "count"),
        artifact_epochs=("epoch_idx", "nunique"),
        artifact_channels=("channel_key", "nunique"),
        mean_hf_30_45_ratio=("eeg_hf_30_45_ratio", "mean"),
        mean_hf_35_45_ratio=("eeg_hf_35_45_ratio", "mean"),
        mean_beta_20_30_power=("eeg_beta_20_30_power", "mean"),
        mean_anphy_artifact_fraction=("anphy_artifact_fraction_channels", "mean"),
        mean_eog_rms=("eog_rms", "mean"),
        mean_emg_rms=("emg_rms", "mean"),
    )


def build_quality_control(table_dir: Path, scenario: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold = load_fold_table(table_dir, scenario)
    fold["heldout_subject_list"] = fold["heldout_subjects"].astype(str).str.split("|")
    lowest_similarity_fold = int(fold.loc[fold["observed_train_heldout_vector_similarity"].idxmin(), "fold"])
    fold["fold_status"] = np.where(fold["fold"].astype(int) == lowest_similarity_fold, "lowest_similarity", "reference")

    membership_rows = []
    for _, row in fold.iterrows():
        for subject_id in row["heldout_subject_list"]:
            membership_rows.append(
                {
                    "subject_id": subject_id,
                    "fold": int(row["fold"]),
                    "fold_status": row["fold_status"],
                    "fold_similarity": float(row["observed_train_heldout_vector_similarity"]),
                    "fold_heldout_rows": int(row["n_heldout_rows"]),
                }
            )
    membership = pd.DataFrame(membership_rows)

    residuals = load_residuals(table_dir, scenario)
    full_vec = np.asarray(upper_edges(residuals), dtype=float)
    subj_resid = per_subject_residual_summary(residuals, full_vec)
    subject = membership.merge(cohort_summary(table_dir), on="subject_id", how="left")
    subject = subject.merge(metadata_summary(table_dir), on="subject_id", how="left")
    subject = subject.merge(artifact_summary(table_dir), on="subject_id", how="left")
    subject = subject.merge(subj_resid, on="subject_id", how="left")

    qc_cols = [
        "age",
        "total_n3_min",
        "total_sleep_min",
        "eeg_position_channel_count",
        "n3_bout_count",
        "median_n3_bout_duration_min",
        "n3_fragmentation_per_hour",
        "mean_relative_delta_power",
        "mean_slow_wave_density",
        "mean_hf_30_45_ratio",
        "mean_hf_35_45_ratio",
        "mean_anphy_artifact_fraction",
        "mean_eog_rms",
        "mean_emg_rms",
        "residual_rows",
        "n_epochs_residual",
        "n_channels_residual",
        "mean_rncp_l2_norm",
        "profile_similarity_to_full_anphy",
        "mean_abs_edge_deviation_from_full",
    ]
    for col in qc_cols:
        if col in subject:
            subject[f"{col}_z"] = zscore(subject[col])
    abs_z_cols = [f"{col}_z" for col in qc_cols if f"{col}_z" in subject]
    subject["max_abs_qc_z"] = subject[abs_z_cols].abs().max(axis=1)
    subject["outlier_flags_abs_z_ge_2"] = subject.apply(
        lambda row: ";".join(col.replace("_z", "") for col in abs_z_cols if pd.notna(row[col]) and abs(row[col]) >= 2),
        axis=1,
    )

    fold_diag = (
        subject.groupby(["fold", "fold_status"], as_index=False)
        .agg(
            n_subjects=("subject_id", "nunique"),
            mean_age=("age", "mean"),
            mean_total_n3_min=("total_n3_min", "mean"),
            mean_n3_fragmentation_per_hour=("n3_fragmentation_per_hour", "mean"),
            mean_hf_30_45_ratio=("mean_hf_30_45_ratio", "mean"),
            mean_anphy_artifact_fraction=("mean_anphy_artifact_fraction", "mean"),
            mean_profile_similarity_to_full=("profile_similarity_to_full_anphy", "mean"),
            mean_abs_edge_deviation_from_full=("mean_abs_edge_deviation_from_full", "mean"),
            max_abs_qc_z=("max_abs_qc_z", "max"),
        )
        .merge(
            fold[[
                "fold",
                "heldout_subjects",
                "observed_train_heldout_vector_similarity",
                "observed_train_heldout_frobenius_distance",
                "similarity_p_high",
                "distance_p_low",
            ]],
            on="fold",
            how="left",
        )
    )

    pairwise = residual_pairwise_by_fold(residuals, fold)
    return fold, subject, fold_diag, pairwise


def residual_pairwise_by_fold(residuals: pd.DataFrame, fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in fold.iterrows():
        heldout = set(str(row["heldout_subjects"]).split("|"))
        train = residuals[~residuals["subject_id"].isin(heldout)]
        held = residuals[residuals["subject_id"].isin(heldout)]
        train_edges = upper_edges(train)
        held_edges = upper_edges(held)
        for label, train_value, held_value in zip(EDGE_LABELS, train_edges, held_edges):
            rows.append(
                {
                    "fold": int(row["fold"]),
                    "fold_similarity": float(row["observed_train_heldout_vector_similarity"]),
                    "pair": label,
                    "train_correlation": train_value,
                    "heldout_correlation": held_value,
                    "absolute_difference": abs(held_value - train_value),
                    "same_sign": np.sign(train_value) == np.sign(held_value),
                }
            )
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, fold: pd.DataFrame, subject: pd.DataFrame, fold_qc: pd.DataFrame, pairwise: pd.DataFrame, prefix: str) -> None:
    lowest_similarity_fold = int(fold.loc[fold["fold_status"].eq("lowest_similarity"), "fold"].iloc[0])
    lowest_similarity_subjects = subject[subject["fold"] == lowest_similarity_fold].sort_values("max_abs_qc_z", ascending=False)
    lines = [
        "# ANPHY Fold Quality Control",
        "",
        f"Lowest-similarity fold: {lowest_similarity_fold}",
        f"Held-out subjects: {fold.loc[fold['fold'].eq(lowest_similarity_fold), 'heldout_subjects'].iloc[0]}",
        "",
        "## Fold reproducibility",
        "",
        fold[[
            "fold",
            "heldout_subjects",
            "observed_train_heldout_vector_similarity",
            "observed_train_heldout_frobenius_distance",
            "similarity_p_high",
            "distance_p_low",
            "fold_status",
        ]].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Fold-level quality control",
        "",
        fold_qc.to_markdown(index=False, floatfmt=".4g"),
        "",
        f"## Lowest-similarity fold {lowest_similarity_fold} subject quality control",
        "",
        lowest_similarity_subjects[[
            "subject_id",
            "age",
            "sex",
            "total_n3_min",
            "n3_fragmentation_per_hour",
            "mean_hf_30_45_ratio",
            "mean_anphy_artifact_fraction",
            "profile_similarity_to_full_anphy",
            "mean_abs_edge_deviation_from_full",
            "max_abs_qc_z",
            "outlier_flags_abs_z_ge_2",
        ]].to_markdown(index=False, floatfmt=".4g"),
        "",
        f"## Lowest-similarity fold {lowest_similarity_fold} pairwise profile",
        "",
        pairwise[pairwise["fold"] == lowest_similarity_fold].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Outputs",
        "",
        f"- `{prefix}_fold_reproducibility.csv`",
        f"- `{prefix}_subject_quality_control.csv`",
        f"- `{prefix}_fold_quality_control.csv`",
        f"- `{prefix}_pairwise_fold_profiles.csv`",
    ]
    (out_dir / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", choices=["artifact_adjusted", "baseline"], default="artifact_adjusted")
    parser.add_argument("--out-prefix", default="anphy_fold_quality_control")
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    fold, subject, fold_qc, pairwise = build_quality_control(table_dir, args.scenario)
    prefix = args.out_prefix

    fold.to_csv(table_dir / f"{prefix}_fold_reproducibility.csv", index=False)
    subject.to_csv(table_dir / f"{prefix}_subject_quality_control.csv", index=False)
    fold_qc.to_csv(table_dir / f"{prefix}_fold_quality_control.csv", index=False)
    pairwise.to_csv(table_dir / f"{prefix}_pairwise_fold_profiles.csv", index=False)
    write_summary(table_dir, fold, subject, fold_qc, pairwise, prefix)

    lowest_similarity_fold = int(fold.loc[fold["fold_status"].eq("lowest_similarity"), "fold"].iloc[0])
    print(f"Lowest-similarity fold: {lowest_similarity_fold}")
    print(fold[["fold", "heldout_subjects", "observed_train_heldout_vector_similarity", "fold_status"]].to_string(index=False))
    print(subject[subject["fold"] == lowest_similarity_fold][["subject_id", "age", "total_n3_min", "mean_hf_30_45_ratio", "mean_anphy_artifact_fraction", "profile_similarity_to_full_anphy", "mean_abs_edge_deviation_from_full", "outlier_flags_abs_z_ge_2"]].to_string(index=False))
    print(f"Wrote analysis outputs to {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
