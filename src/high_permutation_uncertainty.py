#!/usr/bin/env python3
"""High-permutation RNCP null tests and bootstrap uncertainty intervals."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sleep_edf_rncp_reproducibility import (
    RESIDUAL_COLS,
    empirical_p_high,
    empirical_p_low,
    frobenius_distance,
    group_indices,
    make_subject_folds,
    mean_abs_offdiag,
    permute_residuals_within_groups,
    residual_corr,
    upper_triangle,
    vector_similarity,
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    display: str
    residual_path: str
    seed: int


SPECS = {
    "sleep_edf_sc": DatasetSpec(
        dataset="sleep_edf_sc",
        display="Sleep-EDF SC",
        residual_path="sleep_edf_sc_n3_rncp_residuals.csv.gz",
        seed=20260510,
    ),
    "sleep_edf_st": DatasetSpec(
        dataset="sleep_edf_st",
        display="Sleep-EDF ST",
        residual_path="sleep_edf_st_n3_rncp_residuals.csv.gz",
        seed=20260511,
    ),
    "anphy": DatasetSpec(
        dataset="anphy",
        display="ANPHY-Sleep",
        residual_path="anphy_sleep_n3_rncp_residuals.csv.gz",
        seed=20260512,
    ),
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan, np.nan
    return (
        float(np.quantile(valid, alpha / 2.0)),
        float(np.quantile(valid, 1.0 - alpha / 2.0)),
    )


def observed_global_stat(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    corr = residual_corr(df)
    vec = upper_triangle(corr)
    return corr, vec, mean_abs_offdiag(vec)


def run_global_null_high_permutation(df: pd.DataFrame, n_null: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    groups = group_indices(df, ["subject_id", "night_id", "channel"])
    _corr, obs_vec, obs_mean_abs = observed_global_stat(df)

    rows = []
    for i in range(n_null):
        shuffled = permute_residuals_within_groups(df, groups, rng)
        null_vec = upper_triangle(residual_corr(shuffled))
        rows.append(
            {
                "iteration": i,
                "mean_abs_offdiag_corr": mean_abs_offdiag(null_vec),
                **{f"corr_{key}": value for key, value in null_vec.items()},
            }
        )
    null_df = pd.DataFrame(rows)
    null_values = null_df["mean_abs_offdiag_corr"].to_numpy(dtype=float)
    null_ci_low, null_ci_high = ci(null_values)
    summary = {
        "observed_mean_abs_offdiag_corr": obs_mean_abs,
        "null_iterations": int(n_null),
        "null_mean_abs_offdiag_corr_mean": float(np.nanmean(null_values)),
        "null_mean_abs_offdiag_corr_sd": float(np.nanstd(null_values, ddof=1)),
        "null_mean_abs_offdiag_corr_ci_low": null_ci_low,
        "null_mean_abs_offdiag_corr_ci_high": null_ci_high,
        "null_mean_abs_offdiag_corr_p_high": empirical_p_high(obs_mean_abs, null_values),
    }
    return null_df, summary


def subject_index_map(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(subject): idx.to_numpy()
        for subject, idx in df.groupby("subject_id", sort=False).groups.items()
    }


def bootstrap_subject_indices(subject_to_idx: dict[str, np.ndarray], subjects: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sampled = rng.choice(np.asarray(subjects, dtype=object), size=len(subjects), replace=True)
    return np.concatenate([subject_to_idx[str(subject)] for subject in sampled])


def bootstrap_global_stat(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subject_to_idx = subject_index_map(df)
    subjects = np.asarray(sorted(subject_to_idx), dtype=object)
    rows = []
    for i in range(n_boot):
        idx = bootstrap_subject_indices(subject_to_idx, subjects, rng)
        boot_df = df.iloc[idx]
        _corr, _vec, stat = observed_global_stat(boot_df)
        rows.append({"iteration": i, "bootstrap_mean_abs_offdiag_corr": stat})
    return pd.DataFrame(rows)


def run_fold_high_permutation_and_bootstrap(
    df: pd.DataFrame,
    n_folds: int,
    n_null: int,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng_null = np.random.default_rng(seed)
    rng_boot = np.random.default_rng(seed + 10_000)
    subjects = df["subject_id"].dropna().unique()
    folds = make_subject_folds(subjects, n_folds, seed)
    fold_rows = []
    boot_rows = []

    for fold_idx, heldout_subjects in enumerate(folds, start=1):
        heldout_set = set(heldout_subjects)
        train = df[~df["subject_id"].isin(heldout_set)].copy()
        heldout = df[df["subject_id"].isin(heldout_set)].copy().reset_index(drop=True)
        train_corr = residual_corr(train)
        heldout_corr = residual_corr(heldout)
        train_vec = upper_triangle(train_corr)
        heldout_vec = upper_triangle(heldout_corr)
        obs_similarity = vector_similarity(train_vec, heldout_vec)
        obs_distance = frobenius_distance(train_corr, heldout_corr)

        heldout_groups = group_indices(heldout, ["subject_id", "night_id", "channel"])
        null_similarity = []
        null_distance = []
        for _ in range(n_null):
            shuffled = permute_residuals_within_groups(heldout, heldout_groups, rng_null)
            null_corr = residual_corr(shuffled)
            null_vec = upper_triangle(null_corr)
            null_similarity.append(vector_similarity(train_vec, null_vec))
            null_distance.append(frobenius_distance(train_corr, null_corr))
        null_similarity_arr = np.asarray(null_similarity, dtype=float)
        null_distance_arr = np.asarray(null_distance, dtype=float)

        train_reset = train.reset_index(drop=True)
        train_subject_to_idx = subject_index_map(train_reset)
        heldout_subject_to_idx = subject_index_map(heldout)
        train_subjects = np.asarray(sorted(train_subject_to_idx), dtype=object)
        heldout_subjects_arr = np.asarray(sorted(heldout_subject_to_idx), dtype=object)
        boot_similarity = []
        boot_distance = []
        for boot_idx in range(n_boot):
            train_idx = bootstrap_subject_indices(train_subject_to_idx, train_subjects, rng_boot)
            heldout_idx = bootstrap_subject_indices(heldout_subject_to_idx, heldout_subjects_arr, rng_boot)
            train_boot = train_reset.iloc[train_idx]
            heldout_boot = heldout.iloc[heldout_idx]
            train_boot_corr = residual_corr(train_boot)
            heldout_boot_corr = residual_corr(heldout_boot)
            sim = vector_similarity(upper_triangle(train_boot_corr), upper_triangle(heldout_boot_corr))
            dist = frobenius_distance(train_boot_corr, heldout_boot_corr)
            boot_similarity.append(sim)
            boot_distance.append(dist)
            boot_rows.append(
                {
                    "fold": fold_idx,
                    "iteration": boot_idx,
                    "bootstrap_train_heldout_vector_similarity": sim,
                    "bootstrap_train_heldout_frobenius_distance": dist,
                }
            )

        sim_ci_low, sim_ci_high = ci(np.asarray(boot_similarity, dtype=float))
        dist_ci_low, dist_ci_high = ci(np.asarray(boot_distance, dtype=float))
        fold_rows.append(
            {
                "fold": fold_idx,
                "heldout_subjects": "|".join(map(str, sorted(heldout_set))),
                "n_train_subjects": int(train["subject_id"].nunique()),
                "n_heldout_subjects": int(heldout["subject_id"].nunique()),
                "n_train_rows": int(len(train)),
                "n_heldout_rows": int(len(heldout)),
                "observed_train_heldout_vector_similarity": obs_similarity,
                "bootstrap_similarity_ci_low": sim_ci_low,
                "bootstrap_similarity_ci_high": sim_ci_high,
                "null_iterations": int(n_null),
                "null_similarity_mean": float(np.nanmean(null_similarity_arr)),
                "null_similarity_sd": float(np.nanstd(null_similarity_arr, ddof=1)),
                "similarity_p_high": empirical_p_high(obs_similarity, null_similarity_arr),
                "observed_train_heldout_frobenius_distance": obs_distance,
                "bootstrap_distance_ci_low": dist_ci_low,
                "bootstrap_distance_ci_high": dist_ci_high,
                "null_distance_mean": float(np.nanmean(null_distance_arr)),
                "null_distance_sd": float(np.nanstd(null_distance_arr, ddof=1)),
                "distance_p_low": empirical_p_low(obs_distance, null_distance_arr),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(boot_rows)


def write_summary(
    table_dir: Path,
    spec: DatasetSpec,
    df: pd.DataFrame,
    global_summary: dict[str, float],
    global_boot: pd.DataFrame,
    fold_df: pd.DataFrame,
    n_boot: int,
) -> None:
    boot_values = global_boot["bootstrap_mean_abs_offdiag_corr"].to_numpy(dtype=float)
    boot_ci_low, boot_ci_high = ci(boot_values)
    fold_values = fold_df["observed_train_heldout_vector_similarity"].to_numpy(dtype=float)
    rng = np.random.default_rng(spec.seed + 99_000)
    fold_mean_boot = np.asarray(
        [np.nanmean(rng.choice(fold_values, size=len(fold_values), replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    fold_mean_ci_low, fold_mean_ci_high = ci(fold_mean_boot)

    lines = [
        f"# High-Permutation RNCP Uncertainty: {spec.display}",
        "",
        "## Design",
        "",
        f"- Rows tested: {len(df)}",
        f"- Subjects: {df['subject_id'].nunique()}",
        f"- Nights: {df['night_id'].nunique()}",
        f"- Global null iterations: {int(global_summary['null_iterations'])}",
        f"- Fold null iterations per fold: {int(fold_df['null_iterations'].iloc[0]) if len(fold_df) else 0}",
        f"- Subject-level bootstrap iterations: {n_boot}",
        "",
        "Null residual columns were independently permuted within `subject_id + night_id + channel` blocks, matching the primary reproducibility test.",
        "Bootstrap intervals resampled subjects with replacement, preserving each selected subject's rows.",
        "",
        "## Global residual-structure statistic",
        "",
        f"- Observed mean absolute off-diagonal residual correlation: {global_summary['observed_mean_abs_offdiag_corr']:.4f}",
        f"- Subject-bootstrap 95% CI: {boot_ci_low:.4f} to {boot_ci_high:.4f}",
        f"- Null mean: {global_summary['null_mean_abs_offdiag_corr_mean']:.4f}",
        f"- Null 95% interval: {global_summary['null_mean_abs_offdiag_corr_ci_low']:.4f} to {global_summary['null_mean_abs_offdiag_corr_ci_high']:.4f}",
        f"- Empirical p with {int(global_summary['null_iterations'])} permutations: {global_summary['null_mean_abs_offdiag_corr_p_high']:.5f}",
        "",
        "## Subject-held-out folds",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Fold mean uncertainty",
        "",
        f"- Mean observed fold similarity: {np.nanmean(fold_values):.4f}",
        f"- Bootstrap 95% CI for mean fold similarity by resampling folds: {fold_mean_ci_low:.4f} to {fold_mean_ci_high:.4f}",
        "",
        "## Interpretation",
        "",
        "The high-permutation analysis refines the p-value floor from 0.001 to 0.0001 for the global statistic when no null iteration reaches the observed value. Bootstrap intervals quantify sampling uncertainty in the observed statistic and held-out fold similarities.",
    ]
    (table_dir / f"high_permutation_{spec.dataset}_high_permutation_uncertainty_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_dataset(cfg: dict, spec: DatasetSpec, n_folds: int, n_null: int, n_boot: int) -> None:
    table_dir = cfg["project_data_root"] / "tables"
    started = time.time()
    df = pd.read_csv(table_dir / spec.residual_path, low_memory=False)
    df = df.dropna(subset=["subject_id", "night_id", "channel"] + RESIDUAL_COLS).reset_index(drop=True)

    null_df, global_summary = run_global_null_high_permutation(df, n_null=n_null, seed=spec.seed)
    global_boot = bootstrap_global_stat(df, n_boot=n_boot, seed=spec.seed + 1)
    boot_ci_low, boot_ci_high = ci(global_boot["bootstrap_mean_abs_offdiag_corr"].to_numpy(dtype=float))
    global_summary["bootstrap_iterations"] = int(n_boot)
    global_summary["observed_bootstrap_ci_low"] = boot_ci_low
    global_summary["observed_bootstrap_ci_high"] = boot_ci_high

    fold_df, fold_boot = run_fold_high_permutation_and_bootstrap(
        df,
        n_folds=n_folds,
        n_null=n_null,
        n_boot=n_boot,
        seed=spec.seed + 1,
    )

    prefix = f"high_permutation_{spec.dataset}_high_permutation"
    pd.DataFrame([global_summary]).to_csv(table_dir / f"{prefix}_global_summary.csv", index=False)
    null_df.to_csv(table_dir / f"{prefix}_global_null_iterations.csv", index=False)
    global_boot.to_csv(table_dir / f"{prefix}_global_bootstrap.csv", index=False)
    fold_df.to_csv(table_dir / f"{prefix}_fold_reproducibility.csv", index=False)
    fold_boot.to_csv(table_dir / f"{prefix}_fold_bootstrap.csv", index=False)
    write_summary(table_dir, spec, df, global_summary, global_boot, fold_df, n_boot=n_boot)

    print(f"{spec.dataset} analysis complete in {time.time() - started:.1f}s", flush=True)
    print(pd.DataFrame([global_summary]).to_string(index=False), flush=True)
    print(fold_df.to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS) + ["all"], default="sleep_edf_sc")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(cfg, SPECS[dataset], n_folds=args.n_folds, n_null=args.n_null, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
