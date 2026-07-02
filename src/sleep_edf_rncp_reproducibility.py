#!/usr/bin/env python3
"""subject-held-out RNCP residual reproducibility and null tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "lzc",
    "permutation_entropy",
    "spectral_entropy",
    "aperiodic_exponent_specparam",
]
RESIDUAL_COLS = [f"{feature}_rncp_residual_z" for feature in FEATURES]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def upper_triangle(corr: pd.DataFrame) -> pd.Series:
    values = {}
    for i, left in enumerate(RESIDUAL_COLS):
        for right in RESIDUAL_COLS[i + 1 :]:
            values[f"{left}__{right}"] = float(corr.loc[left, right])
    return pd.Series(values)


def residual_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df[RESIDUAL_COLS].corr()


def vector_similarity(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    if joined.iloc[:, 0].std(ddof=0) == 0 or joined.iloc[:, 1].std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])


def frobenius_distance(left: pd.DataFrame, right: pd.DataFrame) -> float:
    return float(np.linalg.norm(left.to_numpy() - right.to_numpy(), ord="fro"))


def mean_abs_offdiag(vec: pd.Series) -> float:
    return float(np.nanmean(np.abs(vec.to_numpy(dtype=float))))


def make_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(subjects), dtype=object)
    rng.shuffle(shuffled)
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold)]


def group_indices(df: pd.DataFrame, group_cols: list[str]) -> list[np.ndarray]:
    return [idx.to_numpy() for _, idx in df.groupby(group_cols, sort=False).groups.items()]


def permute_residuals_within_groups(
    df: pd.DataFrame,
    groups: list[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    for col in RESIDUAL_COLS:
        values = out[col].to_numpy(copy=True)
        for idx in groups:
            if len(idx) > 1:
                values[idx] = rng.permutation(values[idx])
        out[col] = values
    return out


def empirical_p_high(observed: float, null_values: np.ndarray) -> float:
    valid = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(valid) == 0:
        return np.nan
    return float((1 + np.sum(valid >= observed)) / (len(valid) + 1))


def empirical_p_low(observed: float, null_values: np.ndarray) -> float:
    valid = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(valid) == 0:
        return np.nan
    return float((1 + np.sum(valid <= observed)) / (len(valid) + 1))


def run_global_null(df: pd.DataFrame, n_null: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    groups = group_indices(df, ["subject_id", "night_id", "channel"])
    obs_corr = residual_corr(df)
    obs_vec = upper_triangle(obs_corr)
    obs_mean_abs = mean_abs_offdiag(obs_vec)

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
    summary = {
        "observed_mean_abs_offdiag_corr": obs_mean_abs,
        "null_mean_abs_offdiag_corr_mean": float(null_df["mean_abs_offdiag_corr"].mean()),
        "null_mean_abs_offdiag_corr_sd": float(null_df["mean_abs_offdiag_corr"].std(ddof=1)),
        "null_mean_abs_offdiag_corr_p_high": empirical_p_high(
            obs_mean_abs, null_df["mean_abs_offdiag_corr"].to_numpy()
        ),
    }
    return null_df, summary


def run_fold_reproducibility(df: pd.DataFrame, n_folds: int, n_null: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].dropna().unique()
    folds = make_subject_folds(subjects, n_folds, seed)
    rows = []

    for fold_idx, heldout_subjects in enumerate(folds):
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
            shuffled = permute_residuals_within_groups(heldout, heldout_groups, rng)
            null_corr = residual_corr(shuffled)
            null_vec = upper_triangle(null_corr)
            null_similarity.append(vector_similarity(train_vec, null_vec))
            null_distance.append(frobenius_distance(train_corr, null_corr))

        null_similarity_arr = np.asarray(null_similarity, dtype=float)
        null_distance_arr = np.asarray(null_distance, dtype=float)
        rows.append(
            {
                "fold": fold_idx + 1,
                "heldout_subjects": "|".join(map(str, sorted(heldout_set))),
                "n_train_subjects": int(train["subject_id"].nunique()),
                "n_heldout_subjects": int(heldout["subject_id"].nunique()),
                "n_train_rows": int(len(train)),
                "n_heldout_rows": int(len(heldout)),
                "observed_train_heldout_vector_similarity": obs_similarity,
                "null_similarity_mean": float(np.nanmean(null_similarity_arr)),
                "null_similarity_sd": float(np.nanstd(null_similarity_arr, ddof=1)),
                "similarity_p_high": empirical_p_high(obs_similarity, null_similarity_arr),
                "observed_train_heldout_frobenius_distance": obs_distance,
                "null_distance_mean": float(np.nanmean(null_distance_arr)),
                "null_distance_sd": float(np.nanstd(null_distance_arr, ddof=1)),
                "distance_p_low": empirical_p_low(obs_distance, null_distance_arr),
            }
        )
    return pd.DataFrame(rows)


def observed_pair_table(obs_corr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, left in enumerate(RESIDUAL_COLS):
        for right in RESIDUAL_COLS[i + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "correlation": float(obs_corr.loc[left, right]),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    table_dir: Path,
    df: pd.DataFrame,
    obs_corr: pd.DataFrame,
    fold_df: pd.DataFrame,
    global_summary: dict,
    n_null: int,
    seed: int,
) -> None:
    summary_path = table_dir / "rncp_reproducibility_reproducibility_summary.md"
    fold_mean_similarity = float(fold_df["observed_train_heldout_vector_similarity"].mean())
    fold_mean_p = float(fold_df["similarity_p_high"].mean())
    fold_passes = int((fold_df["similarity_p_high"] < 0.05).sum())
    obs_pairs = observed_pair_table(obs_corr)

    lines = [
        "# RNCP Residual Reproducibility Summary",
        "",
        "## Design",
        "",
        f"- Rows tested: {len(df)}",
        f"- Subjects: {df['subject_id'].nunique()}",
        f"- Nights: {df['night_id'].nunique()}",
        f"- Features: {', '.join(FEATURES)}",
        f"- Subject-held-out folds: {len(fold_df)}",
        f"- Null iterations per test: {n_null}",
        f"- Random seed: {seed}",
        "",
        "Null residual columns are independently permuted within `subject_id + night_id + channel` blocks.",
        "",
        "## Global Residual Correlation",
        "",
        obs_corr.to_markdown(floatfmt=".4f"),
        "",
        "## Observed Pairwise RNCP Correlations",
        "",
        obs_pairs.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Global Null Test",
        "",
        f"- Observed mean absolute off-diagonal correlation: {global_summary['observed_mean_abs_offdiag_corr']:.4f}",
        f"- Null mean: {global_summary['null_mean_abs_offdiag_corr_mean']:.4f}",
        f"- Null SD: {global_summary['null_mean_abs_offdiag_corr_sd']:.4f}",
        f"- Empirical p, observed >= null: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}",
        "",
        "## Subject-Held-Out Fold Reproducibility",
        "",
        fold_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Interpretation Guardrail",
        "",
        f"- Mean train-heldout residual-correlation vector similarity: {fold_mean_similarity:.4f}",
        f"- Mean fold empirical similarity p-value: {fold_mean_p:.4f}",
        f"- Folds with empirical p < 0.05: {fold_passes} / {len(fold_df)}",
        "",
        "This tests whether residual covariance structure is reproducible across held-out subjects and stronger than a block-preserving shuffle null. It does not by itself prove conscious experience in N3.",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    residual_path = table_dir / "sleep_edf_sc_n3_rncp_residuals.csv.gz"
    df = pd.read_csv(residual_path, low_memory=False)
    df = df.dropna(subset=["subject_id", "night_id", "channel"] + RESIDUAL_COLS).reset_index(drop=True)

    obs_corr = residual_corr(df)
    obs_pairs = observed_pair_table(obs_corr)
    null_df, global_summary = run_global_null(df, args.n_null, args.seed)
    fold_df = run_fold_reproducibility(df, args.n_folds, args.n_null, args.seed + 1)

    obs_corr.to_csv(table_dir / "rncp_reproducibility_observed_correlation.csv")
    obs_pairs.to_csv(table_dir / "rncp_reproducibility_observed_pairwise_correlations.csv", index=False)
    null_df.to_csv(table_dir / "rncp_reproducibility_global_null_iterations.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(table_dir / "rncp_reproducibility_global_null_summary.csv", index=False)
    fold_df.to_csv(table_dir / "rncp_reproducibility_fold_reproducibility.csv", index=False)
    write_summary(table_dir, df, obs_corr, fold_df, global_summary, args.n_null, args.seed)

    print(f"Rows tested: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Observed mean abs offdiag corr: {global_summary['observed_mean_abs_offdiag_corr']:.4f}")
    print(f"Global empirical p: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}")
    print(f"Mean fold similarity: {fold_df['observed_train_heldout_vector_similarity'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
