#!/usr/bin/env python3
"""ANPHY-Sleep RNCP residual reproducibility and null tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from anphy_sleep_common import load_config
from sleep_edf_rncp_reproducibility import (
    RESIDUAL_COLS,
    FEATURES,
    observed_pair_table,
    residual_corr,
    run_fold_reproducibility,
    run_global_null,
)


def write_summary(table_dir: Path, df: pd.DataFrame, obs_corr: pd.DataFrame, fold_df: pd.DataFrame, global_summary: dict, n_null: int, seed: int) -> None:
    fold_mean_similarity = float(fold_df["observed_train_heldout_vector_similarity"].mean())
    fold_mean_p = float(fold_df["similarity_p_high"].mean())
    fold_passes = int((fold_df["similarity_p_high"] < 0.05).sum())
    obs_pairs = observed_pair_table(obs_corr)
    lines = [
        "# ANPHY-Sleep RNCP Reproducibility Summary",
        "",
        f"- Rows tested: {len(df)}",
        f"- Subjects: {df['subject_id'].nunique()}",
        f"- Recordings/nights: {df['night_id'].nunique()}",
        f"- Features: {', '.join(FEATURES)}",
        f"- Subject-held-out folds: {len(fold_df)}",
        f"- Null iterations per test: {n_null}",
        f"- Random seed: {seed}",
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
        "This tests external replication of the residual covariance structure. It does not by itself prove conscious experience in N3.",
    ]
    (table_dir / "anphy_rncp_reproducibility_reproducibility_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    df = pd.read_csv(table_dir / "anphy_sleep_n3_rncp_residuals.csv.gz", low_memory=False)
    df = df.dropna(subset=["subject_id", "night_id", "channel"] + RESIDUAL_COLS).reset_index(drop=True)

    obs_corr = residual_corr(df)
    obs_pairs = observed_pair_table(obs_corr)
    null_df, global_summary = run_global_null(df, args.n_null, args.seed)
    fold_df = run_fold_reproducibility(df, args.n_folds, args.n_null, args.seed + 1)

    obs_corr.to_csv(table_dir / "anphy_rncp_reproducibility_observed_correlation.csv")
    obs_pairs.to_csv(table_dir / "anphy_rncp_reproducibility_observed_pairwise_correlations.csv", index=False)
    null_df.to_csv(table_dir / "anphy_rncp_reproducibility_global_null_iterations.csv", index=False)
    pd.DataFrame([global_summary]).to_csv(table_dir / "anphy_rncp_reproducibility_global_null_summary.csv", index=False)
    fold_df.to_csv(table_dir / "anphy_rncp_reproducibility_fold_reproducibility.csv", index=False)
    write_summary(table_dir, df, obs_corr, fold_df, global_summary, args.n_null, args.seed)

    print(f"ANPHY rows tested: {len(df)}")
    print(f"ANPHY subjects: {df['subject_id'].nunique()}")
    print(f"ANPHY observed mean abs offdiag corr: {global_summary['observed_mean_abs_offdiag_corr']:.4f}")
    print(f"ANPHY global empirical p: {global_summary['null_mean_abs_offdiag_corr_p_high']:.4f}")
    print(f"ANPHY mean fold similarity: {fold_df['observed_train_heldout_vector_similarity'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
