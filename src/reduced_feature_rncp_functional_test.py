#!/usr/bin/env python3
"""Reduced-feature RNCP functional anchoring test.

This sensitivity analysis drops permutation entropy and aperiodic exponent
entirely, then tests whether a 2D LZc+SE RNCP magnitude predicts near-term
N3 exit under the same covariate structure used for .
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial

from functional_anchoring import (
    SPECS,
    DatasetSpec,
    bool_series,
    build_epoch_anchor_frame,
    cluster_groups,
    load_config,
    zscore,
)


REDUCED_COMPONENTS = [
    "lzc_rncp_residual_z",
    "spectral_entropy_rncp_residual_z",
]


def read_or_build_anchor_frame(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    path = table_dir / f"{spec.output_prefix}_epoch_anchor_frame.csv.gz"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    frame = build_epoch_anchor_frame(table_dir, spec)
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def reduced_epoch_rncp_table(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    usecols = ["subject_id", "night_id", "epoch_idx", "channel", *REDUCED_COMPONENTS]
    residuals = pd.read_csv(table_dir / spec.residual_path, usecols=lambda col: col in usecols, low_memory=False)
    residuals["subject_id"] = residuals["subject_id"].astype(str)
    residuals["night_id"] = residuals["night_id"].astype(str)
    for col in REDUCED_COMPONENTS:
        residuals[col] = pd.to_numeric(residuals[col], errors="coerce")
    residuals = residuals.dropna(subset=REDUCED_COMPONENTS).copy()
    residuals["rncp_lzc_se_l2_norm"] = np.sqrt(np.square(residuals[REDUCED_COMPONENTS]).sum(axis=1))
    grouped = residuals.groupby(["subject_id", "night_id", "epoch_idx"], as_index=False)
    out = grouped.agg(
        rncp_lzc_se_epoch_mean=("rncp_lzc_se_l2_norm", "mean"),
        rncp_lzc_se_epoch_median=("rncp_lzc_se_l2_norm", "median"),
        rncp_lzc_se_epoch_max=("rncp_lzc_se_l2_norm", "max"),
        rncp_lzc_se_channel_count=("channel", "nunique"),
        lzc_residual_epoch_mean=("lzc_rncp_residual_z", "mean"),
        spectral_entropy_residual_epoch_mean=("spectral_entropy_rncp_residual_z", "mean"),
    )
    out["rncp_lzc_se_epoch_z"] = zscore(out["rncp_lzc_se_epoch_mean"])
    return out


def add_reduced_predictor(table_dir: Path, spec: DatasetSpec, frame: pd.DataFrame) -> pd.DataFrame:
    reduced = reduced_epoch_rncp_table(table_dir, spec)
    drop_cols = [
        "rncp_lzc_se_epoch_mean",
        "rncp_lzc_se_epoch_median",
        "rncp_lzc_se_epoch_max",
        "rncp_lzc_se_channel_count",
        "lzc_residual_epoch_mean",
        "spectral_entropy_residual_epoch_mean",
        "rncp_lzc_se_epoch_z",
    ]
    df = frame.drop(columns=[col for col in drop_cols if col in frame.columns], errors="ignore")
    return df.merge(reduced, on=["subject_id", "night_id", "epoch_idx"], how="inner", validate="one_to_one")


def base_controls(frame: pd.DataFrame, model_type: str) -> list[str]:
    controls = [
        "time_since_sleep_onset_z",
        "relative_delta_power_z",
        "slow_wave_density_z",
        "slow_wave_occupancy_z",
        "log_cumulative_swa_z",
    ]
    if model_type == "demographic_depth":
        if "age_z" in frame and frame["age_z"].notna().any():
            controls.append("age_z")
        if "sex" in frame and frame["sex"].nunique(dropna=True) > 1:
            controls.append("C(sex)")
    else:
        controls.append("C(subject_id)")
    return [col for col in controls if col.startswith("C(") or (col in frame and frame[col].notna().any())]


def fit_reduced_models(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    predictor = "rncp_lzc_se_epoch_z"
    df_base = frame[bool_series(frame["has_2min_followup"])].copy()
    df_base["exit_within_2min"] = bool_series(df_base["exit_within_2min"]).astype(int)

    model_types = [
        "demographic_depth",
        "subject_fixed_effects",
        "subject_fixed_effects_bout_sensitivity",
    ]
    for model_type in model_types:
        controls = base_controls(df_base, "subject_fixed_effects" if model_type.startswith("subject") else model_type)
        if model_type == "subject_fixed_effects_bout_sensitivity":
            if "n3_bout_duration_min_z" not in df_base or not df_base["n3_bout_duration_min_z"].notna().any():
                continue
            controls = [*controls, "n3_bout_duration_min_z"]

        required = ["exit_within_2min", predictor, *[c for c in controls if not c.startswith("C(")]]
        df = df_base.dropna(subset=required).copy()
        if len(df) < 100 or df["exit_within_2min"].nunique() < 2:
            rows.append(
                {
                    "dataset": spec.dataset,
                    "outcome": "exit_within_2min",
                    "model": model_type,
                    "predictor": predictor,
                    "status": "skipped_insufficient_variation",
                    "n_rows": len(df),
                }
            )
            continue

        formula = "exit_within_2min ~ " + " + ".join([predictor, *controls])
        groups, cluster_level = cluster_groups(df)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = smf.glm(formula=formula, data=df, family=Binomial()).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": groups, "use_correction": True},
                )
            conf = result.conf_int().loc[predictor]
            beta = float(result.params[predictor])
            rows.append(
                {
                    "dataset": spec.dataset,
                    "outcome": "exit_within_2min",
                    "model": model_type,
                    "predictor": predictor,
                    "predictor_label": "LZc+SE RNCP magnitude",
                    "status": "ok",
                    "n_rows": int(result.nobs),
                    "n_subjects": int(df["subject_id"].nunique()),
                    "n_nights": int(df["night_id"].nunique()),
                    "cluster_level": cluster_level,
                    "n_clusters": int(groups.nunique()),
                    "event_rate": float(df["exit_within_2min"].mean()),
                    "beta": beta,
                    "odds_ratio": float(np.exp(beta)),
                    "ci_low": float(np.exp(conf[0])),
                    "ci_high": float(np.exp(conf[1])),
                    "p_value": float(result.pvalues[predictor]),
                    "aic": float(result.aic),
                    "warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
                    "formula": formula,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset": spec.dataset,
                    "outcome": "exit_within_2min",
                    "model": model_type,
                    "predictor": predictor,
                    "status": f"error_{type(exc).__name__}",
                    "n_rows": len(df),
                    "error": str(exc),
                    "formula": formula,
                }
            )
    return pd.DataFrame(rows)


def reduced_quintile_summary(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    df = frame[bool_series(frame["has_2min_followup"])].dropna(subset=["rncp_lzc_se_epoch_z"]).copy()
    df["exit_within_2min"] = bool_series(df["exit_within_2min"]).astype(int)
    df["rncp_lzc_se_quintile"] = pd.qcut(df["rncp_lzc_se_epoch_z"], q=5, labels=False, duplicates="drop") + 1
    rows = []
    for quintile, part in df.groupby("rncp_lzc_se_quintile", sort=True):
        rows.append(
            {
                "dataset": spec.dataset,
                "rncp_lzc_se_quintile": int(quintile),
                "rows": int(len(part)),
                "rncp_lzc_se_epoch_mean": float(part["rncp_lzc_se_epoch_mean"].mean()),
                "rncp_lzc_se_epoch_z_mean": float(part["rncp_lzc_se_epoch_z"].mean()),
                "exit_within_2min_rate": float(part["exit_within_2min"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_with_full_rncp(table_dir: Path, reduced_models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, spec in SPECS.items():
        path = table_dir / f"{spec.output_prefix}_epoch_transition_models.csv"
        if not path.exists():
            continue
        full = pd.read_csv(path)
        full = full[
            (full["status"] == "ok")
            & (full["outcome"] == "exit_within_2min")
            & (full["model"].isin(["demographic_depth", "subject_fixed_effects", "subject_fixed_effects_bout_sensitivity"]))
        ].copy()
        for _, row in full.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "model": row["model"],
                    "predictor_family": "Full 4-feature RNCP",
                    "odds_ratio": row["rncp_odds_ratio"],
                    "ci_low": row["rncp_ci_low"],
                    "ci_high": row["rncp_ci_high"],
                    "p_value": row["rncp_p_value"],
                    "n_rows": row.get("n_rows", np.nan),
                    "event_rate": row.get("event_rate", np.nan),
                }
            )
    reduced_ok = reduced_models[(reduced_models["status"] == "ok")].copy()
    for _, row in reduced_ok.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "predictor_family": "Reduced LZc+SE RNCP",
                "odds_ratio": row["odds_ratio"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "n_rows": row.get("n_rows", np.nan),
                "event_rate": row.get("event_rate", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["or_95ci"] = out.apply(lambda r: f"{float(r['odds_ratio']):.3f} ({float(r['ci_low']):.3f}-{float(r['ci_high']):.3f})", axis=1)
    out["p_formatted"] = out["p_value"].map(lambda p: f"{float(p):.3g}")
    return out


def write_summary(table_dir: Path, reduced_models: pd.DataFrame, comparison: pd.DataFrame) -> None:
    path = table_dir / "reduced_feature_rncp_functional_test_summary.md"
    lines = [
        "# Reduced-Feature RNCP Functional Test",
        "",
        "## Purpose",
        "",
        "This sensitivity analysis tests whether RNCP functional anchoring survives after dropping permutation entropy and aperiodic exponent entirely. The reduced predictor is the epoch-level mean of channel/derivation-wise sqrt(LZc residual^2 + spectral-entropy residual^2), z-scored within dataset.",
        "",
        "All models use the same 2-min N3-exit outcome, sleep-depth covariates, subject fixed-effect specification, and cluster-robust standard errors used in the functional anchoring analysis.",
        "",
        "## Subject Fixed-Effect Comparison",
        "",
        comparison[comparison["model"] == "subject_fixed_effects"].to_markdown(index=False, floatfmt=".4g") if not comparison.empty else "No comparison table available.",
        "",
        "## All Reduced-Feature Models",
        "",
        reduced_models.to_markdown(index=False, floatfmt=".4g") if not reduced_models.empty else "No models completed.",
        "",
        "## Output Files",
        "",
        "- `reduced_feature_rncp_models.csv`",
        "- `reduced_feature_rncp_quintiles.csv`",
        "- `reduced_feature_rncp_vs_full_functional_comparison.csv`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict, datasets: list[str]) -> None:
    table_dir = cfg["project_data_root"] / "tables"
    started = time.time()
    all_models = []
    all_quintiles = []
    for dataset in datasets:
        spec = SPECS[dataset]
        print(f"reduced-feature RNCP functional test: {dataset}", flush=True)
        frame = read_or_build_anchor_frame(table_dir, spec)
        frame = add_reduced_predictor(table_dir, spec, frame)
        all_models.append(fit_reduced_models(frame, spec))
        all_quintiles.append(reduced_quintile_summary(frame, spec))

    model_df = pd.concat(all_models, ignore_index=True) if all_models else pd.DataFrame()
    quintile_df = pd.concat(all_quintiles, ignore_index=True) if all_quintiles else pd.DataFrame()
    comparison = compare_with_full_rncp(table_dir, model_df)

    model_df.to_csv(table_dir / "reduced_feature_rncp_models.csv", index=False)
    quintile_df.to_csv(table_dir / "reduced_feature_rncp_quintiles.csv", index=False)
    comparison.to_csv(table_dir / "reduced_feature_rncp_vs_full_functional_comparison.csv", index=False)
    write_summary(table_dir, model_df, comparison)

    print(f"analysis complete in {time.time() - started:.1f}s", flush=True)
    display_cols = ["dataset", "model", "status", "odds_ratio", "ci_low", "ci_high", "p_value", "n_rows", "event_rate"]
    display_cols = [col for col in display_cols if col in model_df.columns]
    print(model_df[display_cols].to_string(index=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS) + ["all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    run(cfg, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
