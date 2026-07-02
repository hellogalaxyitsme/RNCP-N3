#!/usr/bin/env python3
"""ANPHY-Sleep conservative N3 residual model tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from anphy_sleep_common import load_config
from sleep_edf_residual_models import (
    FEATURES,
    fixed_effects_table,
    prepare_model_frame,
)


def fixed_formula_subject_fe(outcome: str) -> str:
    # ANPHY has one recording/night per subject. Subject fixed effects are a
    # stricter way to remove subject-level offsets than an unidentifiable night
    # variance component or a singular random intercept.
    return (
        f"{outcome}_z ~ "
        "age_z + C(sex) + C(subject_id) + "
        "time_since_sleep_onset_z + time_since_sleep_onset_z2 + time_since_sleep_onset_z3 + "
        "position_within_bout_fraction_z + "
        "relative_delta_power_z + slow_wave_density_z + slow_wave_occupancy_z + log_cumulative_swa_z + "
        "C(channel)"
    )


def fit_conservative_model(df: pd.DataFrame, feature: str):
    formula = fixed_formula_subject_fe(feature)
    result = smf.ols(formula, data=df).fit()
    return formula, result, []


def variance_summary_ols(result, df: pd.DataFrame, feature: str, formula: str) -> dict:
    fitted = np.asarray(result.fittedvalues)
    residuals = np.asarray(result.resid)
    var_fixed = float(np.nanvar(fitted, ddof=0))
    var_resid = float(np.nanvar(residuals, ddof=0))
    total = var_fixed + var_resid
    return {
        "feature": feature,
        "formula": formula,
        "n_rows": int(result.nobs),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_nights": int(df["night_id"].nunique()),
        "converged": True,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "var_fixed": var_fixed,
        "var_subject_intercept": np.nan,
        "var_night_component": np.nan,
        "var_residual": var_resid,
        "marginal_r2_approx": var_fixed / total if total > 0 else np.nan,
        "conditional_r2_approx": var_fixed / total if total > 0 else np.nan,
        "residual_sd": float(np.nanstd(residuals, ddof=0)),
        "residual_iqr": float(np.nanpercentile(residuals, 75) - np.nanpercentile(residuals, 25)),
        "warnings": "",
        "model_family": "ols_subject_fixed_effects",
    }


def residual_frame_ols(df: pd.DataFrame, model_outputs: dict[str, object]) -> pd.DataFrame:
    out = df[
        [
            "subject_id",
            "night_id",
            "epoch_idx",
            "channel",
            "age",
            "sex",
            "time_since_sleep_onset",
            "position_within_bout_fraction",
            "relative_delta_power",
            "slow_wave_density",
            "slow_wave_occupancy",
            "cumulative_swa",
        ]
    ].copy()
    for feature, result in model_outputs.items():
        out[f"{feature}_z"] = df[f"{feature}_z"].to_numpy()
        out[f"{feature}_fitted_z"] = np.asarray(result.fittedvalues)
        out[f"{feature}_rncp_residual_z"] = np.asarray(result.resid)
    residual_cols = [f"{feature}_rncp_residual_z" for feature in model_outputs]
    out["rncp_l2_norm"] = np.sqrt(np.square(out[residual_cols]).sum(axis=1))
    return out


def write_summary(
    table_dir: Path,
    variance: pd.DataFrame,
    fixed_effects: pd.DataFrame,
    residuals: pd.DataFrame,
    modeled_features: list[str],
) -> None:
    residual_cols = [f"{feature}_rncp_residual_z" for feature in modeled_features]
    residual_corr = residuals[residual_cols].corr()
    key_terms = fixed_effects[
        fixed_effects["term"].isin(
            [
                "relative_delta_power_z",
                "slow_wave_density_z",
                "slow_wave_occupancy_z",
                "log_cumulative_swa_z",
                "time_since_sleep_onset_z",
                "position_within_bout_fraction_z",
            ]
        )
    ].copy()

    lines = [
        "# ANPHY-Sleep Conservative Model Summary",
        "",
        "Rows use `analysis_qc_include == True` from the ANPHY primary analysis matrix.",
        "",
        "Model: same sleep-depth and channel controls as Sleep-EDF, with `C(subject_id)` fixed effects. ANPHY has one night per subject, so subject fixed effects are used instead of an unidentifiable night variance component or singular random intercept.",
        "",
        "## Variance Summary",
        "",
        variance.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Key Fixed Effects",
        "",
        key_terms.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## RNCP Residual Correlation",
        "",
        residual_corr.to_markdown(floatfmt=".4f"),
        "",
        "## Outputs",
        "",
        "- `anphy_residual_model_fixed_effects.csv`",
        "- `anphy_residual_model_variance_summary.csv`",
        "- `anphy_sleep_n3_rncp_residuals.csv.gz`",
    ]
    (table_dir / "anphy_residual_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=FEATURES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / "anphy_sleep_n3_analysis_matrix_primary.csv.gz", low_memory=False)
    df = prepare_model_frame(matrix)

    variance_rows = []
    fixed_effect_frames = []
    model_outputs = {}
    for feature in args.features:
        print(f"Fitting ANPHY {feature}", flush=True)
        formula, result, warning_messages = fit_conservative_model(df, feature)
        model_outputs[feature] = result
        variance_rows.append(variance_summary_ols(result, df, feature, formula))
        fixed_effect_frames.append(fixed_effects_table(result, feature))
        print(
            f"{feature}: converged=True "
            f"resid_sd={np.nanstd(result.resid, ddof=0):.4f}",
            flush=True,
        )

    variance = pd.DataFrame(variance_rows)
    fixed_effects = pd.concat(fixed_effect_frames, ignore_index=True)
    residuals = residual_frame_ols(df, model_outputs)

    variance.to_csv(table_dir / "anphy_residual_model_variance_summary.csv", index=False)
    fixed_effects.to_csv(table_dir / "anphy_residual_model_fixed_effects.csv", index=False)
    residuals.to_csv(table_dir / "anphy_sleep_n3_rncp_residuals.csv.gz", index=False, compression="gzip")
    write_summary(table_dir, variance, fixed_effects, residuals, list(model_outputs))

    print(f"ANPHY modeled rows: {len(df)}")
    print(f"ANPHY residual table rows: {len(residuals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
