#!/usr/bin/env python3
"""conservative N3 residual model tests."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


FEATURES = [
    "lzc",
    "permutation_entropy",
    "spectral_entropy",
    "aperiodic_exponent_specparam",
]
BASE_COLUMNS = [
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


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Path(v) if k.endswith("root") else v for k, v in raw.items()}


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean()
    sd = values.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return values * np.nan
    return (values - mean) / sd


def prepare_model_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    df = matrix[bool_series(matrix["analysis_qc_include"])].copy()
    df["sex"] = df["sex"].astype(str)
    df["channel"] = df["channel"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    df["log_cumulative_swa"] = np.log1p(pd.to_numeric(df["cumulative_swa"], errors="coerce").clip(lower=0))

    continuous = [
        "age",
        "time_since_sleep_onset",
        "position_within_bout_fraction",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
        "log_cumulative_swa",
    ]
    for col in continuous:
        df[f"{col}_z"] = zscore(df[col])
    df["time_since_sleep_onset_z2"] = df["time_since_sleep_onset_z"] ** 2
    df["time_since_sleep_onset_z3"] = df["time_since_sleep_onset_z"] ** 3

    for feature in FEATURES:
        df[f"{feature}_z"] = zscore(df[feature])

    required = [f"{feature}_z" for feature in FEATURES] + [
        "age_z",
        "time_since_sleep_onset_z",
        "time_since_sleep_onset_z2",
        "time_since_sleep_onset_z3",
        "position_within_bout_fraction_z",
        "relative_delta_power_z",
        "slow_wave_density_z",
        "slow_wave_occupancy_z",
        "log_cumulative_swa_z",
        "sex",
        "channel",
        "subject_id",
        "night_id",
    ]
    return df.dropna(subset=required).copy()


def fixed_formula(outcome: str) -> str:
    return (
        f"{outcome}_z ~ "
        "age_z + C(sex) + "
        "time_since_sleep_onset_z + time_since_sleep_onset_z2 + time_since_sleep_onset_z3 + "
        "position_within_bout_fraction_z + "
        "relative_delta_power_z + slow_wave_density_z + slow_wave_occupancy_z + log_cumulative_swa_z + "
        "C(channel)"
    )


def fit_mixed_model(df: pd.DataFrame, feature: str):
    formula = fixed_formula(feature)
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["subject_id"],
        re_formula="1",
        vc_formula={"night": "0 + C(night_id)"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(method="lbfgs", reml=True, maxiter=500, disp=False)
    return formula, result, [str(item.message) for item in caught]


def variance_summary(result, df: pd.DataFrame, feature: str, formula: str, warnings_list: list[str]) -> dict:
    fixed_fitted = np.asarray(result.model.exog @ result.fe_params.to_numpy())
    var_fixed = float(np.nanvar(fixed_fitted, ddof=0))
    var_subject = 0.0
    if result.cov_re is not None and result.cov_re.size:
        var_subject = float(np.asarray(result.cov_re)[0, 0])
    var_night = float(np.nansum(np.asarray(getattr(result, "vcomp", []), dtype=float)))
    var_resid = float(result.scale)
    total = var_fixed + var_subject + var_night + var_resid
    marginal_r2 = var_fixed / total if total > 0 else np.nan
    conditional_r2 = (var_fixed + var_subject + var_night) / total if total > 0 else np.nan
    residuals = np.asarray(result.resid)
    return {
        "feature": feature,
        "formula": formula,
        "n_rows": int(result.nobs),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_nights": int(df["night_id"].nunique()),
        "converged": bool(getattr(result, "converged", False)),
        "log_likelihood": float(result.llf),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "var_fixed": var_fixed,
        "var_subject_intercept": var_subject,
        "var_night_component": var_night,
        "var_residual": var_resid,
        "marginal_r2_approx": marginal_r2,
        "conditional_r2_approx": conditional_r2,
        "residual_sd": float(np.nanstd(residuals, ddof=0)),
        "residual_iqr": float(np.nanpercentile(residuals, 75) - np.nanpercentile(residuals, 25)),
        "warnings": " | ".join(dict.fromkeys(warnings_list)),
    }


def fixed_effects_table(result, feature: str) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    out = pd.DataFrame(
        {
            "feature": feature,
            "term": params.index,
            "estimate": params.values,
            "std_error": result.bse.reindex(params.index).values,
            "z_value": (params / result.bse.reindex(params.index)).values,
            "p_value": result.pvalues.reindex(params.index).values,
            "ci_low": conf.reindex(params.index)[0].values,
            "ci_high": conf.reindex(params.index)[1].values,
        }
    )
    return out


def residual_frame(df: pd.DataFrame, model_outputs: dict[str, object]) -> pd.DataFrame:
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
    summary_path = table_dir / "residual_model_summary.md"
    key_terms = fixed_effects[
        fixed_effects["term"].isin(
            [
                "relative_delta_power_z",
                "slow_wave_density_z",
                "slow_wave_occupancy_z",
                "log_cumulative_swa_z",
                "time_since_sleep_onset_z",
                "position_within_bout_fraction_z",
                "C(channel)[T.EEG Pz-Oz]",
            ]
        )
    ].copy()
    residual_cols = [f"{feature}_rncp_residual_z" for feature in modeled_features]
    residual_corr = residuals[residual_cols].corr()
    lines = [
        "# Initial Conservative Model Summary",
        "",
        "## Model",
        "",
        "Rows use `analysis_qc_include == True` from the primary analysis matrix.",
        "",
        "For each standardized feature:",
        "",
        "`feature_z ~ age_z + C(sex) + time_since_sleep_onset_z + time_since_sleep_onset_z2 + "
        "time_since_sleep_onset_z3 + position_within_bout_fraction_z + relative_delta_power_z + "
        "slow_wave_density_z + slow_wave_occupancy_z + log_cumulative_swa_z + C(channel) + "
        "(1|subject) + (1|night within subject variance component)`",
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
        "- `residual_model_fixed_effects.csv`",
        "- `residual_model_variance_summary.csv`",
        "- `sleep_edf_sc_n3_rncp_residuals.csv.gz`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", nargs="+", default=FEATURES)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    matrix = pd.read_csv(table_dir / "sleep_edf_sc_n3_analysis_matrix_primary.csv.gz", low_memory=False)
    df = prepare_model_frame(matrix)

    variance_rows = []
    fixed_effect_frames = []
    model_outputs = {}
    for feature in args.features:
        print(f"Fitting {feature}", flush=True)
        formula, result, warning_messages = fit_mixed_model(df, feature)
        model_outputs[feature] = result
        variance_rows.append(variance_summary(result, df, feature, formula, warning_messages))
        fixed_effect_frames.append(fixed_effects_table(result, feature))
        print(
            f"{feature}: converged={getattr(result, 'converged', False)} "
            f"resid_sd={np.nanstd(result.resid, ddof=0):.4f}",
            flush=True,
        )

    variance = pd.DataFrame(variance_rows)
    fixed_effects = pd.concat(fixed_effect_frames, ignore_index=True)
    residuals = residual_frame(df, model_outputs)

    variance.to_csv(table_dir / "residual_model_variance_summary.csv", index=False)
    fixed_effects.to_csv(table_dir / "residual_model_fixed_effects.csv", index=False)
    residuals.to_csv(table_dir / "sleep_edf_sc_n3_rncp_residuals.csv.gz", index=False, compression="gzip")
    write_summary(table_dir, variance, fixed_effects, residuals, list(model_outputs))

    print(f"Modeled rows: {len(df)}")
    print(f"Residual table rows: {len(residuals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
