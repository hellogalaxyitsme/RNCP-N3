#!/usr/bin/env python3
"""Transition-boundary circularity sensitivities for RNCP functional anchoring."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    anchor_path: str
    output_prefix: str


SPECS = [
    DatasetSpec("Sleep-EDF SC", "functional_sleep_edf_sc_epoch_anchor_frame.csv.gz", "functional_sleep_edf_sc"),
    DatasetSpec("Sleep-EDF ST", "functional_sleep_edf_st_epoch_anchor_frame.csv.gz", "functional_sleep_edf_st"),
    DatasetSpec("ANPHY-Sleep", "functional_anphy_epoch_anchor_frame.csv.gz", "functional_anphy"),
]


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


def prepare_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["subject_id"] = df["subject_id"].astype(str)
    df["night_id"] = df["night_id"].astype(str)
    if "recording_cluster" not in df:
        df["recording_cluster"] = df["subject_id"] + ":" + df["night_id"]
    if "position_within_bout_fraction" in df and "n3_bout_duration_min" in df:
        position = pd.to_numeric(df["position_within_bout_fraction"], errors="coerce")
        duration = pd.to_numeric(df["n3_bout_duration_min"], errors="coerce")
        df["time_from_n3_bout_start_min"] = position * duration
        df["time_from_n3_bout_end_min"] = (1 - position) * duration
    else:
        df["time_from_n3_bout_start_min"] = np.nan
        df["time_from_n3_bout_end_min"] = np.nan

    df["exit_within_2min_bool"] = bool_series(df["exit_within_2min"])
    df["exit_within_5min_bool"] = bool_series(df["exit_within_5min"])
    df["has_2min_followup_bool"] = bool_series(df["has_2min_followup"])
    df["has_5min_followup_bool"] = bool_series(df["has_5min_followup"])
    df["exit_2to5min"] = (~df["exit_within_2min_bool"]) & df["exit_within_5min_bool"]
    df["rncp_epoch_z_sensitivity"] = zscore(df["rncp_epoch_mean"])
    return df


def model_controls(df: pd.DataFrame, subject_fe: bool) -> list[str]:
    controls = [
        "rncp_epoch_z_sensitivity",
        "time_since_sleep_onset_z",
        "relative_delta_power_z",
        "slow_wave_density_z",
        "slow_wave_occupancy_z",
        "log_cumulative_swa_z",
    ]
    if not subject_fe:
        if "age_z" in df and df["age_z"].notna().any():
            controls.append("age_z")
        if "sex" in df and df["sex"].nunique(dropna=True) > 1:
            controls.append("C(sex)")
    else:
        controls.append("C(subject_id)")
    return controls


def fit_logistic(df: pd.DataFrame, outcome: str, model_name: str, scenario: str, dataset: str) -> dict[str, object]:
    subject_fe = model_name == "subject_fixed_effects"
    controls = model_controls(df, subject_fe=subject_fe)
    numeric_controls = [c for c in controls if not c.startswith("C(")]
    required = [outcome, "rncp_epoch_z_sensitivity"] + [c for c in numeric_controls if c != "rncp_epoch_z_sensitivity"]
    model_df = df.dropna(subset=required + ["recording_cluster"]).copy()
    model_df[outcome] = bool_series(model_df[outcome]).astype(int)

    row: dict[str, object] = {
        "dataset": dataset,
        "scenario": scenario,
        "outcome": outcome,
        "model": model_name,
        "n_rows": int(len(model_df)),
        "n_subjects": int(model_df["subject_id"].nunique()) if len(model_df) else 0,
        "n_nights": int(model_df["night_id"].nunique()) if len(model_df) else 0,
        "n_recording_clusters": int(model_df["recording_cluster"].nunique()) if len(model_df) else 0,
        "event_rate": float(model_df[outcome].mean()) if len(model_df) else np.nan,
    }
    if len(model_df) < 100 or model_df[outcome].nunique() < 2:
        row["status"] = "skipped_insufficient_variation"
        return row

    formula = f"{outcome} ~ " + " + ".join(controls)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.glm(formula=formula, data=model_df, family=Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": model_df["recording_cluster"].astype(str), "use_correction": True},
            )
        term = "rncp_epoch_z_sensitivity"
        conf = result.conf_int().loc[term]
        beta = float(result.params[term])
        row.update(
            {
                "status": "ok",
                "rncp_beta": beta,
                "rncp_odds_ratio": float(np.exp(beta)),
                "rncp_ci_low": float(np.exp(conf[0])),
                "rncp_ci_high": float(np.exp(conf[1])),
                "rncp_p_value": float(result.pvalues[term]),
                "aic": float(result.aic),
                "formula": formula,
                "warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
            }
        )
    except Exception as exc:
        row.update({"status": f"error_{type(exc).__name__}", "error": str(exc), "formula": formula})
    return row


def scenario_frames(df: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, str]]:
    primary = df[df["has_2min_followup_bool"]].copy()
    post_entry = df[df["has_2min_followup_bool"] & (df["time_from_n3_bout_start_min"] >= 5)].copy()

    delayed = df[df["has_5min_followup_bool"] & (~df["exit_within_2min_bool"])].copy()
    delayed_post_entry = delayed[delayed["time_from_n3_bout_start_min"] >= 5].copy()

    return [
        ("primary_2min_exit", "exit_within_2min_bool", primary, "Original 2-min N3-exit model."),
        (
            "post_entry5_2min_exit",
            "exit_within_2min_bool",
            post_entry,
            "2-min N3-exit model after excluding the first 5 min of each N3 bout.",
        ),
        (
            "delayed_2to5min_exit",
            "exit_2to5min",
            delayed,
            "Immediate 0-2 min exits removed; outcome is N3 exit 2.5-5 min ahead.",
        ),
        (
            "post_entry5_delayed_2to5min_exit",
            "exit_2to5min",
            delayed_post_entry,
            "Immediate 0-2 min exits removed and first 5 min of each N3 bout excluded; outcome is N3 exit 2.5-5 min ahead.",
        ),
    ]


def summarize_scenario(dataset: str, scenario: str, note: str, df: pd.DataFrame, outcome: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "scenario": scenario,
        "outcome": outcome,
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if len(df) else 0,
        "n_nights": int(df["night_id"].nunique()) if len(df) else 0,
        "event_rate": float(bool_series(df[outcome]).mean()) if len(df) else np.nan,
        "median_time_from_bout_start_min": float(pd.to_numeric(df["time_from_n3_bout_start_min"], errors="coerce").median()) if len(df) else np.nan,
        "median_time_to_n3_exit_min": float(pd.to_numeric(df["time_to_n3_exit_min"], errors="coerce").median()) if len(df) else np.nan,
        "note": note,
    }


def write_markdown(path: Path, model_rows: pd.DataFrame, scenario_rows: pd.DataFrame) -> None:
    primary_models = model_rows[
        (model_rows["status"] == "ok")
        & (model_rows["model"] == "subject_fixed_effects")
        & (model_rows["scenario"].isin(["primary_2min_exit", "post_entry5_2min_exit", "delayed_2to5min_exit", "post_entry5_delayed_2to5min_exit"]))
    ].copy()
    lines = [
        "# Transition-Circularity Sensitivity",
        "",
        "A literal exclusion of all epochs within 5 min before N3 exit is incompatible with a 2-min exit outcome, because it removes every positive 2-min exit event. We therefore used two valid anti-circularity sensitivities: (i) excluding the first 5 min of each N3 bout while preserving the 2-min exit outcome, and (ii) removing immediate 0-2 min exits and testing whether RNCP predicts delayed exit 2.5-5 min ahead.",
        "",
        "All models use the same sleep-depth covariates as the primary functional anchoring analysis and recording-clustered standard errors.",
        "",
        "## Scenario sizes",
        "",
        scenario_rows.to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Subject fixed-effect model summary",
        "",
        "| Dataset | Scenario | N epochs | Event rate | RNCP OR | 95% CI | p value |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in primary_models.iterrows():
        lines.append(
            "| {dataset} | {scenario} | {n_rows} | {event_rate:.3f} | {or_:.2f} | {lo:.2f}-{hi:.2f} | {p:.3g} |".format(
                dataset=row["dataset"],
                scenario=row["scenario"],
                n_rows=int(row["n_rows"]),
                event_rate=float(row["event_rate"]),
                or_=float(row["rncp_odds_ratio"]),
                lo=float(row["rncp_ci_low"]),
                hi=float(row["rncp_ci_high"]),
                p=float(row["rncp_p_value"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `post_entry5_2min_exit` tests whether the primary result is driven by transition-adjacent epochs immediately after N3 entry.",
            "- `delayed_2to5min_exit` removes all immediate 0-2 min exits, so a significant association indicates that RNCP rises before the final boundary epochs.",
            "- `post_entry5_delayed_2to5min_exit` is the most conservative version of this sensitivity because it removes both immediate exits and early-bout epochs.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", type=Path, required=True)
    args = parser.parse_args()

    all_model_rows = []
    all_scenario_rows = []
    for spec in SPECS:
        frame = prepare_frame(args.table_dir / spec.anchor_path)
        for scenario, outcome, scenario_df, note in scenario_frames(frame):
            all_scenario_rows.append(summarize_scenario(spec.dataset, scenario, note, scenario_df, outcome))
            for model_name in ["demographic_depth", "subject_fixed_effects"]:
                all_model_rows.append(fit_logistic(scenario_df, outcome, model_name, scenario, spec.dataset))

    model_df = pd.DataFrame(all_model_rows)
    scenario_df = pd.DataFrame(all_scenario_rows)
    model_path = args.table_dir / "transition_circularity_sensitivity_models.csv"
    scenario_path = args.table_dir / "transition_circularity_sensitivity_scenarios.csv"
    md_path = args.table_dir / "transition_circularity_sensitivity_summary.md"
    model_df.to_csv(model_path, index=False)
    scenario_df.to_csv(scenario_path, index=False)
    write_markdown(md_path, model_df, scenario_df)
    print(md_path)
    print(model_df[["dataset", "scenario", "model", "status", "n_rows", "event_rate", "rncp_odds_ratio", "rncp_ci_low", "rncp_ci_high", "rncp_p_value"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
