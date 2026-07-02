#!/usr/bin/env python3
"""functional anchoring of RNCP to sleep continuity outcomes."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial


EPOCH_SEC = 30.0
SLEEP_STAGES = {"N1", "N2", "N3", "REM"}
FEATURES = ["lzc", "permutation_entropy", "spectral_entropy", "aperiodic_exponent_specparam"]
RESIDUAL_COLS = [f"{feature}_rncp_residual_z" for feature in FEATURES]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    matrix_path: str
    residual_path: str
    metadata_path: str
    cohort_path: str
    output_prefix: str


SPECS = {
    "sleep_edf_sc": DatasetSpec(
        dataset="sleep_edf_sc",
        matrix_path="sleep_edf_sc_n3_analysis_matrix_primary.csv.gz",
        residual_path="sleep_edf_sc_n3_rncp_residuals.csv.gz",
        metadata_path="sleep_edf_sc_epoch_metadata.csv.gz",
        cohort_path="sleep_edf_sc_cohort_table.csv",
        output_prefix="functional_sleep_edf_sc",
    ),
    "sleep_edf_st": DatasetSpec(
        dataset="sleep_edf_st",
        matrix_path="sleep_edf_st_n3_analysis_matrix_primary.csv.gz",
        residual_path="sleep_edf_st_n3_rncp_residuals.csv.gz",
        metadata_path="sleep_edf_st_epoch_metadata.csv.gz",
        cohort_path="sleep_edf_st_cohort_table.csv",
        output_prefix="functional_sleep_edf_st",
    ),
    "anphy": DatasetSpec(
        dataset="anphy",
        matrix_path="anphy_sleep_n3_analysis_matrix_primary.csv.gz",
        residual_path="anphy_sleep_n3_rncp_residuals.csv.gz",
        metadata_path="anphy_sleep_epoch_metadata.csv.gz",
        cohort_path="anphy_sleep_cohort_table.csv",
        output_prefix="functional_anphy",
    ),
}


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


def read_epoch_metadata(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    metadata = pd.read_csv(table_dir / spec.metadata_path, low_memory=False)
    cols = [
        "subject_id",
        "night_id",
        "night",
        "epoch_idx",
        "epoch_start_min",
        "stage",
        "time_since_sleep_onset",
        "after_sleep_onset",
        "stage_is_sleep",
        "stage_is_n3",
        "artifact_flag",
        "age",
        "sex",
    ]
    cols = [col for col in cols if col in metadata.columns]
    out = metadata[cols].drop_duplicates(["subject_id", "night_id", "epoch_idx"]).copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["night_id"] = out["night_id"].astype(str)
    out["epoch_idx"] = pd.to_numeric(out["epoch_idx"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["epoch_idx"]).copy()
    out["epoch_idx"] = out["epoch_idx"].astype(int)
    if "stage_is_sleep" not in out:
        out["stage_is_sleep"] = out["stage"].isin(SLEEP_STAGES)
    else:
        out["stage_is_sleep"] = bool_series(out["stage_is_sleep"])
    if "after_sleep_onset" in out:
        out["after_sleep_onset"] = bool_series(out["after_sleep_onset"])
    else:
        out["after_sleep_onset"] = True
    return out


def add_transition_outcomes(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subject_id, night_id), df in metadata.groupby(["subject_id", "night_id"], sort=False):
        df = df.sort_values("epoch_idx").reset_index(drop=True)
        stages = df["stage"].astype(str).to_numpy()
        epoch_idx = df["epoch_idx"].to_numpy(dtype=int)
        n = len(df)
        for pos, idx in enumerate(epoch_idx):
            if stages[pos] != "N3":
                continue
            future = stages[pos + 1 :]
            next_stage = future[0] if len(future) else ""
            follow2 = future[:4]
            follow5 = future[:10]
            exit_offsets = np.flatnonzero(future != "N3")
            rows.append(
                {
                    "subject_id": subject_id,
                    "night_id": night_id,
                    "epoch_idx": int(idx),
                    "next_stage": next_stage,
                    "exit_next_epoch": bool(len(future) > 0 and next_stage != "N3"),
                    "exit_within_2min": bool(len(follow2) == 4 and np.any(follow2 != "N3")),
                    "exit_within_5min": bool(len(follow5) == 10 and np.any(follow5 != "N3")),
                    "has_2min_followup": bool(pos + 4 < n),
                    "has_5min_followup": bool(pos + 10 < n),
                    "time_to_n3_exit_min": (
                        float((exit_offsets[0] + 1) * EPOCH_SEC / 60.0) if len(exit_offsets) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def epoch_rncp_table(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    residual_cols = ["subject_id", "night_id", "epoch_idx", "channel", "rncp_l2_norm"] + RESIDUAL_COLS
    residuals = pd.read_csv(table_dir / spec.residual_path, usecols=lambda col: col in residual_cols, low_memory=False)
    residuals["subject_id"] = residuals["subject_id"].astype(str)
    residuals["night_id"] = residuals["night_id"].astype(str)
    grouped = residuals.groupby(["subject_id", "night_id", "epoch_idx"], as_index=False)
    out = grouped.agg(
        rncp_epoch_mean=("rncp_l2_norm", "mean"),
        rncp_epoch_median=("rncp_l2_norm", "median"),
        rncp_epoch_max=("rncp_l2_norm", "max"),
        rncp_channel_count=("channel", "nunique"),
    )
    for col in RESIDUAL_COLS:
        if col in residuals.columns:
            feature = col.replace("_rncp_residual_z", "")
            out[f"{feature}_residual_epoch_mean"] = grouped[col].mean()[col].to_numpy()
    out["rncp_epoch_z"] = zscore(out["rncp_epoch_mean"])
    return out


def matrix_control_table(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    matrix = pd.read_csv(table_dir / spec.matrix_path, low_memory=False)
    if "analysis_qc_include" in matrix:
        matrix = matrix[bool_series(matrix["analysis_qc_include"])].copy()
    matrix["subject_id"] = matrix["subject_id"].astype(str)
    matrix["night_id"] = matrix["night_id"].astype(str)
    control_cols = [
        "age",
        "sex",
        "time_since_sleep_onset",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
        "cumulative_swa",
        "n3_bout_num",
        "n3_bout_duration_min",
        "position_within_bout_fraction",
        "total_sleep_min",
        "psg_duration_min",
        "annotation_duration_min",
    ]
    available = [col for col in control_cols if col in matrix.columns]
    agg = {}
    for col in available:
        if col == "sex":
            agg[col] = "first"
        else:
            agg[col] = "mean"
    return matrix.groupby(["subject_id", "night_id", "epoch_idx"], as_index=False).agg(agg)


def build_epoch_anchor_frame(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    metadata = read_epoch_metadata(table_dir, spec)
    transitions = add_transition_outcomes(metadata)
    rncp = epoch_rncp_table(table_dir, spec)
    controls = matrix_control_table(table_dir, spec)
    frame = rncp.merge(transitions, on=["subject_id", "night_id", "epoch_idx"], how="inner", validate="one_to_one")
    frame = frame.merge(controls, on=["subject_id", "night_id", "epoch_idx"], how="left", validate="one_to_one")
    for col in [
        "age",
        "time_since_sleep_onset",
        "relative_delta_power",
        "slow_wave_density",
        "slow_wave_occupancy",
        "n3_bout_duration_min",
    ]:
        if col in frame:
            frame[f"{col}_z"] = zscore(frame[col])
    if "cumulative_swa" in frame:
        frame["log_cumulative_swa_z"] = zscore(np.log1p(pd.to_numeric(frame["cumulative_swa"], errors="coerce").clip(lower=0)))
    if "sex" in frame:
        frame["sex"] = frame["sex"].astype(str)
    if "n3_bout_num" in frame:
        bout = pd.to_numeric(frame["n3_bout_num"], errors="coerce").round().astype("Int64").astype(str)
        frame["n3_bout_cluster"] = frame["subject_id"].astype(str) + ":" + frame["night_id"].astype(str) + ":bout" + bout
        frame.loc[bout == "<NA>", "n3_bout_cluster"] = pd.NA
    frame["recording_cluster"] = frame["subject_id"].astype(str) + ":" + frame["night_id"].astype(str)
    return frame


def cluster_groups(df: pd.DataFrame) -> tuple[pd.Series, str]:
    if "n3_bout_cluster" in df and df["n3_bout_cluster"].notna().nunique() >= 10:
        return df["n3_bout_cluster"].astype(str), "n3_bout"
    return df["recording_cluster"].astype(str), "recording"


def fit_epoch_transition_models(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    outcomes = [
        ("exit_next_epoch", None),
        ("exit_within_2min", "has_2min_followup"),
        ("exit_within_5min", "has_5min_followup"),
    ]
    base_controls = ["rncp_epoch_z", "time_since_sleep_onset_z", "relative_delta_power_z", "slow_wave_density_z", "slow_wave_occupancy_z", "log_cumulative_swa_z"]
    if "age_z" in frame and frame["age_z"].notna().any():
        base_controls.append("age_z")
    if "sex" in frame and frame["sex"].nunique(dropna=True) > 1:
        base_controls.append("C(sex)")
    for outcome, followup_col in outcomes:
        df = frame.copy()
        if followup_col:
            df = df[bool_series(df[followup_col])].copy()
        df[outcome] = bool_series(df[outcome]).astype(int)
        controls = [col for col in base_controls if col.startswith("C(") or (col in df and df[col].notna().any())]
        required = [outcome, "rncp_epoch_z"] + [col for col in controls if not col.startswith("C(")]
        df = df.dropna(subset=required).copy()
        if len(df) < 100 or df[outcome].nunique() < 2:
            rows.append({"dataset": spec.dataset, "outcome": outcome, "status": "skipped_insufficient_variation", "n_rows": len(df)})
            continue
        groups, cluster_level = cluster_groups(df)
        formulas = [
            ("demographic_depth", f"{outcome} ~ " + " + ".join(controls)),
            ("subject_fixed_effects", f"{outcome} ~ " + " + ".join([c for c in controls if c not in {"age_z", "C(sex)"}] + ["C(subject_id)"])),
            ("subject_fixed_effects_bout_sensitivity", f"{outcome} ~ " + " + ".join([c for c in controls if c not in {"age_z", "C(sex)"}] + ["n3_bout_duration_min_z", "C(subject_id)"])),
        ]
        for model_name, formula in formulas:
            if "n3_bout_duration_min_z" in formula and "n3_bout_duration_min_z" not in df:
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = smf.glm(formula=formula, data=df, family=Binomial()).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": groups, "use_correction": True},
                    )
                term = "rncp_epoch_z"
                conf = result.conf_int().loc[term]
                beta = float(result.params[term])
                rows.append(
                    {
                        "dataset": spec.dataset,
                        "outcome": outcome,
                        "model": model_name,
                        "status": "ok",
                        "n_rows": int(result.nobs),
                        "n_subjects": int(df["subject_id"].nunique()),
                        "n_nights": int(df["night_id"].nunique()),
                        "cluster_level": cluster_level,
                        "n_clusters": int(groups.nunique()),
                        "event_rate": float(df[outcome].mean()),
                        "rncp_beta": beta,
                        "rncp_odds_ratio": float(np.exp(beta)),
                        "rncp_ci_low": float(np.exp(conf[0])),
                        "rncp_ci_high": float(np.exp(conf[1])),
                        "rncp_p_value": float(result.pvalues[term]),
                        "aic": float(result.aic),
                        "warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught)),
                        "formula": formula,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": spec.dataset,
                        "outcome": outcome,
                        "model": model_name,
                        "status": f"error_{type(exc).__name__}",
                        "n_rows": len(df),
                        "error": str(exc),
                        "formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def rncp_quintile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.dropna(subset=["rncp_epoch_z"]).copy()
    df["rncp_quintile"] = pd.qcut(df["rncp_epoch_z"], q=5, labels=False, duplicates="drop") + 1
    rows = []
    for quintile, part in df.groupby("rncp_quintile", sort=True):
        rows.append(
            {
                "rncp_quintile": int(quintile),
                "rows": len(part),
                "rncp_epoch_mean": float(part["rncp_epoch_mean"].mean()),
                "exit_next_epoch_rate": float(bool_series(part["exit_next_epoch"]).mean()),
                "exit_within_2min_rate": float(bool_series(part.loc[bool_series(part["has_2min_followup"]), "exit_within_2min"]).mean()),
                "exit_within_5min_rate": float(bool_series(part.loc[bool_series(part["has_5min_followup"]), "exit_within_5min"]).mean()),
                "median_time_to_n3_exit_min": float(pd.to_numeric(part["time_to_n3_exit_min"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def recording_continuity_table(table_dir: Path, spec: DatasetSpec, epoch_frame: pd.DataFrame) -> pd.DataFrame:
    metadata = read_epoch_metadata(table_dir, spec)
    cohort_path = table_dir / spec.cohort_path
    cohort = pd.read_csv(cohort_path, low_memory=False) if cohort_path.exists() else pd.DataFrame()
    if not cohort.empty:
        cohort["subject_id"] = cohort["subject_id"].astype(str)
        cohort["night_id"] = cohort["night_id"].astype(str)

    rncp = epoch_frame.groupby(["subject_id", "night_id"], as_index=False).agg(
        rncp_recording_mean=("rncp_epoch_mean", "mean"),
        rncp_recording_median=("rncp_epoch_median", "median"),
        rncp_recording_sd=("rncp_epoch_mean", "std"),
        n3_qc_epochs=("epoch_idx", "nunique"),
        exit_within_2min_rate=("exit_within_2min", lambda s: float(bool_series(s).mean())),
        exit_next_epoch_rate=("exit_next_epoch", lambda s: float(bool_series(s).mean())),
    )

    rows = []
    for (subject_id, night_id), df in metadata.groupby(["subject_id", "night_id"], sort=False):
        df = df.sort_values("epoch_idx")
        n_total = len(df)
        sleep = df["stage"].isin(SLEEP_STAGES)
        after_onset = bool_series(df["after_sleep_onset"]) if "after_sleep_onset" in df else pd.Series(True, index=df.index)
        wake_after_onset = after_onset & (df["stage"].astype(str) == "W")
        n3 = df["stage"].astype(str) == "N3"
        bout_count = int(df.loc[n3, "epoch_idx"].diff().ne(1).sum()) if n3.any() else 0
        n3_min = float(n3.sum() * EPOCH_SEC / 60.0)
        rows.append(
            {
                "subject_id": subject_id,
                "night_id": night_id,
                "metadata_epochs": n_total,
                "sleep_efficiency_epoch": float(sleep.mean()) if n_total else np.nan,
                "waso_min_epoch": float(wake_after_onset.sum() * EPOCH_SEC / 60.0),
                "n3_min_epoch": n3_min,
                "n3_bout_count_epoch": bout_count,
                "n3_fragmentation_per_hour_epoch": float(bout_count / (n3_min / 60.0)) if n3_min > 0 else np.nan,
            }
        )
    cont = pd.DataFrame(rows)
    out = rncp.merge(cont, on=["subject_id", "night_id"], how="left", validate="one_to_one")
    if not cohort.empty:
        keep = [
            "subject_id",
            "night_id",
            "age",
            "sex",
            "total_sleep_min",
            "total_n3_min",
            "psg_duration_min",
            "annotation_duration_min",
        ]
        keep = [col for col in keep if col in cohort.columns]
        out = out.merge(cohort[keep].drop_duplicates(["subject_id", "night_id"]), on=["subject_id", "night_id"], how="left")
    duration_col = "psg_duration_min" if "psg_duration_min" in out else "annotation_duration_min" if "annotation_duration_min" in out else None
    if duration_col and "total_sleep_min" in out:
        out["sleep_efficiency_cohort"] = pd.to_numeric(out["total_sleep_min"], errors="coerce") / pd.to_numeric(out[duration_col], errors="coerce")
    return out


def recording_correlations(recording: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    outcomes = [
        "sleep_efficiency_cohort",
        "sleep_efficiency_epoch",
        "waso_min_epoch",
        "n3_min_epoch",
        "n3_bout_count_epoch",
        "n3_fragmentation_per_hour_epoch",
        "exit_next_epoch_rate",
        "exit_within_2min_rate",
    ]
    rows = []
    for outcome in outcomes:
        if outcome not in recording:
            continue
        df = recording[["rncp_recording_mean", outcome]].dropna()
        if len(df) < 5 or df[outcome].nunique() < 2:
            continue
        rows.append(
            {
                "dataset": spec.dataset,
                "outcome": outcome,
                "n_recordings": len(df),
                "spearman_r": float(df["rncp_recording_mean"].corr(df[outcome], method="spearman")),
                "pearson_r": float(df["rncp_recording_mean"].corr(df[outcome], method="pearson")),
            }
        )
    return pd.DataFrame(rows)


def write_summary(table_dir: Path, spec: DatasetSpec, epoch_models: pd.DataFrame, quintiles: pd.DataFrame, correlations: pd.DataFrame) -> None:
    lines = [
        f"# Functional Anchoring Summary: {spec.dataset}",
        "",
        "## Purpose",
        "",
        "this analysis tests whether RNCP magnitude is physiologically anchored to sleep continuity rather than being only a residual statistical structure.",
        "",
        "Primary epoch-level outcome: probability of leaving N3 within the next 2 minutes.",
        "",
        "Primary model excludes retrospective N3 bout-position variables because those encode future bout termination. A separate bout-duration sensitivity model is reported.",
        "",
        "Logistic models report cluster-robust standard errors at the N3-bout level, with recording-level clustering as fallback when bout clusters are unavailable.",
        "",
        "## Epoch Transition Models",
        "",
        epoch_models.to_markdown(index=False, floatfmt=".4f") if not epoch_models.empty else "No epoch models completed.",
        "",
        "## RNCP Quintile Transition Rates",
        "",
        quintiles.to_markdown(index=False, floatfmt=".4f") if not quintiles.empty else "No quintile summary available.",
        "",
        "## Recording-Level Correlations",
        "",
        correlations.to_markdown(index=False, floatfmt=".4f") if not correlations.empty else "No recording-level correlations available.",
        "",
        "## Interpretation Guardrail",
        "",
        "These analyses are post-hoc functional anchors. They can strengthen biological plausibility if RNCP relates to transition risk or sleep continuity, but they should not be framed as direct evidence of conscious experience.",
    ]
    (table_dir / f"{spec.output_prefix}_functional_anchoring_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(cfg: dict, spec: DatasetSpec) -> None:
    table_dir = cfg["project_data_root"] / "tables"
    started = time.time()
    epoch_frame = build_epoch_anchor_frame(table_dir, spec)
    epoch_models = fit_epoch_transition_models(epoch_frame, spec)
    quintiles = rncp_quintile_summary(epoch_frame)
    recording = recording_continuity_table(table_dir, spec, epoch_frame)
    correlations = recording_correlations(recording, spec)

    epoch_frame.to_csv(table_dir / f"{spec.output_prefix}_epoch_anchor_frame.csv.gz", index=False, compression="gzip")
    epoch_models.to_csv(table_dir / f"{spec.output_prefix}_epoch_transition_models.csv", index=False)
    quintiles.to_csv(table_dir / f"{spec.output_prefix}_rncp_quintile_transition_rates.csv", index=False)
    recording.to_csv(table_dir / f"{spec.output_prefix}_recording_continuity.csv", index=False)
    correlations.to_csv(table_dir / f"{spec.output_prefix}_recording_correlations.csv", index=False)
    write_summary(table_dir, spec, epoch_models, quintiles, correlations)
    print(f"{spec.dataset} analysis complete in {time.time() - started:.1f}s", flush=True)
    print(f"Epoch anchor rows: {len(epoch_frame)}; recording rows: {len(recording)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(SPECS) + ["all"], default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = list(SPECS) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        run_dataset(cfg, SPECS[dataset])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
