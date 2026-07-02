#!/usr/bin/env python3
"""RNCP robustness and sensitivity checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
KEYS = ["subject_id", "night_id", "epoch_idx", "channel"]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    mode: str
    channel: str | None = None
    specparam_error_max: float | None = None
    min_total_n3_min: float | None = None
    aggregate: list[str] | None = None


SCENARIOS = [
    Scenario("baseline_epoch_all_channels", "Primary QC rows, both EEG channels", "epoch"),
    Scenario("channel_fpzcz_epoch", "Primary QC rows, EEG Fpz-Cz only", "epoch", channel="EEG Fpz-Cz"),
    Scenario("channel_pzoz_epoch", "Primary QC rows, EEG Pz-Oz only", "epoch", channel="EEG Pz-Oz"),
    Scenario("specparam_error_le_0p10", "Primary QC rows, specparam error <= 0.10", "epoch", specparam_error_max=0.10),
    Scenario("specparam_error_le_0p12", "Primary QC rows, specparam error <= 0.12", "epoch", specparam_error_max=0.12),
    Scenario("specparam_error_le_0p15", "Primary QC rows, specparam error <= 0.15", "epoch", specparam_error_max=0.15),
    Scenario("total_n3_min_ge_45", "Primary QC rows, recording total N3 >= 45 min", "epoch", min_total_n3_min=45.0),
    Scenario("total_n3_min_ge_60", "Primary QC rows, recording total N3 >= 60 min", "epoch", min_total_n3_min=60.0),
    Scenario(
        "fpzcz_specparam_error_le_0p12",
        "EEG Fpz-Cz only with specparam error <= 0.12",
        "epoch",
        channel="EEG Fpz-Cz",
        specparam_error_max=0.12,
    ),
    Scenario(
        "subject_night_channel_mean",
        "Mean RNCP residuals per subject-night-channel",
        "aggregate",
        aggregate=["subject_id", "night_id", "channel"],
    ),
    Scenario(
        "subject_channel_mean",
        "Mean RNCP residuals per subject-channel",
        "aggregate",
        aggregate=["subject_id", "channel"],
    ),
    Scenario("subject_mean", "Mean RNCP residuals per subject", "aggregate", aggregate=["subject_id"]),
]


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


def mean_abs_offdiag(vec: pd.Series) -> float:
    return float(np.nanmean(np.abs(vec.to_numpy(dtype=float))))


def vector_similarity(left: pd.Series, right: pd.Series) -> float:
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    if joined.iloc[:, 0].std(ddof=0) == 0 or joined.iloc[:, 1].std(ddof=0) == 0:
        return np.nan
    return float(np.corrcoef(joined.iloc[:, 0], joined.iloc[:, 1])[0, 1])


def empirical_p_high(observed: float, null_values: np.ndarray) -> float:
    valid = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(valid) == 0:
        return np.nan
    return float((1 + np.sum(valid >= observed)) / (len(valid) + 1))


def make_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(subjects), dtype=object)
    rng.shuffle(shuffled)
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold)]


def group_indices(df: pd.DataFrame, group_cols: list[str]) -> list[np.ndarray]:
    if not group_cols:
        return [df.index.to_numpy()]
    return [idx.to_numpy() for _, idx in df.groupby(group_cols, sort=False).groups.items()]


def permute_residuals(
    df: pd.DataFrame,
    groups: list[np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    out = df.copy()
    for col in RESIDUAL_COLS:
        values = out[col].to_numpy(copy=True)
        any_group_shuffled = False
        for idx in groups:
            if len(idx) > 1:
                values[idx] = rng.permutation(values[idx])
                any_group_shuffled = True
        if not any_group_shuffled and len(values) > 1:
            values = rng.permutation(values)
        out[col] = values
    return out


def load_analysis_data(table_dir: Path) -> pd.DataFrame:
    residuals = pd.read_csv(table_dir / "sleep_edf_sc_n3_rncp_residuals.csv.gz", low_memory=False)
    matrix_cols = KEYS + ["specparam_error", "total_n3_min", "analysis_qc_include"]
    matrix = pd.read_csv(
        table_dir / "sleep_edf_sc_n3_analysis_matrix_primary.csv.gz",
        usecols=matrix_cols,
        low_memory=False,
    )
    data = residuals.merge(matrix, on=KEYS, how="left", validate="one_to_one")
    data = data.dropna(subset=["subject_id", "night_id", "channel", "specparam_error"] + RESIDUAL_COLS)
    return data.reset_index(drop=True)


def apply_scenario(data: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    df = data.copy()
    if scenario.channel is not None:
        df = df[df["channel"] == scenario.channel].copy()
    if scenario.specparam_error_max is not None:
        df = df[df["specparam_error"] <= scenario.specparam_error_max].copy()
    if scenario.min_total_n3_min is not None:
        df = df[df["total_n3_min"] >= scenario.min_total_n3_min].copy()

    if scenario.aggregate:
        agg = df.groupby(scenario.aggregate, as_index=False).agg(
            {
                **{col: "mean" for col in RESIDUAL_COLS},
                "specparam_error": "mean",
                "total_n3_min": "mean",
                "epoch_idx": "count",
            }
        )
        agg = agg.rename(columns={"epoch_idx": "n_source_epoch_channel_rows"})
        for col in ["night_id", "channel"]:
            if col not in agg.columns:
                agg[col] = "aggregated"
        df = agg
    return df.dropna(subset=RESIDUAL_COLS).reset_index(drop=True)


def scenario_group_cols(scenario: Scenario, df: pd.DataFrame) -> list[str]:
    if scenario.mode == "epoch":
        return [col for col in ["subject_id", "night_id", "channel"] if col in df.columns]
    return []


def global_null(df: pd.DataFrame, scenario: Scenario, n_null: int, seed: int) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    obs_vec = upper_triangle(residual_corr(df))
    obs_mean_abs = mean_abs_offdiag(obs_vec)
    groups = group_indices(df, scenario_group_cols(scenario, df))
    null_rows = []
    for iteration in range(n_null):
        shuffled = permute_residuals(df, groups, rng)
        null_vec = upper_triangle(residual_corr(shuffled))
        null_rows.append(
            {
                "scenario": scenario.name,
                "iteration": iteration,
                "mean_abs_offdiag_corr": mean_abs_offdiag(null_vec),
            }
        )
    null_values = np.asarray([row["mean_abs_offdiag_corr"] for row in null_rows], dtype=float)
    summary = {
        "observed_mean_abs_offdiag_corr": obs_mean_abs,
        "null_mean_abs_offdiag_corr_mean": float(np.nanmean(null_values)),
        "null_mean_abs_offdiag_corr_sd": float(np.nanstd(null_values, ddof=1)),
        "global_p_high": empirical_p_high(obs_mean_abs, null_values),
    }
    return summary, null_rows


def fold_reproducibility(df: pd.DataFrame, scenario: Scenario, n_folds: int, n_null: int, seed: int) -> dict:
    if df["subject_id"].nunique() < n_folds:
        return {
            "fold_mean_similarity": np.nan,
            "fold_min_similarity": np.nan,
            "fold_mean_p_high": np.nan,
            "folds_p_lt_0p05": 0,
            "n_folds_completed": 0,
        }
    rng = np.random.default_rng(seed)
    folds = make_subject_folds(df["subject_id"].unique(), n_folds, seed)
    similarities = []
    p_values = []
    for heldout_subjects in folds:
        heldout = df[df["subject_id"].isin(set(heldout_subjects))].reset_index(drop=True)
        train = df[~df["subject_id"].isin(set(heldout_subjects))].reset_index(drop=True)
        if len(train) < 10 or len(heldout) < 10:
            continue
        train_vec = upper_triangle(residual_corr(train))
        heldout_vec = upper_triangle(residual_corr(heldout))
        observed_similarity = vector_similarity(train_vec, heldout_vec)
        groups = group_indices(heldout, scenario_group_cols(scenario, heldout))
        null_similarity = []
        for _ in range(n_null):
            shuffled = permute_residuals(heldout, groups, rng)
            null_similarity.append(vector_similarity(train_vec, upper_triangle(residual_corr(shuffled))))
        similarities.append(observed_similarity)
        p_values.append(empirical_p_high(observed_similarity, np.asarray(null_similarity, dtype=float)))
    return {
        "fold_mean_similarity": float(np.nanmean(similarities)) if similarities else np.nan,
        "fold_min_similarity": float(np.nanmin(similarities)) if similarities else np.nan,
        "fold_mean_p_high": float(np.nanmean(p_values)) if p_values else np.nan,
        "folds_p_lt_0p05": int(np.sum(np.asarray(p_values, dtype=float) < 0.05)) if p_values else 0,
        "n_folds_completed": len(similarities),
    }


def run_scenarios(data: pd.DataFrame, n_folds: int, n_null: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pair_rows = []
    null_rows = []
    for idx, scenario in enumerate(SCENARIOS):
        df = apply_scenario(data, scenario)
        if len(df) < 20 or df["subject_id"].nunique() < 5:
            continue
        obs_corr = residual_corr(df)
        obs_pairs = upper_triangle(obs_corr)
        g_summary, g_null_rows = global_null(df, scenario, n_null, seed + idx * 101)
        f_summary = fold_reproducibility(df, scenario, n_folds, max(100, n_null // 2), seed + idx * 101 + 1)
        summary_rows.append(
            {
                "scenario": scenario.name,
                "description": scenario.description,
                "mode": scenario.mode,
                "n_rows": int(len(df)),
                "n_subjects": int(df["subject_id"].nunique()),
                "n_nights": int(df["night_id"].nunique()) if "night_id" in df.columns else 0,
                "n_channels": int(df["channel"].nunique()) if "channel" in df.columns else 0,
                **g_summary,
                **f_summary,
            }
        )
        for pair, value in obs_pairs.items():
            left, right = pair.split("__")
            pair_rows.append({"scenario": scenario.name, "left": left, "right": right, "correlation": value})
        null_rows.extend(g_null_rows)
    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows), pd.DataFrame(null_rows)


def write_summary(table_dir: Path, summary: pd.DataFrame, pairs: pd.DataFrame, n_null: int, n_folds: int, seed: int) -> None:
    path = table_dir / "robustness_summary.md"
    robust = summary[
        (summary["global_p_high"] < 0.05)
        & (summary["fold_mean_similarity"] > 0.80)
        & (summary["n_subjects"] >= 30)
    ]
    lines = [
        "# Robustness and Sensitivity Summary",
        "",
        "## Design",
        "",
        f"- Scenarios evaluated: {len(summary)}",
        f"- Subject-held-out folds: {n_folds}",
        f"- Global null iterations per scenario: {n_null}",
        f"- Fold null iterations per scenario: {max(100, n_null // 2)}",
        f"- Random seed: {seed}",
        "",
        "Epoch-level nulls independently permute RNCP residual columns within `subject_id + night_id + channel` blocks. Aggregated scenarios permute each residual column across aggregate rows.",
        "",
        "## Scenario Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Robustness Screen",
        "",
        f"- Scenarios passing global p < 0.05, fold mean similarity > 0.80, and >=30 subjects: {len(robust)} / {len(summary)}",
        "",
        robust[["scenario", "n_rows", "n_subjects", "observed_mean_abs_offdiag_corr", "global_p_high", "fold_mean_similarity"]].to_markdown(index=False, floatfmt=".4f") if not robust.empty else "No scenarios passed the robustness screen.",
        "",
        "## Pairwise Correlations",
        "",
        pairs.to_markdown(index=False, floatfmt=".4f"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    cfg = load_config(args.config)
    table_dir = cfg["project_data_root"] / "tables"
    data = load_analysis_data(table_dir)
    summary, pairs, null_iterations = run_scenarios(data, args.n_folds, args.n_null, args.seed)

    summary.to_csv(table_dir / "robustness_sensitivity_summary.csv", index=False)
    pairs.to_csv(table_dir / "robustness_pairwise_correlations.csv", index=False)
    null_iterations.to_csv(table_dir / "robustness_global_null_iterations.csv", index=False)
    write_summary(table_dir, summary, pairs, args.n_null, args.n_folds, args.seed)

    print(f"Scenarios evaluated: {len(summary)}")
    print(summary[["scenario", "n_rows", "n_subjects", "observed_mean_abs_offdiag_corr", "global_p_high", "fold_mean_similarity"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
