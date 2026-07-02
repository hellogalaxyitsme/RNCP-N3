#!/usr/bin/env python3
"""Component-wise and PC-based RNCP functional anchoring models."""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial

from functional_anchoring import SPECS, DatasetSpec, bool_series, build_epoch_anchor_frame, cluster_groups, load_config, zscore


COMPONENTS = [
    ("lzc", "LZc", "lzc_residual_epoch_mean"),
    ("pe", "PE", "permutation_entropy_residual_epoch_mean"),
    ("se", "SE", "spectral_entropy_residual_epoch_mean"),
    ("ae", "AE", "aperiodic_exponent_specparam_residual_epoch_mean"),
]


@dataclass(frozen=True)
class PredictorSet:
    name: str
    display_name: str
    predictors: tuple[str, ...]
    predictor_labels: tuple[str, ...]
    model_family: str


def read_or_build_anchor_frame(table_dir: Path, spec: DatasetSpec) -> pd.DataFrame:
    path = table_dir / f"{spec.output_prefix}_epoch_anchor_frame.csv.gz"
    if path.exists():
        return pd.read_csv(path, low_memory=False)
    frame = build_epoch_anchor_frame(table_dir, spec)
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def add_component_and_pc_predictors(frame: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    component_z_cols = []
    for key, _label, source_col in COMPONENTS:
        z_col = f"rncp_{key}_component_z"
        df[z_col] = zscore(df[source_col])
        component_z_cols.append(z_col)

    valid = df[component_z_cols].dropna()
    corr = valid.corr().to_numpy(dtype=float)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for col_idx in range(eigvecs.shape[1]):
        strongest = int(np.argmax(np.abs(eigvecs[:, col_idx])))
        if eigvecs[strongest, col_idx] < 0:
            eigvecs[:, col_idx] *= -1

    score_matrix = df[component_z_cols].to_numpy(dtype=float)
    row_ok = np.isfinite(score_matrix).all(axis=1)
    pc_scores = np.full((len(df), len(component_z_cols)), np.nan, dtype=float)
    pc_scores[row_ok] = score_matrix[row_ok] @ eigvecs
    for idx in range(pc_scores.shape[1]):
        df[f"rncp_pc{idx + 1}_z"] = zscore(pd.Series(pc_scores[:, idx], index=df.index))

    loading_rows = []
    for pc_idx in range(eigvecs.shape[1]):
        for feature_idx, (_key, label, _source_col) in enumerate(COMPONENTS):
            loading_rows.append(
                {
                    "dataset": dataset,
                    "component": f"PC{pc_idx + 1}",
                    "feature": label,
                    "eigenvalue": float(eigvals[pc_idx]),
                    "variance_explained": float(eigvals[pc_idx] / eigvals.sum()),
                    "eigenvector_coefficient": float(eigvecs[feature_idx, pc_idx]),
                    "loading": float(eigvecs[feature_idx, pc_idx] * np.sqrt(eigvals[pc_idx])),
                }
            )
    eigen_rows = [
        {
            "dataset": dataset,
            "component": f"PC{idx + 1}",
            "eigenvalue": float(eig),
            "variance_explained": float(eig / eigvals.sum()),
            "cumulative_variance_explained": float(np.cumsum(eigvals / eigvals.sum())[idx]),
        }
        for idx, eig in enumerate(eigvals)
    ]
    return df, pd.DataFrame(loading_rows), pd.DataFrame(eigen_rows)


def predictor_sets() -> list[PredictorSet]:
    sets: list[PredictorSet] = []
    for key, label, _source_col in COMPONENTS:
        sets.append(
            PredictorSet(
                name=f"component_{key}_univariate",
                display_name=f"{label} residual",
                predictors=(f"rncp_{key}_component_z",),
                predictor_labels=(label,),
                model_family="single_component",
            )
        )
    sets.append(
        PredictorSet(
            name="components_multivariable",
            display_name="All four residual components",
            predictors=tuple(f"rncp_{key}_component_z" for key, _label, _source_col in COMPONENTS),
            predictor_labels=tuple(label for _key, label, _source_col in COMPONENTS),
            model_family="multivariable_components",
        )
    )
    for idx in range(1, 5):
        sets.append(
            PredictorSet(
                name=f"pc{idx}_univariate",
                display_name=f"RNCP PC{idx}",
                predictors=(f"rncp_pc{idx}_z",),
                predictor_labels=(f"PC{idx}",),
                model_family="single_pc",
            )
        )
    sets.append(
        PredictorSet(
            name="pc1_pc2_multivariable",
            display_name="RNCP PC1 + PC2",
            predictors=("rncp_pc1_z", "rncp_pc2_z"),
            predictor_labels=("PC1", "PC2"),
            model_family="multivariable_pcs",
        )
    )
    return sets


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


def fit_models(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    rows = []
    df_base = frame[bool_series(frame["has_2min_followup"])].copy()
    df_base["exit_within_2min"] = bool_series(df_base["exit_within_2min"]).astype(int)

    for pred_set in predictor_sets():
        for model_type in ("demographic_depth", "subject_fixed_effects"):
            controls = base_controls(df_base, model_type)
            required = ["exit_within_2min", *pred_set.predictors, *[c for c in controls if not c.startswith("C(")]]
            df = df_base.dropna(subset=required).copy()
            if len(df) < 100 or df["exit_within_2min"].nunique() < 2:
                rows.append(
                    {
                        "dataset": spec.dataset,
                        "outcome": "exit_within_2min",
                        "model": model_type,
                        "predictor_set": pred_set.name,
                        "model_family": pred_set.model_family,
                        "status": "skipped_insufficient_variation",
                        "n_rows": len(df),
                    }
                )
                continue

            formula = "exit_within_2min ~ " + " + ".join([*pred_set.predictors, *controls])
            groups, cluster_level = cluster_groups(df)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = smf.glm(formula=formula, data=df, family=Binomial()).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": groups, "use_correction": True},
                    )
                warnings_text = " | ".join(dict.fromkeys(str(item.message) for item in caught))
                for predictor, label in zip(pred_set.predictors, pred_set.predictor_labels):
                    conf = result.conf_int().loc[predictor]
                    beta = float(result.params[predictor])
                    rows.append(
                        {
                            "dataset": spec.dataset,
                            "outcome": "exit_within_2min",
                            "model": model_type,
                            "predictor_set": pred_set.name,
                            "predictor_label": label,
                            "model_family": pred_set.model_family,
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
                            "warnings": warnings_text,
                            "formula": formula,
                        }
                    )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": spec.dataset,
                        "outcome": "exit_within_2min",
                        "model": model_type,
                        "predictor_set": pred_set.name,
                        "model_family": pred_set.model_family,
                        "status": f"error_{type(exc).__name__}",
                        "n_rows": len(df),
                        "error": str(exc),
                        "formula": formula,
                    }
                )
    return pd.DataFrame(rows)


def write_summary(table_dir: Path, models: pd.DataFrame, eigen: pd.DataFrame) -> None:
    path = table_dir / "component_functional_anchoring_summary.md"
    lines = [
        "# Component-Wise Functional Anchoring",
        "",
        "## Purpose",
        "",
        "These analyses test whether the 2-min N3-exit association is confined to the aggregate RNCP L2 norm or is visible in individual residual components and principal components.",
        "",
        "All models use the 2-min N3-exit epoch anchor frame, the same sleep-depth covariates, and cluster-robust standard errors. Subject fixed-effect models are the primary specification.",
        "",
        "## Subject Fixed-Effect Component and PC Models",
        "",
        models.to_markdown(index=False, floatfmt=".4f") if not models.empty else "No models completed.",
        "",
        "## PC Variance",
        "",
        eigen.to_markdown(index=False, floatfmt=".4f") if not eigen.empty else "No PC summary available.",
        "",
        "## Interpretation Guardrail",
        "",
        "Component and PC models are sensitivity analyses. They should be framed as showing distributed functional anchoring of RNCP dimensions, not as replacing the prespecified RNCP magnitude model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: dict, datasets: list[str]) -> None:
    table_dir = cfg["project_data_root"] / "tables"
    all_models = []
    all_loadings = []
    all_eigen = []
    started = time.time()
    for dataset in datasets:
        spec = SPECS[dataset]
        print(f"component functional anchoring: {dataset}", flush=True)
        frame = read_or_build_anchor_frame(table_dir, spec)
        frame, loadings, eigen = add_component_and_pc_predictors(frame, dataset)
        models = fit_models(frame, spec)
        all_models.append(models)
        all_loadings.append(loadings)
        all_eigen.append(eigen)

    model_df = pd.concat(all_models, ignore_index=True)
    loading_df = pd.concat(all_loadings, ignore_index=True)
    eigen_df = pd.concat(all_eigen, ignore_index=True)

    model_df.to_csv(table_dir / "component_functional_anchoring_models.csv", index=False)
    loading_df.to_csv(table_dir / "component_pc_loadings.csv", index=False)
    eigen_df.to_csv(table_dir / "component_pc_eigenvalues.csv", index=False)
    write_summary(table_dir, model_df, eigen_df)
    print(f"analysis complete in {time.time() - started:.1f}s", flush=True)
    display_cols = ["dataset", "model", "predictor_set", "predictor_label", "status", "odds_ratio", "ci_low", "ci_high", "p_value"]
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
