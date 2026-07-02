#!/usr/bin/env python3
"""PE-AE exclusion and PCA sensitivity analyses for RNCP residual structure."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"

RESIDUAL_COLS = [
    "lzc_rncp_residual_z",
    "permutation_entropy_rncp_residual_z",
    "spectral_entropy_rncp_residual_z",
    "aperiodic_exponent_specparam_rncp_residual_z",
]
FEATURE_LABELS = {
    "lzc_rncp_residual_z": "LZc",
    "permutation_entropy_rncp_residual_z": "PE",
    "spectral_entropy_rncp_residual_z": "SE",
    "aperiodic_exponent_specparam_rncp_residual_z": "AE",
}
EXCLUDED_PAIR = ("permutation_entropy_rncp_residual_z", "aperiodic_exponent_specparam_rncp_residual_z")


DATASETS = [
    {
        "dataset": "Sleep-EDF SC",
        "analysis": "baseline",
        "residual_path": TABLE_DIR / "sleep_edf_sc_n3_rncp_residuals.csv.gz",
        "null_path": TABLE_DIR / "rncp_reproducibility_global_null_iterations.csv",
    },
    {
        "dataset": "Sleep-EDF ST",
        "analysis": "replication",
        "residual_path": TABLE_DIR / "sleep_edf_st_n3_rncp_residuals.csv.gz",
        "null_path": TABLE_DIR / "sleep_edf_st_rncp_reproducibility_global_null_iterations.csv",
    },
    {
        "dataset": "ANPHY-Sleep",
        "analysis": "artifact-adjusted",
        "residual_path": TABLE_DIR / "artifact_anphy_artifact_adjusted_rncp_residuals.csv.gz",
        "null_path": TABLE_DIR / "artifact_anphy_artifact_adjusted_global_null_iterations.csv",
    },
]


def pair_key(left: str, right: str) -> str:
    return f"{left}__{right}"


def pair_label(left: str, right: str) -> str:
    return f"{FEATURE_LABELS[left]}-{FEATURE_LABELS[right]}"


PAIR_KEYS = [
    pair_key(left, right)
    for i, left in enumerate(RESIDUAL_COLS)
    for right in RESIDUAL_COLS[i + 1 :]
]
PAIR_LABELS = {
    pair_key(left, right): pair_label(left, right)
    for i, left in enumerate(RESIDUAL_COLS)
    for right in RESIDUAL_COLS[i + 1 :]
}
EXCLUDED_PAIR_KEY = pair_key(*EXCLUDED_PAIR)


def read_residual_matrix(path: Path) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[list[float]] = []
    with opener(path, "rt", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            ok = True
            for col in RESIDUAL_COLS:
                try:
                    value = float(row[col])
                except (KeyError, TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(value):
                    ok = False
                    break
                values.append(value)
            if ok:
                rows.append(values)
    return np.asarray(rows, dtype=float)


def read_null_iterations(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if key == "iteration":
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = np.nan
            rows.append(parsed)
    return rows


def empirical_p_high(observed: float, null_values: np.ndarray) -> float:
    valid = null_values[np.isfinite(null_values)]
    if len(valid) == 0:
        return np.nan
    return float((1 + np.sum(valid >= observed)) / (len(valid) + 1))


def pca_from_corr(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    for j in range(eigvecs.shape[1]):
        idx = int(np.argmax(np.abs(eigvecs[:, j])))
        if eigvecs[idx, j] < 0:
            eigvecs[:, j] *= -1
    return eigvals, eigvecs


def n_components_for(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def fmt_float(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    eigen_rows: list[dict[str, object]] = []
    loading_rows: list[dict[str, object]] = []

    for spec in DATASETS:
        residuals = read_residual_matrix(spec["residual_path"])
        corr = np.corrcoef(residuals, rowvar=False)
        obs_pairs: dict[str, float] = {}
        for i, left in enumerate(RESIDUAL_COLS):
            for j, right in enumerate(RESIDUAL_COLS[i + 1 :], start=i + 1):
                key = pair_key(left, right)
                obs_pairs[key] = float(corr[i, j])
                pair_rows.append(
                    {
                        "dataset": spec["dataset"],
                        "analysis": spec["analysis"],
                        "pair": PAIR_LABELS[key],
                        "correlation": obs_pairs[key],
                        "abs_correlation": abs(obs_pairs[key]),
                        "excluded_from_pe_ae_sensitivity": key == EXCLUDED_PAIR_KEY,
                    }
                )

        observed_all = float(np.mean([abs(obs_pairs[key]) for key in PAIR_KEYS]))
        keep_keys = [key for key in PAIR_KEYS if key != EXCLUDED_PAIR_KEY]
        observed_excluding_pe_ae = float(np.mean([abs(obs_pairs[key]) for key in keep_keys]))

        null_rows = read_null_iterations(spec["null_path"])
        null_all = np.asarray([row["mean_abs_offdiag_corr"] for row in null_rows], dtype=float)
        null_excluding = []
        for row in null_rows:
            values = []
            for key in keep_keys:
                col = f"corr_{key}"
                if col in row and np.isfinite(row[col]):
                    values.append(abs(row[col]))
            null_excluding.append(np.mean(values) if values else np.nan)
        null_excluding_arr = np.asarray(null_excluding, dtype=float)

        eigvals, eigvecs = pca_from_corr(corr)
        variance_explained = eigvals / np.sum(eigvals)
        cumulative = np.cumsum(variance_explained)
        participation_ratio = float((np.sum(eigvals) ** 2) / np.sum(eigvals**2))

        summary_rows.append(
            {
                "dataset": spec["dataset"],
                "analysis": spec["analysis"],
                "n_rows": int(residuals.shape[0]),
                "observed_mean_abs_r_all_6_pairs": observed_all,
                "null_mean_abs_r_all_6_pairs": float(np.nanmean(null_all)),
                "observed_mean_abs_r_excluding_pe_ae_5_pairs": observed_excluding_pe_ae,
                "null_mean_abs_r_excluding_pe_ae_5_pairs": float(np.nanmean(null_excluding_arr)),
                "null_sd_excluding_pe_ae_5_pairs": float(np.nanstd(null_excluding_arr, ddof=1)),
                "empirical_p_excluding_pe_ae": empirical_p_high(observed_excluding_pe_ae, null_excluding_arr),
                "pe_ae_correlation": obs_pairs[EXCLUDED_PAIR_KEY],
                "pc1_variance_explained": float(variance_explained[0]),
                "pc2_variance_explained": float(variance_explained[1]),
                "pc1_pc2_cumulative_variance": float(cumulative[1]),
                "n_components_80pct": n_components_for(cumulative, 0.80),
                "n_components_90pct": n_components_for(cumulative, 0.90),
                "participation_ratio_effective_dimensions": participation_ratio,
            }
        )

        for idx, (eig, var, cum) in enumerate(zip(eigvals, variance_explained, cumulative), start=1):
            eigen_rows.append(
                {
                    "dataset": spec["dataset"],
                    "analysis": spec["analysis"],
                    "component": f"PC{idx}",
                    "eigenvalue": float(eig),
                    "variance_explained": float(var),
                    "cumulative_variance_explained": float(cum),
                }
            )
        for pc_idx in range(eigvecs.shape[1]):
            for feature_idx, feature_col in enumerate(RESIDUAL_COLS):
                loading_rows.append(
                    {
                        "dataset": spec["dataset"],
                        "analysis": spec["analysis"],
                        "component": f"PC{pc_idx + 1}",
                        "feature": FEATURE_LABELS[feature_col],
                        "eigenvector_coefficient": float(eigvecs[feature_idx, pc_idx]),
                        "loading": float(eigvecs[feature_idx, pc_idx] * np.sqrt(eigvals[pc_idx])),
                    }
                )

    summary_path = TABLE_DIR / "rncp_peae_exclusion_pca_summary.csv"
    pair_path = TABLE_DIR / "rncp_pairwise_correlations_for_peae_sensitivity.csv"
    eigen_path = TABLE_DIR / "rncp_pca_eigenvalues.csv"
    loading_path = TABLE_DIR / "rncp_pca_loadings.csv"
    md_path = TABLE_DIR / "rncp_peae_exclusion_pca_summary.md"

    write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    write_csv(pair_path, pair_rows, list(pair_rows[0].keys()))
    write_csv(eigen_path, eigen_rows, list(eigen_rows[0].keys()))
    write_csv(loading_path, loading_rows, list(loading_rows[0].keys()))

    lines = [
        "# RNCP PE-AE Exclusion and Effective Dimensionality Sensitivity",
        "",
        "This sensitivity analysis asks whether the primary RNCP structure survives after removing the dominant PE-AE residual pair from the mean absolute off-diagonal correlation statistic. The same saved block-preserving null iterations were rescored after excluding PE-AE.",
        "",
        "## PE-AE-excluded RNCP statistic",
        "",
        "| Dataset | Analysis | Rows | Observed all 6 | Observed excluding PE-AE | Null mean excluding PE-AE | p excluding PE-AE | PE-AE r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dataset} | {analysis} | {n_rows} | {all_obs} | {obs_ex} | {null_ex} | {p_ex} | {peae} |".format(
                dataset=row["dataset"],
                analysis=row["analysis"],
                n_rows=row["n_rows"],
                all_obs=fmt_float(row["observed_mean_abs_r_all_6_pairs"]),
                obs_ex=fmt_float(row["observed_mean_abs_r_excluding_pe_ae_5_pairs"]),
                null_ex=fmt_float(row["null_mean_abs_r_excluding_pe_ae_5_pairs"]),
                p_ex=fmt_float(row["empirical_p_excluding_pe_ae"]),
                peae=fmt_float(row["pe_ae_correlation"]),
            )
        )
    lines.extend(
        [
            "",
            "## PCA / effective dimensionality",
            "",
            "| Dataset | Analysis | PC1 var | PC2 var | PC1+PC2 var | PCs for 80% | PCs for 90% | Participation-ratio dimensions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {dataset} | {analysis} | {pc1} | {pc2} | {pc12} | {n80} | {n90} | {pr} |".format(
                dataset=row["dataset"],
                analysis=row["analysis"],
                pc1=fmt_float(row["pc1_variance_explained"]),
                pc2=fmt_float(row["pc2_variance_explained"]),
                pc12=fmt_float(row["pc1_pc2_cumulative_variance"]),
                n80=row["n_components_80pct"],
                n90=row["n_components_90pct"],
                pr=fmt_float(row["participation_ratio_effective_dimensions"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the PE-AE-excluded statistic remains above the block-preserving null, RNCP is not solely a PE-AE anticorrelation.",
            "- PCA/effective dimensionality should be interpreted as descriptive structure of the residual correlation matrix, not as a new inferential test.",
            "",
            "Generated files:",
            f"- `{summary_path.relative_to(ROOT)}`",
            f"- `{pair_path.relative_to(ROOT)}`",
            f"- `{eigen_path.relative_to(ROOT)}`",
            f"- `{loading_path.relative_to(ROOT)}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    for row in summary_rows:
        print(
            f"{row['dataset']} ({row['analysis']}): excluding PE-AE observed="
            f"{row['observed_mean_abs_r_excluding_pe_ae_5_pairs']:.4f}, null="
            f"{row['null_mean_abs_r_excluding_pe_ae_5_pairs']:.4f}, p="
            f"{row['empirical_p_excluding_pe_ae']:.4f}, PR dims="
            f"{row['participation_ratio_effective_dimensions']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
