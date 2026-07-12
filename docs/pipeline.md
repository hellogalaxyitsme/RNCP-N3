# Analysis Pipeline

This repository contains the analysis code for the associated study:

**Residual neural-complexity profiles reveal graded internal structure and transition dynamics within N3 sleep**

The pipeline is organized as sequential analysis modules. Scripts write derived tables under the `project_data_root` specified in the JSON config files.

## Primary Sleep-EDF SC pipeline

Run from the repository root after installing dependencies and preparing a local config based on `configs/sleep_edf_example.json`.

```bash
python src/sleep_edf_audit.py --config configs/sleep_edf_local.json
python src/sleep_edf_stage_durations.py --config configs/sleep_edf_local.json
python src/sleep_edf_epoch_metadata.py --config configs/sleep_edf_local.json --only-usable-n3
python src/sleep_edf_signal_features.py --config configs/sleep_edf_local.json
python src/sleep_edf_complexity_features.py --config configs/sleep_edf_local.json
python src/sleep_edf_analysis_matrix.py --config configs/sleep_edf_local.json
python src/sleep_edf_residual_models.py --config configs/sleep_edf_local.json
python src/sleep_edf_rncp_reproducibility.py --config configs/sleep_edf_local.json
python src/sleep_edf_robustness.py --config configs/sleep_edf_local.json
```

## Replication, robustness, and functional anchoring

```bash
python src/sleep_edf_sleep_telemetry_replication.py --config configs/sleep_edf_local.json
python src/artifact_robustness.py --config configs/sleep_edf_local.json --dataset sleep_edf_sc
python src/artifact_robustness.py --config configs/sleep_edf_local.json --dataset sleep_edf_st
python src/functional_anchoring.py --config configs/sleep_edf_local.json --dataset all
```

ANPHY-Sleep analyses use a config based on `configs/anphy_example.json`:

```bash
python src/anphy_sleep_audit.py --config configs/anphy_local.json
python src/anphy_sleep_epoch_metadata.py --config configs/anphy_local.json --only-usable-n3
python src/anphy_sleep_n3_features.py --config configs/anphy_local.json
python src/anphy_sleep_analysis_matrix.py --config configs/anphy_local.json
python src/anphy_sleep_residual_models.py --config configs/anphy_local.json
python src/anphy_sleep_rncp_reproducibility.py --config configs/anphy_local.json
python src/artifact_robustness.py --config configs/anphy_local.json --dataset anphy
```

## Additional sensitivity analyses

The following scripts implement extended sensitivity checks. They are not required to reproduce the core pipeline, but are included for transparency.

```bash
python src/component_functional_anchoring.py --config configs/sleep_edf_local.json --dataset all
python src/specparam_fit_quality.py --config configs/sleep_edf_local.json
python src/high_permutation_uncertainty.py --config configs/sleep_edf_local.json --dataset sleep_edf_sc
python src/anphy_spatial_topography.py --config configs/anphy_local.json --residuals-file artifact_anphy_artifact_adjusted_rncp_residuals.csv.gz --out-prefix anphy_spatial_artifact_scalp
python src/within_bout_rncp_dynamics.py --config configs/sleep_edf_local.json --min-bout-min 5 --out-prefix within_bout_min5
python src/sc_night_to_night_stability.py --config configs/sleep_edf_local.json
python src/sc_age_rncp_relationship.py --config configs/sleep_edf_local.json
python src/anphy_fold_quality_control.py --config configs/anphy_local.json --out-prefix anphy_fold_quality_control
python src/reduced_feature_rncp_functional_test.py --config configs/sleep_edf_local.json --dataset all
python src/rncp_peae_pca_sensitivity.py
python src/rncp_sigma_covariate_sensitivity.py --config configs/sleep_edf_local.json
python src/rncp_lzc_bandpass_sensitivity.py --config configs/sleep_edf_local.json
```
