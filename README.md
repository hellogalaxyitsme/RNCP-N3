# RNCP-N3

Analysis code for **"Residual neural-complexity profiles reveal graded internal structure and transition dynamics within N3 sleep."**

This analysis tests whether visually scored human N3 sleep contains reproducible residual electrophysiological heterogeneity after accounting for age, sex, homeostatic timing, slow-wave burden, channel, and subject/night structure. The analysis defines a Residual Neural-Complexity Profile (RNCP) from four EEG-derived features: Lempel-Ziv complexity, permutation entropy, spectral entropy, and the aperiodic spectral exponent.

## Repository contents

- `src/`: analysis modules for Sleep-EDF SC, Sleep-EDF ST, ANPHY-Sleep replication, robustness checks, and functional anchoring.
- `configs/`: example JSON configs with placeholder paths.
- `docs/pipeline.md`: command-level overview of the reproduction pipeline.

## Datasets

The analysis uses public polysomnography datasets:

- Sleep-EDF Expanded from PhysioNet, including Sleep Cassette as the primary cohort and Sleep Telemetry as a replication cohort.
- ANPHY-Sleep as an external high-density EEG replication dataset.

Users are responsible for following the access terms for each source dataset.

## Environment

Python 3.10 was used for the analyses.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main scientific dependencies are NumPy, SciPy, pandas, MNE-Python, AntroPy, specparam, statsmodels, scikit-learn, and Numba.

## Quick start

Copy one of the example configs and edit the paths:

```bash
cp configs/sleep_edf_example.json configs/sleep_edf_local.json
```

Then run the Sleep-EDF primary pipeline:

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

See `docs/pipeline.md` for replication, artifact robustness, functional anchoring, and optional sensitivity analyses.

The repository also includes analysis scripts for component-wise anchoring, spectral-parameterization quality control, high-permutation uncertainty, ANPHY spatial topography, within-bout RNCP dynamics, night-to-night stability, age controls, ANPHY fold quality control, and reduced-feature RNCP functional tests.

## Reproducibility notes

Random seeds are fixed in the relevant scripts for null permutations, cross-validation folds, and sensitivity analyses. The repository intentionally excludes raw datasets, large generated intermediate outputs, and machine-specific helper scripts.
