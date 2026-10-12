# Sensor error scenario simulation

This folder contains a hypothetical robustness simulation for the selected
CpG-panel classifiers. It does not emulate or validate a physical sensor:
there are no real measurements from the intended device, so all error amounts
below are explicit assumptions.

## What it does

The script evaluates the Normal-versus-CIN3 task separately in GSE99511 and
GSE46306 using stratified five-fold cross-validation. For each panel size
(5, 10, 20, and 50), it chooses the selector/classifier with the highest
balanced accuracy in the existing benchmark, then refits the complete
feature-selection/model pipeline in each training fold. Each perturbation is
applied only to that fold's held-out samples. Selection, imputation, and
scaling stay within the training fold.

The chosen configurations are screened from an existing one-pass cross-
validation benchmark on the same small cohorts. This winner-selection step can
make apparent scores optimistic. Treat these results as an exploratory
sensitivity analysis, not independent model validation.

## Assumed scenarios

- Gaussian readout noise with standard deviation 0.01, 0.03, or 0.05 beta.
- A per-sample offset uniformly drawn from ±0.02 or ±0.05 beta, shared across
  that sample's panel.
- Per-sample gain error of 0.9–1.1× or 0.8–1.2× around beta 0.5.
- Linear drift from −0.02 to +0.02 or −0.05 to +0.05 beta across an arbitrary
  synthetic acquisition order (order is not observed and is unrelated to
  diagnosis).
- 5%, 10%, or 20% missing marker readings, replaced with training-fold
  medians.
- Combined moderate and severe settings. The moderate combination uses noise
  SD 0.03, offset/drift ±0.03, gain ±10%, and 10% missing. Severe uses noise
  SD 0.05, offset/drift ±0.05, gain ±20%, and 20% missing.

Every perturbation amount is illustrative. None is estimated from a paper or
from a real measurement system. Beta values are clipped to [0, 1].

## Run it

From the repository root, run:

    .\.venv\Scripts\python.exe .\sensor-sim\run_sensor_simulation.py
    .\.venv\Scripts\python.exe .\sensor-sim\make_report.py

The simulation uses 50 Monte Carlo draws per non-clean scenario. Outputs:

- `simulation_runs.csv`: metric values for each simulation draw.
- `simulation_summary.csv`: mean and standard deviation by dataset, panel,
  selected method, and scenario.
- `model_choices.csv`: the benchmark-selected configurations used.
- `../html/sensor_simulation_results.html`: self-contained report for viewing.

GSE46306 has HPV-negative Normal controls and HPV-positive CIN3 cases, so HPV
status is confounded with diagnosis. Interpret its model scores cautiously.

## Correct interpretation

Use the scenario results to identify panels that appear less sensitive to
assumed perturbations and to plan future bench measurements. They cannot
establish actual sensor noise, calibration, drift, missing-read rates,
clinical performance, or reliability. The intended assay/device must be
measured on repeated reference samples, and its error distributions should
replace these assumptions before drawing device-level conclusions.
