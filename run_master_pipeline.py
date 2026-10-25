"""
run_master_pipeline.py
----------------------
Master reproducible pipeline for the Cervical Cancer Methylation Feature Selection project.
Executes all core and advanced modules in sequence:
1. Data verification and setup
2. Evaluation audit with 95% CIs and Permutation Testing
3. Batch Confound Analysis (M-values, Leave-One-Chip-Out, chip-CpG association)
4. Feature Engineering (Literature baseline panel, Region promoter features, Cell composition deconvolution)
5. Stability Selection (100 bootstraps) and single-shot External Validation on GSE46306
6. Sensor Simulation (Diffusion kinetics, Pap-style dilution, combined stress heatmap)
7. Generative Augmentation Benchmark (TabDDPM, SMOTE, Gaussian noise)
8. Output compilation
"""

import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "feature_selection"))
sys.path.insert(0, os.path.join(ROOT, "sensor-sim"))


def log_step(step_num: int, title: str):
    bar = "=" * 75
    print(f"\n{bar}")
    print(f"STEP {step_num}: {title.upper()}")
    print(f"{bar}\n")


def main():
    t_start = time.time()
    print("=" * 75)
    print("STARTING COMPLETE MASTER REPRODUCIBLE PIPELINE (BRANCH: ritweek)")
    print("=" * 75)

    # Step 1: Batch Confound & M-value analysis
    log_step(1, "Batch Confound & M-Value Analysis (LOGO CV & Chip Associations)")
    from batch_analysis import run_batch_experiments
    run_batch_experiments(verbose=True)

    # Step 2: Literature Baseline & Region-Level Features & Cell Composition
    log_step(2, "Feature Engineering: Literature Baseline, Region Promoters & EpiDISH Deconvolution")
    from literature_and_regions import run_feature_engineering_experiments
    run_feature_engineering_experiments(verbose=True)

    # Step 3: Stability Selection & External Validation
    log_step(3, "Stability Selection (100 Bootstraps) & External Validation on GSE46306")
    from stable_panels_and_external import run_stable_panels_and_external_validation
    run_stable_panels_and_external_validation(verbose=True)

    # Step 4: Permutation Testing
    log_step(4, "Empirical Permutation Testing (100 Label Shuffles)")
    from run_permutation_test import run_permutation_benchmark
    run_permutation_benchmark(n_permutations=100, k=50, verbose=True)

    # Step 5: Sensor Simulation (Diffusion Kinetics & Pap Dilution)
    log_step(5, "Sensor Error Simulation: Diffusion Kinetics & Pap-Style Dilution")
    from sensor_kinetics_dilution_sim import run_sensor_experiments
    run_sensor_experiments(verbose=True)

    # Step 6: Generative Augmentation Benchmark
    log_step(6, "Generative Data Augmentation Benchmark (TabDDPM, SMOTE, Jitter)")
    from generative_augmentation import run_augmentation_experiments
    run_augmentation_experiments(verbose=True)

    elapsed = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("All results, tables, and plots have been generated under results/ and html/.")
    print("=" * 75)


if __name__ == "__main__":
    main()
