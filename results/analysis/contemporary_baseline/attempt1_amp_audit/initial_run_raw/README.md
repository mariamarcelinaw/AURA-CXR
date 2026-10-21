# Initial (mixed-precision) ConvNeXt V2 Tiny run: raw server outputs

Unmodified outputs of the first Stage-8 comparator run (AMP float16 autocast with GradScaler), released so
that both comparator runs can be recomputed from per-case predictions. This run failed quality control
(Fold 4 collapsed; see `../BUG_DISCOVERED_STAGE8.md`) and is not the reported comparator.

| File | Content |
|---|---|
| `06_BASELINE_LOCKED_PREDICTIONS.csv` | Per-case locked RSNA test probabilities (n = 5,338) at the run's development-derived threshold (0.470) |
| `08_BASELINE_OOF_PREDICTIONS.csv` | Per-case development OOF probabilities (n = 21,346); its SHA-256 equals `oof_sha256` in `05_BASELINE_DEPLOYMENT_LOCK.json` |
| `09_BASELINE_FOLD_RESULTS.csv`, `09_BASELINE_FOLD_RESULTS_RAW.csv`, `logs/fold_*.csv` | Fold-level results and epoch logs; Fold 4 shows the collapse (holdout ROC-AUC 0.7107, specificity 0) |
| `10_BASELINE_DEVELOPMENT_RESULTS.csv`, `11_BASELINE_LOCKED_RESULTS.csv` | Development OOF and locked-test metrics |
| `12_A8_BASELINE_PAIRED_COMPARISON.csv`, `bootstrap_differences.csv` | Paired comparison with frozen A8 (ROC-AUC difference 0.0027, 95% CI −0.0025 to 0.0080, paired DeLong p = 0.307) and its 2,000 bootstrap replicates |
| `02_*`, `04_*`, `05_*`, `14_*`, `final_refit_runtime.json`, `stage8_dynamic_summary.json`, `stage8_initial_amp.log` | Selection lock, training configuration, deployment lock, complexity, refit record, summary and run log |

The only edit is in `05_BASELINE_DEPLOYMENT_LOCK.json`, where the server path of the checkpoint is replaced by
`${AURA_PROJECT_DIR}`. `python scripts/analysis/paired_comparisons.py --check` recomputes the paired comparison
and the bootstrap replicates of this run from `06_BASELINE_LOCKED_PREDICTIONS.csv`.
