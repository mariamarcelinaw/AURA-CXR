# Corrected FP32 ConvNeXt V2 Tiny run: raw server outputs

Unmodified outputs of the reported (corrected, FP32) comparator run, kept as an audit trail next to
the release files in the parent folder.

| File | Content |
|---|---|
| `stage8_corrected_fp32.log` | Run log: preflight, five outer folds (epoch-wise inner-validation AUC and early stopping), five-epoch full-development refit, locked-test metrics, and the paired DeLong comparison with frozen A8, in that order |
| `09_BASELINE_FOLD_RESULTS_RAW.csv` | Per-fold best and stopping epochs, holdout AUC, runtime, and SHA-256 hashes of the initial and best fold checkpoints |
| `final_refit_runtime.json` | Refit epochs (5 = rounded median of the fold best epochs 4, 5, 5, 4, 17), runtime, and final checkpoint hash |
| `stage8_dynamic_summary.json` | Run summary (development, locked test, paired comparison, complexity, protocol flags) |
| `bootstrap_differences.csv` | The 2,000 paired bootstrap replicates (A8 minus comparator) behind the secondary-metric CIs |

Checks against the release: the locked-test and OOF predictions and the fold results are byte-identical
to the files in the parent folder; the paired comparison is numerically identical; and
`python scripts/analysis/paired_comparisons.py --check` reproduces `bootstrap_differences.csv` from the
per-case predictions. In the log, Fold 4 epochs 1-6 match the FP32 arm of the development-only
diagnostic in `../attempt1_amp_audit/` (inner-validation AUC 0.4799, 0.6022, 0.7428, 0.7866, 0.8101, 0.8309).
The deployment lock of this run is released as `../05_BASELINE_DEPLOYMENT_LOCK.json`, with the server path
replaced by `${AURA_PROJECT_DIR}`.
