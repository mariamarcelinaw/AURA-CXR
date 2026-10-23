# Initial (mixed-precision) ConvNeXt V2 Tiny run: audit record

The first Stage-8 run of the ConvNeXt V2 Tiny comparator used AMP (float16 autocast with GradScaler).
Outer Fold 4 collapsed (non-finite gradients, near-constant probabilities). A development-only
diagnostic on Fold 4, with the locked test disabled, reproduced the failure under AMP and showed
normal learning under FP32. All five folds and the final refit were therefore rerun in FP32 with
no other change. The rerun was decided on development evidence, but after the locked-test output of
the initial run had been viewed; the corrected run is the reported comparator, and this folder keeps
the initial run's record so that both comparisons can be inspected.

| File | Content |
|---|---|
| `ATTEMPT_1_AUDIT_SUMMARY.json` | Initial run: locked-test metrics (n = 5,338) and the paired DeLong comparison with frozen A8 |
| `BUG_DISCOVERED_STAGE8.md` | Description of the Fold-4 failure and of the development-only diagnosis |
| `diagnostic_summary.json`, `fold4_amp_repeat_diagnostic.csv`, `fold4_fp32_diagnostic.csv` | Fold-4 diagnostic (six epochs, AMP repeat versus FP32; locked test not accessed) |
| `CORRECTIVE_RERUN_LOCK.json` | Lock of the corrective change (precision only) written before the second locked-test evaluation |

Paired ROC-AUC comparison of frozen A8 with the comparator (A8 minus comparator, paired DeLong):

| Comparator run | Comparator ROC-AUC | Difference | 95% CI | p |
|---|---|---|---|---|
| Corrected FP32 run (reported) | 0.8922 | 0.0066 | 0.0012 to 0.0121 | 0.018 |
| Initial AMP run (failed QC) | 0.8960 | 0.0027 | −0.0025 to 0.0080 | 0.307 |

The raw outputs of the initial run, including its per-case locked-test and OOF predictions, are in
`initial_run_raw/`. `python scripts/analysis/paired_comparisons.py --check` recomputes both rows of the
table above from per-case predictions and compares them with the records produced at the time, and
`python aura_cxr.py verify` checks that the initial-run record is internally consistent and that its
locked-test ROC-AUC is reproduced from `initial_run_raw/06_BASELINE_LOCKED_PREDICTIONS.csv`.
