# BUG_DISCOVERED_STAGE8 — AMP numerical instability in Fold 4

## Status
Confirmed by a development-only corrective diagnostic. The locked RSNA test was disabled in the diagnostic.

## Attempt 1 observation
Stage-8 attempt 1 completed, but outer Fold 4 showed a qualitatively different optimization trajectory from Folds 0–3. Fold 4 selected epoch 1 and subsequently produced near-constant probabilities.

The attempt-1 locked-test result is retained for audit; it is not the reported comparator result, and both runs are reported in the paper (Section 5.5).

## Development-only reproduction
The same Fold 4, same seed (46), same split, same architecture, same preprocessing, same optimizer, same learning rate, and same augmentation were rerun for six epochs under two precision modes.

### AMP repeat
- Epoch 1 non-finite gradient batches: 3
- Epoch 1 AMP scale drops: 3
- Epoch 1 maximum observed gradient norm: Infinity
- Best validation AUC: 0.716812
- Epoch 6 validation AUC: 0.554064
- Epoch 6 probability SD: 0.00002278

### FP32
- Non-finite gradient batches: 0 in all six diagnostic epochs
- Validation AUC rose from 0.479897 to 0.830918 by epoch 6
- Epoch 6 probability SD: 0.124661

## Diagnosis
The evidence supports AMP-related numerical instability as the technical cause of the Fold-4 collapse. The diagnosis is based only on development data and does not depend on the locked-test result.

## Corrective action
A full five-fold corrected rerun is required in FP32, not merely a Fold-4 replacement. This keeps precision uniform across all five folds and the final development refit.

The baseline identity, splits, labels, base preprocessing, augmentation, loss, optimizer, LR, scheduler, batch size, early-stopping rule, threshold-selection rule, and final-refit rule remain unchanged.

## Test-access disclosure
The locked test had already been evaluated in attempt 1 before this numerical failure was fully diagnosed. Therefore attempt 2 is explicitly labeled a corrective rerun. Its correction was locked from a development-only diagnostic before the second locked-test evaluation. Attempt-1 outputs must remain preserved.
