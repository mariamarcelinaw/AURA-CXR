# Run records

Records written by the training and evaluation scripts during the study run (directory `reports/` of the run), released so that the order of selection, locking, refitting and locked-test access can be checked. They are copied unchanged except that the run-directory prefixes were replaced by `${AURA_PROJECT_DIR}` and `${EVAX_REPO_DIR}`; `SANITIZATION.json` lists each file with the SHA-256 of the original and of the released copy, and the replaced prefixes, so that the content hashes stored in the records (lock identities) can be recomputed. `python aura_cxr.py verify` (line `RUN RECORDS`) performs these checks.

## Timeline (2026-07-30 unless stated)

| Time | Record | Event |
|---|---|---|
| 07-24 22:19 | `development_oof_fold_manifest.json` | Five patient-identifier-disjoint development folds fixed (seed 42) |
| 07-24 to 07-25 | `radiomics_screening_*`, `radiomics_global_screening_stability.csv` | Radiomics learner screening on 4,000 development rows |
| 08:16 | `oof_radiomics_*_manifest.json`, `radiomics_learner_selection_manifest.json`, `radiomics_fixed_candidate_oof_comparison.csv` | Refined radiomics OOF; LightGBM selected |
| 08:16 to 14:05 | `oof_{efficientnetv2,resnet50,xrv,eva_x}_manifest.json`, `*_oof_protocol.json`, `dl_candidate_oof_metrics.csv` | Deep-candidate OOF predictions |
| 14:05:55 | `dl_pair_selection_manifest.json`, `dl_pair_selection_validation.csv`, `meta_learner_features.json`, `fusion_thresholds_validation_tuned.json`, `q1_final_experiment_lock.json` | Pair, stack and thresholds selected from development OOF; **selection lock** |
| 14:57 to 15:06 | `q1_locked_deployment_manifest.json`, `selected_development_refit_audit.json`, `radiomics_development_refit_*.json`, `xrv_/eva_x_development_refit_manifest.json` | Full-development refits; **deployment lock** |
| 15:48 to 21:27 | `locked_test_radiomics_feature_manifest.json`, `*_exploratory_test_manifest.json`, `*_locked_test_inference.json`, `selected_pair_locked_test_inference_audit.json` | Label-free locked-test inference (selected and exploratory models) |
| 21:27:31 | `selected_pair_locked_test_fusion_audit.json`; `test_opened_at` in `q1_final_experiment_lock.json` | Frozen A8 fusion; **locked test opened** |
| 08-07 | `xai_cam_configuration_validation.json`, `xai_cam_layer_selection_validation.json`, `xai_cam_candidate_compatibility_validation.csv` | XAI settings selected on validation boxes only |
| 09-21 | `radiomics_contribution_oof_audit*.{json,csv}` | Post-lock, development-only audit of the radiomics contribution |

## What the check establishes

- `selection_lock_id` and `deployment_lock_id` are recomputed from the content of the pair-selection and deployment manifests and match the released configuration (`configs/specs/deployment_manifest.json`).
- `selection_manifest_hash` in `q1_final_experiment_lock.json` is reproduced from the pair-selection manifest with only its three test-time fields reset, so the selection content did not change after the lock.
- Every OOF and learner-selection record predates the selection lock; all refits fall between the selection lock and the opening of the locked test; the locked-test inference records state that labels were not used.
- The checkpoint hashes in the refit and deployment records equal those of the released checkpoints, and the locked-test stack hash equals the one in the released deployment manifest.
- The XAI configuration records state `selection_split: validation` and `test_set_used_for_selection: false`.

The development OOF predictions behind these records are in `results/predictions/development_oof.csv`; `scripts/analysis/recompute_selection.py --check` recomputes the selection from them.
