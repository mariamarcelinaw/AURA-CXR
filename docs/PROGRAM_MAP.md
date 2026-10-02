# AURA-CXR Program Map

This document summarizes the public programs included in the repository.

## Core workflow

### `scripts/train_aura.py`
Main orchestration script for the AURA-CXR training workflow.

### `scripts/aura_dl_pair_selection.py`
Builds patient-level OOF predictions, evaluates deep-model pairs, fits A4/A8 fusion models, selects the operating threshold from development data, and records the frozen deployment configuration.

### `scripts/train_xrv_backbone.py`
Training and final development refit for the DenseNet121-XRV branch.

### `scripts/train_eva_x_backbone.py`
Training and final development refit for the EVA-X-S branch.

### `scripts/external_validation_kermany.py`
Runs frozen external evaluation on the Kermany pediatric test cohort without model refitting or threshold adaptation.

### `scripts/eval_only_xai.py`
Runs branch-scoped XAI evaluation, including localization and faithfulness analyses for the selected Hybrid CNN–ViT image branch.

### `scripts/q1_stats_report.py`
Statistical-reporting utilities used for quantitative evaluation summaries.

## Data and verification utilities

### `scripts/prepare_splits.py`
Creates or validates patient-level data splits from the official RSNA metadata.

### `scripts/verify_repository.py`
Checks repository configuration and expected files.

### `scripts/verify_deep_config.py`
Validates public deep-learning configuration metadata.

### `scripts/verify_radiomics_reproducibility.py`
Validates the released radiomics configuration and feature specification.

### `scripts/verify_xai.py`
Validates the released XAI configuration and scope rules.

## Python package

`src/aura_cxr/` contains reusable helpers for configuration, data handling, evaluation, fusion, model utilities, radiomics, training, and XAI.

## Reproducibility files

- `configs/` stores frozen configuration, seeds, checkpoint metadata, and example paths.
- `splits/` stores the exact RSNA patient-level split derivatives used by the public workflow.
- `results/reported_metrics.json` stores compact paper-facing metrics.
- `docs/` contains method-specific technical documentation.
