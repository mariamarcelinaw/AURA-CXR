# Reproducibility Guide

AURA-CXR separates three reproducibility levels.

## 1. Audit reproduction — no raw images or checkpoints required

Use released prediction tables:

```bash
python scripts/validate_public_release.py
python aura_cxr.py verify
python aura_cxr.py reproduce --mode audit
```

This path checks split integrity, frozen A8 configuration, XAI scope, and reproduces the released internal/external metrics from the public probabilities.

## 2. Checkpoint-based evaluation

Exact inference requires:

- original RSNA/Kermany image data obtained from official providers;
- the correct upstream pretrained models;
- the derived/frozen checkpoint bytes matching the documented identities/hashes where available;
- the frozen preprocessing and deployment configuration.

Checkpoint bytes are not redistributed in this repository unless redistribution rights are established.

## 3. Full training

Full model training is computationally expensive. Use the source lineage under `reproducibility/source_snapshot/` together with:

- `configs/frozen_config.yaml`;
- `configs/seeds.json`;
- `splits/development_oof_folds.csv`;
- `reproducibility/deep_training_specification.json`;
- `reproducibility/radiomics_specification.json`;
- `reproducibility/deployment_manifest.json`.

## Evaluation firewall

The locked RSNA test and external Kermany cohort are evaluation-only. They must not be used for model selection, hyperparameter optimization, threshold tuning, or recalibration.

Public post-hoc analyses under `analysis/` are marked non-selection-eligible and do not alter A8.

## Determinism and limitations

Exact bitwise replay across different GPU/driver/library stacks is not guaranteed. The repository records seeds, software versions/configurations, checkpoint identities, and machine-readable result fingerprints to support scientific rather than hardware-level reproducibility.

See `docs/MODEL_CHECKPOINTS.md`, `docs/PROVENANCE.md`, and the public SHA-256 manifest for additional provenance information.
