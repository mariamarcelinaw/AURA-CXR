# AURA-CXR

Development-locked Hybrid CNN–ViT and Adaptive Radiomics stacking with quantitative XAI for pneumonia detection on chest X-rays.

**Paper:** Widyastuti, M. M., Fatichah, C., and Sunaryono, D. *AURA-CXR: Development-Locked Hybrid CNN–ViT and Adaptive Radiomics Stacking with Quantitative XAI for Pneumonia Detection.* International Journal of Intelligent Engineering and Systems, 2026. DOI pending.

> Research use only. AURA-CXR is not a medical device and must not be used for clinical decisions.

## Overview

AURA-CXR combines a DenseNet121-XRV + EVA-X-S image branch with an adaptive radiomics branch (entropy-guided wavelet selection, GLCM/LBP features, LightGBM) and fuses them with a logistic-regression stack (configuration A8). All models, fusion coefficients and the decision threshold were fixed on development out-of-fold data before the locked RSNA test set was evaluated; the Kermany pediatric test set was used only for frozen external evaluation. The repository contains the code, frozen configurations, patient-level splits and per-case predictions needed to check every reported number without raw images or a GPU.

## Repository structure

```
AURA-CXR/
├── aura_cxr.py          # verify / reproduce
├── THIRD_PARTY_NOTICES.md  # upstream licenses, changes made, data attribution
├── LICENSES/            # Apache-2.0 text (TorchXRayVision)
├── run_aura_cxr.sh      # full pipeline from raw images (GPU)
├── checkpoints/         # downloaded release weights (not tracked)
├── configs/             # frozen configuration, checkpoint registry, seeds, XAI settings
│   ├── specs/           # deep-training, radiomics, MRFO and deployment specifications
│   └── analysis/        # configurations of the post-hoc analyses
├── data/splits/         # patient-level train/val/test splits and 5 OOF folds
├── src/aura_cxr/        # fusion, evaluation, radiomics, XAI and data utilities
├── scripts/             # training, model selection, XAI and external-validation code
│   └── analysis/        # post-hoc analyses (baseline, wavelet ablation, XAI statistics, paired comparisons)
├── results/
│   ├── metrics/         # reported metrics and the result registry with CIs and p-values
│   ├── predictions/     # per-case locked-test, external and XAI outputs
│   ├── analysis/        # outputs of the post-hoc analyses
│   └── audit/           # golden reference, provenance, QC records, reproduction output
└── tests/
```

## Setup

Python 3.13 was used for the reported results.

```bash
pip install -r requirements.txt
```

The pinned packages are enough for verification and metric reproduction on a CPU. Retraining additionally needs TensorFlow, PyTorch, TorchXRayVision and timm; the versions used are listed at the end of `requirements.txt` and in `configs/specs/environment_lock.json`.

## Contents at a glance

| Item | Location |
|---|---|
| Patient-level split identifiers (train/validation/locked test) and five-fold OOF assignment | `data/splits/` (`manifest.json` lists counts and hashes) |
| Training and evaluation code | `scripts/` (order of execution in `run_aura_cxr.sh`), shared utilities in `src/aura_cxr/` |
| Radiomics feature extraction | `src/aura_cxr/radiomics.py`, `scripts/train_aura.py`; 746-feature schema in `configs/specs/feature_schema_746.csv`, settings in `configs/specs/radiomics_specification.json` |
| MRFO search space, budgets and objective | `configs/specs/mrfo_search_space.csv`, `configs/frozen_config.yaml` (`radiomics.mrfo`) |
| Random seeds | `configs/seeds.json` |
| Checkpoint identifiers, trained checkpoints and their SHA-256 | `configs/checkpoints.json`, `configs/release_weights.sha256`, `configs/specs/deep_11_checkpoint_provenance.csv`; files in GitHub release v1.0 |
| Frozen stack coefficients, feature order and threshold (A8) | `configs/specs/deployment_manifest.json` |
| Deep-training specification (normalization, loss, augmentation, schedules, freezing, refit rule) | `configs/specs/deep_training_specification.json`, `configs/specs/deep_*.csv` |
| XAI layers, weights, token handling and thresholds | `configs/xai_frozen.json` |
| Case-level predictions (locked RSNA test, Kermany, XAI) | `results/predictions/` |
| Reported numbers with CIs and p-values | `results/metrics/result_registry.csv`, `results/analysis/` |

## Data

Raw images are not redistributed (see *Availability and restrictions*). Obtain them from the original distributors and place them under `data/raw/` (ignored by git), or set other paths in `configs/paths.example.yaml` and `run_aura_cxr.sh`:

- **RSNA Pneumonia Detection Challenge (2018)**, stage-2 training set: `stage_2_train_labels.csv` and `stage_2_train_images/` (26,684 patients). Available from the RSNA AI challenge page and the Kaggle competition page, under the RSNA terms of use and attribution (citations, including the NIH ChestX-ray source, in `THIRD_PARTY_NOTICES.md`).
- **Kermany pediatric chest X-ray data**, Mendeley Data, doi:10.17632/rscbjbr9sj.2: the official `chest_xray/test` partition (234 NORMAL, 390 PNEUMONIA), used only for frozen external evaluation.

`data/splits/` contains only patient identifiers, labels and fold assignments. `scripts/prepare_splits.py` rebuilds the full split files used by the training scripts from `stage_2_train_labels.csv`, keeping the released patient membership, order, `sample_id` and labels; the five-fold assignment regenerates exactly from seed 42 (checked by `tests/test_prepare_splits.py`).

## Availability and restrictions

| Material | Status | Reproducible alternative |
|---|---|---|
| Code, configurations, splits, seeds, frozen A8 parameters, XAI settings, case-level predictions, analysis outputs | Released in this repository | — |
| RSNA and Kermany images | Not redistributed: they are governed by their distributors' terms | Official sources above; `prepare_splits.py` restores the exact split from the official label file |
| Upstream pretrained checkpoints (TorchXRayVision `densenet121-res224-all`, EVA-X-S `eva_x_small_patch16_merged520k_mim.pt`, Keras ImageNet weights) | Not redistributed: they are third-party files | Public downloads listed below (TorchXRayVision: Apache-2.0; EVA-X-S weights: MIT, per the model card) |
| Trained AURA-CXR checkpoints (DenseNet121-XRV, EVA-X-S, LightGBM radiomics) | Released as assets of GitHub release [v1.0](https://github.com/mariamarcelinaw/AURA-CXR/releases/tag/v1.0), not in the git tree (see *Trained checkpoints*) | — |
| A8 meta-learner | Fully specified by its released intercept, coefficients, feature order and threshold (`configs/specs/deployment_manifest.json`, `src/aura_cxr/fusion.py`) | — |

Upstream checkpoints:

| Backbone | Identifier | Download |
|---|---|---|
| DenseNet121-XRV | `densenet121-res224-all` (TorchXRayVision 1.5.2) | Downloaded automatically by `xrv.models.DenseNet(weights="densenet121-res224-all")` from https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt (public file SHA-256 `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`, checked 2026-10-21) |
| EVA-X-S | `eva_x_small_patch16_merged520k_mim.pt` | https://huggingface.co/MapleF/eva_x (model code: https://github.com/hustvl/EVA-X, `eva_x.py`) |
| EfficientNetV2S, ResNet50 | Keras ImageNet weights | Downloaded automatically by `tf.keras.applications` |

## Trained checkpoints

The three frozen components of A8, exactly as used for the locked RSNA test and the Kermany evaluation, are assets of GitHub release [v1.0](https://github.com/mariamarcelinaw/AURA-CXR/releases/tag/v1.0):

| File | Component | Size | SHA-256 |
|---|---|---|---|
| `xrv_development_refit.pt` | DenseNet121-XRV, full-development refit (2 epochs, dropout 0.3, 2-class head) | 27.1 MiB | `ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a` |
| `eva_x_development_refit.pt` | EVA-X-S, full-development refit (2 epochs, dropout 0.2, 12 blocks, 384-d tokens) | 82.7 MiB | `83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6` |
| `radiomics_mrfo_development_refit.pkl` | LightGBM radiomics classifier (scikit-learn Pipeline, 746 features, 400 trees) | 1.7 MiB | `15d862c5c9ddfd088aebdbfd146b4cb133c089f577cd09ed73bf71058ab1c1b7` |

```bash
mkdir -p checkpoints && cd checkpoints
for f in xrv_development_refit.pt eva_x_development_refit.pt radiomics_mrfo_development_refit.pkl; do
  curl -LO https://github.com/mariamarcelinaw/AURA-CXR/releases/download/v1.0/$f
done
sha256sum -c ../configs/release_weights.sha256 && cd ..
python aura_cxr.py verify        # the WEIGHTS line confirms the three hashes
```

`checkpoints/` is not tracked by git. The PyTorch files hold a dictionary with `state_dict`, `dropout`, `refit_epochs`, `optimizer_updates`, `image_size` and `parameter_count`, and load with `torch.load(path, map_location="cpu", weights_only=True)`. They are loaded into the architectures built by `build_model` in `scripts/train_xrv_backbone.py` (with `weights=None`) and `scripts/train_eva_x_backbone.py` (which needs the official EVA-X code and upstream checkpoint); `xrv_infer` and `load_xrv_binary_state_dict_strict` in `scripts/external_validation_kermany.py` show the exact loading used for the external evaluation, including the one unused TorchXRayVision buffer (`backbone.op_threshs`) that is dropped before strict loading. The radiomics file loads with `joblib.load` under the pinned scikit-learn, LightGBM and joblib versions and takes the 746-feature vector of `src/aura_cxr/radiomics.py`; as a pickle, it should be loaded only after its hash has been checked. The EVA-X-S file records, in its `checkpoint` field, the original server path of the upstream file; the checkpoints are released unmodified so that they match the recorded hashes. Use of the weights is subject to the upstream licenses (TorchXRayVision: Apache-2.0; EVA-X-S weights: MIT) and the RSNA terms of use and attribution; the notices, changes made and required citations are in `THIRD_PARTY_NOTICES.md`.

## Running the full pipeline

`run_aura_cxr.sh` runs the study from raw images in its original order: split preparation, development-only radiomics and deep OOF predictions, pair selection and stacking (selection lock), full-development refit (deployment lock), a single locked-test inference and fusion, frozen Kermany evaluation, and branch-scoped XAI. Every stage before the pair lock reads only `train.csv` and `val.csv`, and the locked test can be opened only once per run directory. It needs a CUDA GPU (the study used one NVIDIA A100 40 GB), the retraining packages, the official EVA-X repository and checkpoint; all options of each script are listed by `--help`.

```bash
RSNA_LABELS=... RSNA_DICOM=... KERMANY_DIR=... EVAX_REPO=... EVAX_CKPT=... bash run_aura_cxr.sh
```

### Post-hoc analyses

These analyses were run after the locked evaluation and never change A8. The CPU-only checks recompute the reported statistics from released files; the others regenerate the inputs from images.

| Analysis (paper section) | Script | Settings | Released outputs |
|---|---|---|---|
| A8 versus A4, A8 versus EVA-X-S, A8 versus ConvNeXt V2 Tiny (5.2, 5.5, 5.6) | `scripts/analysis/paired_comparisons.py --check` (CPU) | — | `results/analysis/a4_vs_a8/`, `results/analysis/contemporary_baseline/` |
| Holm-corrected pairwise DeLong family (5.2) | `scripts/analysis/locked_pairwise_delong.py --check` (CPU) | — | `results/analysis/locked_pairwise_delong/` |
| Adaptive versus fixed wavelets (5.3) | `scripts/analysis/wavelet_ablation_run.py`, then `wavelet_ablation_statistics.py` | `configs/analysis/wavelet_ablation.json` | `results/analysis/wavelet_ablation/` |
| Same-split ConvNeXt V2 Tiny comparator (5.5) | `scripts/analysis/convnextv2_baseline.py` | `configs/analysis/contemporary_baseline_training.json` | `results/analysis/contemporary_baseline/` |
| EVA-X-S LayerCAM versus Integrated Gradients (5.4) | `scripts/analysis/transformer_xai_analysis.py` | `configs/analysis/transformer_xai.json`, `transformer_xai_comparator.json` | `results/analysis/transformer_xai/` |
| XAI multiplicity families (5.4) | `scripts/analysis/xai_multiplicity.py` | `configs/analysis/xai_statistics.json` | `results/analysis/xai_statistics/` |

## Reproducibility

```bash
python aura_cxr.py verify
python aura_cxr.py reproduce
python scripts/analysis/paired_comparisons.py --check
python scripts/analysis/locked_pairwise_delong.py --check
```

`verify` checks file integrity against `SHA256SUMS.txt`, the frozen configuration, split and prediction counts, analysis outputs, provenance, the absence of credentials, raw data and weights in the git tree, and, when they have been downloaded to `checkpoints/`, the SHA-256 of the released trained checkpoints; `reproduce` recomputes the locked-test A4/A8 and Kermany A8 metrics from `results/predictions/` and compares them with `results/metrics/reported_metrics.json`; `paired_comparisons.py` recomputes the paired A8-versus-A4 (locked RSNA test and Kermany), A8-versus-EVA-X-S (Kermany), A8-versus-ConvNeXt V2 Tiny (both comparator runs) and encoder-versus-ConvNeXt V2 Tiny comparisons (paired DeLong for ROC-AUC, 2,000 paired class-stratified bootstrap resamples for the other metrics) from the same per-case predictions, writes them to `runs/paired_comparisons/`, and with `--check` compares them with the released results in `results/analysis/`. `locked_pairwise_delong.py` recomputes the original Holm-corrected family of ten pairwise DeLong comparisons among DenseNet121-XRV, EVA-X-S, LightGBM, soft voting and A8 on the locked test (A8 versus each encoder is reported in Section 5.2 of the paper) and checks it against `results/analysis/locked_pairwise_delong/pairwise_delong_holm.csv`. The record of the comparator's initial mixed-precision run, which was replaced by the FP32 rerun, is in `results/analysis/contemporary_baseline/attempt1_amp_audit/`, with its raw outputs and per-case locked-test and OOF predictions in `initial_run_raw/`; `verify` checks that the record is internally consistent and reproduces its locked-test ROC-AUC from the per-case predictions. The raw server outputs of the reported FP32 comparator run (log, fold-level results, refit record and bootstrap replicates) are in `results/analysis/contemporary_baseline/fp32_run_raw/`. The same script also compares each selected encoder (DenseNet121-XRV and EVA-X-S) with ConvNeXt V2 Tiny on the locked test (`results/analysis/contemporary_baseline/15_COMPONENT_VS_BASELINE.csv`); the upstream pretraining data of every backbone are listed in `configs/specs/upstream_pretraining_data.csv`.

## Notes on the records

- **Method names.** The quantitative attribution reported in the paper is LayerCAM with the layers, weights and token handling in `configs/xai_frozen.json`. Some option names, comments and the free-text column `xai_failure_reason` in `results/predictions/` keep the generic label "GradCAM" from the code's history; they refer to the same maps.
- **EVA-X-S structure.** The training-time specifications (`configs/frozen_config.yaml`, `configs/specs/deep_training_specification.json`, `configs/specs/deep_17_eva_model_structure.csv`) were written before the XAI analysis and mark the block count, embedding dimension and class token as not captured. They were verified afterwards on the frozen checkpoint: 384-dimensional embeddings, one leading class token and 196 patch tokens in row-major order (`results/analysis/transformer_xai/token_structure.json`, `configs/xai_frozen.json`); the official `eva_x_small_patch16` definition has 12 blocks, so the hooked `blocks.11` is the final block. The local file name `eva_x_s_16.pt` in some records is the downloaded official `eva_x_small_patch16_merged520k_mim.pt`.
- **Upstream checkpoint hashes.** The upstream files were loaded from library caches (TorchXRayVision, Keras) or a local copy (EVA-X-S), and their SHA-256 values were not recorded during training (`UNRESOLVED_FROM_AVAILABLE_PROVENANCE`); the SHA-256 values of the trained checkpoints were recorded, are listed in `configs/checkpoints.json`, and match the released files.
- **Configuration file hashes.** `frozen_config_file_sha256` in `configs/specs/deployment_manifest.json` and `results/audit/golden_reference.json`, and the hashes in some `input_hashes.json` files, identify the configuration file as it existed when those records were written. The released `configs/frozen_config.yaml` is a later, documentation-expanded version; the values that define A8 (meta-feature order, coefficients and threshold) are identical in this file, the deployment manifest, the golden reference and `src/aura_cxr/fusion.py`, and `python aura_cxr.py verify` checks this together with the trained-checkpoint hashes.
- **Path placeholders.** Machine-specific directories in the records are replaced by `${WORK_DIR}`, `${AURA_PACKAGE_DIR}` or `${AURA_PROJECT_DIR}`.

## Results

Metrics at the frozen thresholds, read from `results/metrics/reported_metrics.json`:

| Test set | Configuration | ROC-AUC | AUPRC | Sensitivity | Specificity | MCC |
|---|---|---|---|---|---|---|
| Locked RSNA test (n = 5,338) | A8 (frozen stack) | 0.8988 | 0.7338 | 0.8512 | 0.7715 | 0.5392 |
| Locked RSNA test (n = 5,338) | A4 (XRV + EVA-X-S mean) | 0.8996 | 0.7365 | 0.8520 | 0.7768 | 0.5458 |
| Kermany pediatric test (n = 624) | A8 (frozen stack) | 0.9309 | 0.9546 | 0.7949 | 0.9274 | 0.7001 |

A8 is reported because it was the configuration locked before testing; the deep-only A4 mean scored slightly higher in ROC-AUC on both test sets (Kermany A4: 0.9368, A8 − A4 −0.0060, paired DeLong p < 0.001; `results/analysis/a4_vs_a8/kermany_a8_vs_a4.csv`), whereas A8 had a lower Brier score on both, and the radiomics branch added no incremental discrimination. Confidence intervals, p-values and the remaining analyses are in `results/metrics/result_registry.csv` and `results/analysis/`.

## Citation

```bibtex
@article{widyastuti_aura_cxr_2026,
  author  = {Widyastuti, Maria Marcelina and Fatichah, Chastine and Sunaryono, Dwi},
  title   = {{AURA-CXR}: Development-Locked Hybrid {CNN--ViT} and Adaptive Radiomics Stacking with Quantitative {XAI} for Pneumonia Detection},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026}
}
```

## License

Code: MIT License (see `LICENSE`). External datasets and upstream pretrained weights remain under their own licenses and are not redistributed here. The trained checkpoints in release v1.0 are derived from the upstream weights and the RSNA data, and their use is subject to those licenses and terms; see `THIRD_PARTY_NOTICES.md` (upstream licenses, changes made and data attribution) and `LICENSES/Apache-2.0.txt`.
