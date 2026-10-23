# AURA-CXR

Development-locked Hybrid CNN–ViT and Adaptive Radiomics stacking with quantitative XAI for pneumonia detection on chest X-rays.

**Paper:** Widyastuti, M. M., Fatichah, C., and Sunaryono, D. *AURA-CXR: Development-Locked Hybrid CNN–ViT and Adaptive Radiomics Stacking with Quantitative XAI for Pneumonia Detection.* International Journal of Intelligent Engineering and Systems, 2026. DOI pending.

> Research use only. AURA-CXR is not a medical device and must not be used for clinical decisions.

## Overview

AURA-CXR combines a DenseNet121-XRV + EVA-X-S image branch with an adaptive radiomics branch (entropy-guided wavelet selection, GLCM/LBP features, LightGBM) and fuses them with a logistic-regression stack (configuration A8). All models, fusion coefficients and the decision threshold were fixed on development out-of-fold data before the locked RSNA test set was evaluated; the Kermany pediatric test set was used only for frozen external evaluation. The repository contains the code, frozen configurations, splits, development out-of-fold and per-case test predictions, the original run records and the trained weights needed to check every reported number and every selection step; the checks below need neither raw images nor a GPU.

## Quick start

Python 3.13 was used for the reported results. The pinned packages need Python ≥ 3.11 (the checks below were also run with Python 3.12).

```bash
pip install -r requirements.txt
python aura_cxr.py verify       # integrity, frozen configuration, splits, predictions, analyses -> VERIFY = PASS
python aura_cxr.py reproduce    # recomputes the reported A4/A8 and Kermany metrics -> REPRODUCE = PASS
```

To also check the trained weights, download them first (see *Trained checkpoints*) and run `verify` again.

## Results

Metrics at the frozen thresholds, read from `results/metrics/reported_metrics.json`:

| Test set | Configuration | ROC-AUC | AUPRC | Sensitivity | Specificity | MCC |
|---|---|---|---|---|---|---|
| Locked RSNA test (n = 5,338) | A8 (frozen stack) | 0.8988 | 0.7338 | 0.8512 | 0.7715 | 0.5392 |
| Locked RSNA test (n = 5,338) | A4 (XRV + EVA-X-S mean) | 0.8996 | 0.7365 | 0.8520 | 0.7768 | 0.5458 |
| Kermany pediatric test (n = 624) | A8 (frozen stack) | 0.9309 | 0.9546 | 0.7949 | 0.9274 | 0.7001 |

A8 is reported because it was the configuration locked before testing. The deep-only A4 mean scored slightly higher in ROC-AUC on both test sets (Kermany A4: 0.9368, A8 − A4 −0.0060, paired DeLong p < 0.001; `results/analysis/a4_vs_a8/kermany_a8_vs_a4.csv`), whereas A8 had a lower Brier score on both, and the radiomics branch added no incremental discrimination. Confidence intervals, p-values and the remaining analyses are in `results/metrics/result_registry.csv` and `results/analysis/`.

## Contents at a glance

| Item | Location |
|---|---|
| Split identifiers (RSNA `patientId`, which is image-level; train/validation/locked test) and five-fold OOF assignment | `data/splits/` (`manifest.json` lists counts and hashes) |
| Development out-of-fold predictions of all candidates (inputs of every selection step) | `results/predictions/development_oof.csv` |
| Original run records (selection and deployment locks, refits, locked-test access, XAI selection) | `results/audit/run_records/` |
| Training and evaluation code | `scripts/` (order of execution in `run_aura_cxr.sh`), shared utilities in `src/aura_cxr/` |
| Radiomics feature extraction | `src/aura_cxr/radiomics.py`, `scripts/train_aura.py`; 746-feature schema in `configs/specs/feature_schema_746.csv`, settings in `configs/specs/radiomics_specification.json` |
| MRFO search space, budgets and objective | `configs/specs/mrfo_search_space.csv`, `configs/frozen_config.yaml` (`radiomics.mrfo`) |
| Random seeds | `configs/seeds.json` |
| Checkpoint identifiers, trained checkpoints and their SHA-256 | `configs/checkpoints.json`, `configs/release_weights.sha256`, `configs/specs/deep_11_checkpoint_provenance.csv`; files in GitHub release v1.0 |
| Frozen stack coefficients, feature order and threshold (A8) | `configs/specs/deployment_manifest.json` |
| Deep-training specification (normalization, loss, augmentation, schedules, freezing, refit rule) | `configs/specs/deep_training_specification.json`, `configs/specs/deep_*.csv` |
| XAI layers, weights, token handling and thresholds | `configs/xai_frozen.json` |
| Case-level predictions (locked RSNA test for all A1–A10 configurations, Kermany, XAI) | `results/predictions/` (A1–A10: `locked_rsna_a1_a10.csv`) |
| Reported numbers with CIs and p-values | `results/metrics/result_registry.csv`, `results/analysis/` |

## Data and availability

Raw images are not redistributed. Obtain them from the original distributors and place them under `data/raw/` (ignored by git), or set other paths in `configs/paths.example.yaml` and `run_aura_cxr.sh`:

- **RSNA Pneumonia Detection Challenge (2018)**, stage-2 training set: `stage_2_train_labels.csv` and `stage_2_train_images/` (26,684 patientIds), obtained for this study from the Kaggle competition page (https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data; the RSNA AI challenge page links the same data), under the RSNA terms of use and attribution (required citations, including the NIH ChestX-ray source, in `THIRD_PARTY_NOTICES.md`).
- **Kermany pediatric chest X-ray data**, Mendeley Data, doi:10.17632/rscbjbr9sj.2: the official `chest_xray/test` partition (234 NORMAL, 390 PNEUMONIA), used only for frozen external evaluation.

`data/splits/` contains only RSNA identifiers, labels and fold assignments. The RSNA `patientId` field identifies an image, not a person (Kaggle: "Each patientId corresponds to a unique image"); the 26,684 patientIds map to 11,452 NIH patients, so the splits are `patientId`-disjoint but not patient-disjoint (see *NIH patient overlap* below). `scripts/prepare_splits.py` rebuilds the full split files used by the training scripts from `stage_2_train_labels.csv`, keeping the released `patientId` membership, order, `sample_id` and labels. The train/validation/locked-test membership itself is the stratified 70/10/20 random split of `split_dataframe` in `scripts/train_aura.py` with seed 42, so the locked test was drawn at random rather than chosen; both this partition and the five-fold assignment regenerate exactly from seed 42 (checked by `tests/test_prepare_splits.py`).

| Material | Status | Reproducible alternative |
|---|---|---|
| Code, configurations, splits, seeds, frozen A8 parameters, XAI settings, case-level predictions, analysis outputs | Released in this repository | — |
| Trained AURA-CXR checkpoints (DenseNet121-XRV, EVA-X-S, LightGBM radiomics) | Released as assets of GitHub release [v1.0](https://github.com/mariamarcelinaw/AURA-CXR/releases/tag/v1.0) | — |
| A8 meta-learner | Fully specified by its released intercept, coefficients, feature order and threshold (`configs/specs/deployment_manifest.json`, `src/aura_cxr/fusion.py`) | — |
| RSNA and Kermany images | Not redistributed: governed by their distributors' terms | Official sources above; `prepare_splits.py` restores the exact split from the official label file |
| Upstream pretrained checkpoints | Not redistributed: third-party files | Public downloads below |

| Upstream backbone | Identifier | Download | License |
|---|---|---|---|
| DenseNet121-XRV | `densenet121-res224-all` (TorchXRayVision 1.5.2) | Automatic via `xrv.models.DenseNet(weights="densenet121-res224-all")` | Apache-2.0 |
| EVA-X-S | `eva_x_small_patch16_merged520k_mim.pt` | https://huggingface.co/MapleF/eva_x (model code: https://github.com/hustvl/EVA-X) | MIT (model card) |
| EfficientNetV2S, ResNet50 | Keras ImageNet weights | Automatic via `tf.keras.applications` | See the Keras documentation |

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

- `checkpoints/` is not tracked by git, and the files are released unmodified so that they match the recorded hashes.
- The PyTorch files load with `torch.load(path, map_location="cpu", weights_only=True)` into the architectures built by `build_model` in `scripts/train_xrv_backbone.py` and `scripts/train_eva_x_backbone.py`; the radiomics file loads with `joblib.load` under the pinned package versions (load this pickle only after checking its hash). Details: `docs/NOTES.md`.
- Use of the weights is subject to the upstream licenses (TorchXRayVision: Apache-2.0; EVA-X-S weights: MIT) and the RSNA terms of use and attribution; see `THIRD_PARTY_NOTICES.md`.

## Running the full pipeline

`run_aura_cxr.sh` runs the study from raw images in its original order: split preparation, development-only radiomics and deep OOF predictions, pair selection and stacking (selection lock), full-development refit (deployment lock), a single locked-test inference and fusion, frozen Kermany evaluation, and branch-scoped XAI. Every stage before the pair lock reads only `train.csv` and `val.csv`, and the locked test can be opened only once per run directory. It needs a CUDA GPU (the study used one NVIDIA A100 40 GB), the retraining packages (TensorFlow, PyTorch, TorchXRayVision and timm; versions at the end of `requirements.txt` and in `configs/specs/environment_lock.json`), and the official EVA-X repository and checkpoint. All options of each script are listed by `--help`. GPU training was not run in bit-exact deterministic mode (`deterministic_algorithms: false` in `configs/seeds.json`), so a retrained model can differ slightly from the released one; the released checkpoints, per-case predictions and run records are the reference for every reported number.

```bash
RSNA_LABELS=... RSNA_DICOM=... KERMANY_DIR=... EVAX_REPO=... EVAX_CKPT=... bash run_aura_cxr.sh
```

### Predefined A1–A10 ablation (not post hoc)

The ten configurations A1–A10 (Table 5, Section 5.2) were defined before the locked test was opened (Section 3.4), and their locked-test outputs were produced by the `--stage ablation` step of the locked run in `run_aura_cxr.sh`. `scripts/analysis/ablation_a1_a10.py --check` (CPU) is only a recomputation check of this predefined ablation: it rebuilds every configuration from its members, re-derives every threshold from the development OOF predictions, and recomputes Table 5 from `results/predictions/locked_rsna_a1_a10.csv` (reference: `results/metrics/ablation_selected_dl_pair.csv`). The paired A8 − A4 test and the other comparisons below are post hoc.

### Post-hoc analyses

These analyses were run after the locked evaluation and never change A8. The CPU-only checks recompute the reported statistics from released files; the others regenerate the inputs from images.

| Analysis (paper section) | Script | Settings | Released outputs |
|---|---|---|---|
| A8 versus A4, A8 versus EVA-X-S, A8 versus ConvNeXt V2 Tiny (5.2, 5.5, 5.6) | `scripts/analysis/paired_comparisons.py --check` (CPU) | — | `results/analysis/a4_vs_a8/`, `results/analysis/contemporary_baseline/` |
| Holm-corrected pairwise DeLong family (5.2) | `scripts/analysis/locked_pairwise_delong.py --check` (CPU) | — | `results/analysis/locked_pairwise_delong/` |
| Adaptive versus fixed wavelets, Table 6 (5.3) | `scripts/analysis/wavelet_ablation_check.py --check` (CPU); regeneration: `wavelet_ablation_run.py`, then `wavelet_ablation_statistics.py` | `configs/analysis/wavelet_ablation.json` | `results/analysis/wavelet_ablation/` |
| Same-split ConvNeXt V2 Tiny comparator (5.5) | `scripts/analysis/convnextv2_baseline.py` | `configs/analysis/contemporary_baseline_training.json` | `results/analysis/contemporary_baseline/` |
| EVA-X-S LayerCAM versus Integrated Gradients (5.4) | `scripts/analysis/transformer_xai_analysis.py` | `configs/analysis/transformer_xai.json`, `transformer_xai_comparator.json` | `results/analysis/transformer_xai/` |
| XAI multiplicity families (5.4) | `scripts/analysis/xai_multiplicity.py` | `configs/analysis/xai_statistics.json` | `results/analysis/xai_statistics/` |
| Child-level bootstrap of the frozen Kermany evaluation (4.1, 5.6) | `scripts/analysis/kermany_child_bootstrap.py --check` (CPU) | — | `results/analysis/kermany_child_bootstrap/` |
| NIH patient overlap between development and locked test (6.2) | `scripts/analysis/nih_patient_overlap.py --mapping <RSNA mapping JSON>`; `--check` (CPU, no mapping) | RSNA mapping file (see below) | `results/analysis/nih_patient_overlap/` |

### NIH patient overlap

RSNA `patientId` values index images (Kaggle: "Each patientId corresponds to a unique image"); the 30,000 challenge images came from 12,274 NIH patients (Shih et al., 2019). Mapping each identifier to its NIH ChestX-ray14 image with the official RSNA file (`pneumonia-challenge-dataset-mappings_2018.json`, on the RSNA challenge page) shows that 3,738 of the 5,338 locked-test patientIds (70.0%) belong to NIH patients who also have development patientIds (2,227 shared patients). The internal estimates are therefore `patientId`-disjoint (image-level) rather than patient-disjoint; the Kermany cohort is the patient-independent evaluation.

The released predictions of all eleven evaluated configurations (A1–A10 and ConvNeXt V2 Tiny) were re-scored at their frozen thresholds on the 1,600 locked-test patientIds of the 1,514 NIH patients without development patientIds (125 positive, prevalence 7.8% versus 22.5% overall):

| | All locked test (n = 5,338) | No NIH patient in development (n = 1,600) |
|---|---|---|
| A8 ROC-AUC (95% CI) | 0.8988 (Table 5) | 0.9139 (0.8871 to 0.9387; patient-clustered bootstrap 0.8844 to 0.9382) |
| A8 sensitivity / specificity at τ = 0.470 | 0.8512 / 0.7715 | 0.7520 / 0.9105 |
| Two highest ROC-AUC | A4 0.8996, A8 0.8988 | A4 0.9156, A8 0.9139 |
| A8 − DenseNet121-XRV (Holm-adjusted DeLong p) | +0.0075 (< 0.001) | +0.0091 (0.004) |
| A8 − EVA-X-S (Holm-adjusted DeLong p) | +0.0059 (0.004) | +0.0022 (1.000; raw p = 0.649) |
| A8 − soft voting A7 (Holm-adjusted DeLong p) | +0.0116 (< 0.001) | +0.0095 (0.269) |
| A8 − ConvNeXt V2 Tiny (DeLong p) | +0.0066 (0.018) | +0.0260 (0.007) |
| A8 − A4 (DeLong p) | −0.0008 (0.044) | −0.0017 (0.125) |

The subset has a different case mix, so its values are not directly comparable with the full test. A4 and A8 remained the two highest configurations and LightGBM the lowest, while the order of the configurations in between changed (for example, EVA-X-S moved from fifth to third and ConvNeXt V2 Tiny from sixth to eighth). A8 remained above DenseNet121-XRV and ConvNeXt V2 Tiny, but its RSNA advantage over EVA-X-S and over soft voting was not resolved in this subset, as on Kermany for EVA-X-S, and its sensitivity at the frozen threshold was lower. The same holds within the development cohort: 14,277 of its 21,346 images belong to NIH patients whose images fall in more than one OOF fold, and 1,803 of the 2,668 validation images belong to patients with training images. Recomputed from the released OOF predictions on the 7,069 development images of patients confined to one fold, the ranking of the four deep candidates, of the two radiomics learners by ROC-AUC, and of all six pair + radiomics stacks was unchanged, with DenseNet121-XRV + EVA-X-S first (fold-mean ROC-AUC 0.9004 versus 0.8955 for the runner-up; 0.8916 versus 0.8883 on all development images), so the development-only selection does not depend on the patient overlap. Patient-specific cues cannot be excluded: among test images of shared patients whose development images all carry one label, the 1,510 images whose label agrees with those images were discriminated very well (A8 ROC-AUC 0.9548), whereas the 279 images whose label differs were poorly discriminated by every configuration (ROC-AUC 0.53 to 0.62; A8 0.5735). The same pattern occurred for the ImageNet-pretrained ConvNeXt V2 Tiny comparator, which was trained only on the development cohort (0.9520 versus 0.6161), so it may reflect patient-specific cues learned from the development images, or genuinely harder cases, rather than upstream pretraining. The shared-patient test images as a whole were not discriminated better than the unshared ones (A8 ROC-AUC 0.8765 on the 3,738 shared-patient images versus 0.9139 on the 1,600 others), and resampling NIH patients instead of images left the full-test A8 interval essentially unchanged (patient-clustered bootstrap 0.8886 to 0.9085). All subsets, models and tests are in `results/analysis/nih_patient_overlap/` (`nih_patient_overlap_metrics.csv`, `nih_patient_overlap_delong.csv`, `nih_patient_overlap_development_selection.csv`).

## Reproducibility checks

All commands run on a CPU without images:

```bash
python aura_cxr.py verify
python aura_cxr.py reproduce
python scripts/analysis/recompute_selection.py --check
python scripts/analysis/ablation_a1_a10.py --check
python scripts/analysis/paired_comparisons.py --check
python scripts/analysis/locked_pairwise_delong.py --check
python scripts/analysis/wavelet_ablation_check.py --check
python scripts/analysis/nih_patient_overlap.py --check
python scripts/analysis/kermany_child_bootstrap.py --check
```

- **`verify`** checks file integrity against `SHA256SUMS.txt`, the frozen configuration, split and prediction counts, analysis outputs and provenance; recomputes the selection and deployment lock identities from the original run records and checks that every out-of-fold and selection record predates the selection lock, that all refits fall between that lock and the opening of the locked test, and that the XAI settings were selected on validation data (`results/audit/run_records/README.md`); checks that no credentials, raw data or weights are in the git tree; and, when they have been downloaded to `checkpoints/`, the SHA-256 of the trained checkpoints.
- **`reproduce`** recomputes the locked-test A4/A8 and Kermany A8 metrics from `results/predictions/` and compares them with `results/metrics/reported_metrics.json`.
- **`recompute_selection.py --check`** recomputes, from `results/predictions/development_oof.csv` only, the candidate OOF metrics and the radiomics learner choice (Table 3), the six cross-fitted pair stacks and the selected pair (Table 4), the full-development A8 coefficients, and the development-derived thresholds of A8 (0.470), A4, A7 and the single components, and compares them with the released values.
- **`ablation_a1_a10.py --check`** rebuilds every A1–A10 configuration from its member probabilities (A9/A10 from stacks fitted on the development OOF predictions), re-derives every threshold from the development OOF predictions, and recomputes Table 5 from `results/predictions/locked_rsna_a1_a10.csv`.
- **`paired_comparisons.py --check`** recomputes the paired comparisons (A8 versus A4 on the locked RSNA test and Kermany, A8 versus EVA-X-S on Kermany, A8 versus both ConvNeXt V2 Tiny runs, and each encoder versus ConvNeXt V2 Tiny; paired DeLong for ROC-AUC and 2,000 paired class-stratified bootstrap resamples for the other metrics) from the per-case predictions, writes them to `runs/paired_comparisons/` and compares them with the released results.
- **`locked_pairwise_delong.py --check`** recomputes the Holm-corrected family of ten pairwise DeLong comparisons among DenseNet121-XRV, EVA-X-S, LightGBM, soft voting and A8 on the locked test (Section 5.2) and compares it with `results/analysis/locked_pairwise_delong/pairwise_delong_holm.csv`.
- **`wavelet_ablation_check.py --check`** recomputes Table 6 (pooled OOF metrics of the adaptive and four fixed wavelet policies), the four adaptive-minus-fixed paired DeLong tests with Holm adjustment and the adaptive selection frequencies from `results/analysis/wavelet_ablation/oof/`, and compares them with the released results.
- **`kermany_child_bootstrap.py --check`** resamples children instead of images in the frozen Kermany evaluation (the 624 test images come from 427 children, identified by their file names; class-stratified, 2,000 resamples, seed 42) and recomputes the child-level 95% CIs of the ROC-AUC of A1–A4, A7 and A8 and of the paired differences A8 − EVA-X-S and A8 − A4 from `results/predictions/kermany.csv`. The A8 interval (0.9081 to 0.9514) is essentially the image-level interval of the paper (0.9103 to 0.9498), and the conclusions of both paired comparisons are unchanged.
- **`nih_patient_overlap.py --check`** re-scores the NIH patient-overlap subsets for all eleven configurations, with class-stratified and patient-clustered bootstrap CIs and the paired DeLong tests, recomputes the development-only selection on patients confined to one OOF fold, from the released per-case overlap file and predictions, and compares them with the released results (about 1.5 minutes; regenerating the overlap itself needs the RSNA mapping file).

Notes on the audit records (MC-dropout inference of the deep branches, the delta-method A8 predictive width, the validation-only selection of the XAI settings, the lung masks of the descriptive outside-lung measure, legacy method names, EVA-X-S structure fields, upstream and configuration hashes, the comparator's initial run, and checkpoint loading) are in `docs/NOTES.md`.

## Repository structure

```
AURA-CXR/
├── aura_cxr.py              # verify / reproduce
├── run_aura_cxr.sh          # full pipeline from raw images (GPU)
├── THIRD_PARTY_NOTICES.md   # upstream licenses, changes made, data attribution
├── LICENSES/                # Apache-2.0 text (TorchXRayVision)
├── docs/NOTES.md            # notes on the audit records and checkpoint loading
├── checkpoints/             # downloaded release weights (not tracked)
├── configs/                 # frozen configuration, checkpoint registry, seeds, XAI settings
│   ├── specs/               # deep-training, radiomics, MRFO and deployment specifications
│   └── analysis/            # configurations of the post-hoc analyses
├── data/splits/             # patientId-level (image-level) train/val/test splits and 5 OOF folds
├── src/aura_cxr/            # fusion, evaluation, radiomics, XAI and data utilities
├── scripts/                 # training, model selection, XAI and external-validation code
│   └── analysis/            # post-hoc analyses
├── results/
│   ├── metrics/             # reported metrics and the result registry with CIs and p-values
│   ├── predictions/         # per-case locked-test, external and XAI outputs
│   ├── analysis/            # outputs of the post-hoc analyses
│   └── audit/               # golden reference, provenance, QC records, reproduction output
└── tests/
```

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

Code: MIT License (see `LICENSE`). External datasets and upstream pretrained weights remain under their own licenses and are not redistributed here. The trained checkpoints in release v1.0 are derived from the upstream weights and the RSNA data, and their use is subject to those licenses and terms; see `THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`.
