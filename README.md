# AURA-CXR

**Public repository:** https://github.com/mariamarcelinaw/AURA-CXR

**Development-Locked Hybrid CNN–ViT and Adaptive Radiomics Stacking With Quantitative XAI for Pneumonia Detection**

AURA-CXR is a research implementation for binary pneumonia detection from chest X-rays. The framework combines a selected **DenseNet121-XRV + EVA-X-S Hybrid CNN–ViT image branch**, an image-specific **Adaptive Radiomics** branch using entropy-guided wavelet selection with GLCM/LBP features and LightGBM, and a frozen **Logistic Regression A8 stack**.

> **Research use only.** This repository is not a medical device and must not be used for clinical diagnosis or patient-care decisions.

## What is public here

This repository deliberately contains only the material needed for **scientific use and reproducibility**:

- implementation source code;
- frozen model/configuration metadata;
- random seeds and exact RSNA split derivatives;
- training, inference, XAI, ablation, comparator, and external-validation scripts;
- compact paper-facing numerical summaries;
- dependency/environment files;
- citation and technical documentation.

Internal working materials, manuscript-production files, local logs/caches, redundant intermediate outputs, and large per-case audit exports are **not** part of the public repository. See [`PUBLIC_CONTENT_POLICY.md`](PUBLIC_CONTENT_POLICY.md).

Raw RSNA/Kermany images and checkpoint binaries are also not redistributed. See [`THIRD_PARTY_AND_DATA_NOTICE.md`](THIRD_PARTY_AND_DATA_NOTICE.md).

## Framework overview

```text
Chest X-ray
   │
   ├── DenseNet121-XRV ─┐
   │                     ├── Hybrid CNN–ViT image branch
   ├── EVA-X-S ──────────┘
   │
   └── Adaptive Radiomics
       entropy-guided wavelet selection
       + GLCM/LBP texture features
       + LightGBM
              │
              ▼
[pXRV, pEVA-X, pRadiomics]
              │
              ▼
 Logistic Regression A8
              │
              ▼
    locked threshold = 0.470
```

The final A8 meta-feature order is `[p_xrv, p_eva_x, p_radiomics]`.

## Main reported results

- **Locked RSNA test:** 5,338 cases; frozen A8 ROC-AUC ≈ **0.8988**.
- **Kermany pediatric test:** 624 cases; frozen A8 ROC-AUC ≈ **0.9309**.
- **A4 vs A8:** the deep-only A4 configuration had a slightly higher locked-test ROC-AUC than A8; the radiomics probability is therefore not presented as an established incremental discrimination gain.
- **Wavelet controls:** fixed Haar, db4, and sym4 controls exceeded the adaptive policy in development OOF ROC-AUC; adaptive-wavelet superiority is not claimed.
- **Transformer attribution:** the attribution result was **method-dependent** when token-space LayerCAM was compared with Integrated Gradients.
- **XAI scope:** spatial heatmaps explain the **Hybrid CNN–ViT image branch only**, not the LightGBM radiomics branch or complete A8 stack.
- **Contemporary comparator:** A8 showed a small ROC-AUC advantage over the controlled same-split ConvNeXt V2 Tiny comparator; this is not a state-of-the-art claim.
- **External scope:** Kermany supports only limited transportability to the evaluated pediatric cohort; broader adult/multicenter generalization remains unestablished.

Compact machine-readable values are in [`results/reported_metrics.json`](results/reported_metrics.json).

## Repository layout

```text
AURA-CXR/
├── aura_cxr.py
├── src/aura_cxr/        # reusable framework modules
├── scripts/             # training, evaluation and analysis scripts
├── configs/             # frozen configuration and seed metadata
├── splits/              # exact RSNA patient-level split derivatives
├── results/             # compact paper-facing result summaries
├── tests/               # integrity/unit tests
├── docs/                # technical documentation
├── CITATION.cff
├── CITATION.bib
└── THIRD_PARTY_AND_DATA_NOTICE.md
```

## Main programs

| Program | Purpose |
|---|---|
| `scripts/train_aura.py` | Main AURA-CXR training/orchestration workflow. |
| `scripts/aura_dl_pair_selection.py` | OOF construction, deep-pair selection, A4/A8 fusion, threshold selection, and deployment lock. |
| `scripts/train_xrv_backbone.py` | DenseNet121-XRV training and final development refit. |
| `scripts/train_eva_x_backbone.py` | EVA-X-S training and final development refit. |
| `scripts/external_validation_kermany.py` | Frozen Kermany evaluation. |
| `scripts/eval_only_xai.py` | Branch-scoped localization and faithfulness analysis. |
| `scripts/q1_stats_report.py` | Statistical reporting utilities. |

See [`docs/PROGRAM_MAP.md`](docs/PROGRAM_MAP.md) for more detail.

## Installation

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Basic integrity checks

```bash
python aura_cxr.py verify
python scripts/verify_deep_config.py
python scripts/verify_radiomics_reproducibility.py
python scripts/verify_xai.py
pytest -q
```

Full training requires the original datasets, upstream pretrained models, and substantial GPU resources. Dataset images and third-party checkpoint binaries are not bundled.

## Reproducibility

The repository includes the frozen configuration, random seeds, exact patient-level RSNA split derivatives, model/checkpoint identifiers where available, training methods, and analysis scripts. Start with:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/DATASETS.md`](docs/DATASETS.md)
- [`docs/DEEP_TRAINING.md`](docs/DEEP_TRAINING.md)
- [`docs/RADIOMICS.md`](docs/RADIOMICS.md)
- [`docs/XAI.md`](docs/XAI.md)
- [`docs/MODEL_CHECKPOINTS.md`](docs/MODEL_CHECKPOINTS.md)

## Citation

If you use AURA-CXR, please cite the associated paper. This repository does **not** use a separate software DOI. Add only the official journal article DOI when it is issued by the publisher.

See [`CITATION.md`](CITATION.md), [`CITATION.cff`](CITATION.cff), and [`CITATION.bib`](CITATION.bib).

## Authors

- **Maria Marcelina Widyastuti**
- **Chastine Fatichah** — corresponding author
- **Dwi Sunaryono**

Department of Informatics Engineering, Institut Teknologi Sepuluh Nopember, Surabaya, Indonesia.

Corresponding author: **Chastine Fatichah** — `chastine@its.ac.id`

## License and third-party material

Public visibility is not itself a software license. Until an explicit software license is approved by the authors/institution, reuse rights should not be inferred. See [`LICENSE_POLICY.md`](LICENSE_POLICY.md).

Datasets and upstream pretrained models remain subject to their own terms. See [`THIRD_PARTY_AND_DATA_NOTICE.md`](THIRD_PARTY_AND_DATA_NOTICE.md).
