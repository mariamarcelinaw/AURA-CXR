# Adaptive Radiomics Reproducibility Specification

## RSNA preprocessing

DICOM → float32 → RescaleSlope/Intercept → MONOCHROME1 inversion when needed → per-image min-max normalization → uint8 → Pillow bilinear resize to 224×224 → float32/255.

## Wavelet selection

Candidate wavelets are ordered as:

```text
haar, db4, sym4, coif3
```

A one-level 2D DWT is evaluated per image. The adaptive policy selects the wavelet according to the locked entropy-based rule defined in `reproducibility/radiomics_specification.json` and the source snapshot.

The controlled public ablation under `analysis/wavelet_ablation/` compares this adaptive policy with globally fixed Haar, db4, sym4, and coif3 policies using the same development patients/folds and downstream radiomics learner configuration.

## Feature representation

The radiomics branch constructs a 746-feature texture representation from the retained wavelet-derived subbands:

- GLCM features: **720**
- LBP features: **26**
- Total: **746**

Exact quantization, distances, angles, GLCM properties, LBP parameters, concatenation order, and feature names are stored in:

- `reproducibility/radiomics_specification.json`
- `reproducibility/feature_schema_746.csv`

## Learner and MRFO

The frozen selected radiomics learner is LightGBM. The final MRFO configuration uses population 15, 15 iterations, inner StratifiedKFold=3, and seed 42. The objective is negative mean macro-F1; minimizing it maximizes macro-F1.

The complete search space is in `reproducibility/mrfo_search_space.csv`.

## Frozen artifacts

- Development feature cache SHA256: `18442202a80092bf8ac57547b425b5985dba4843d22a3baefe7930ee3a9d744b`
- Canonical LightGBM OOF SHA256: `f180acec675977caa304d2b88e66c3392bdfa6d8b359b7843a488c2a75a0b1bb`
- Frozen radiomics checkpoint SHA256: `15d862c5c9ddfd088aebdbfd146b4cb133c089f577cd09ed73bf71058ab1c1b7`

## Software provenance

Frozen runtime evidence includes Python 3.13.12, NumPy 2.4.2, pandas 3.0.0, SciPy 1.17.0, scikit-learn 1.8.0, Pillow 12.1.1, pydicom 3.0.2, PyWavelets 1.8.0, scikit-image 0.26.0, joblib 1.5.3, and LightGBM 4.7.0.

The historical development source did not pin PyWavelets. PyWavelets 1.8.0 is captured by the frozen external-validation preflight, while a later installation log recorded 1.9.0. The released adaptive OOF vector was independently recovered to float32-level equivalence, which numerically locks the public behavior without inventing an unavailable historical package pin.

## External preprocessing caveat

Kermany uses a dataset-specific Pillow grayscale loader whose resize defaults to bicubic, while RSNA radiomics uses explicit bilinear resize. The downstream extractor and frozen LightGBM model are unchanged. This dataset-specific preprocessing difference is retained in the provenance and should not be silently homogenized.
