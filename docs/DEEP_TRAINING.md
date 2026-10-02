# Deep-Model Training Specification

This document summarizes the executable training/refit lineage for the four deep-learning candidates and the selected DenseNet121-XRV + EVA-X-S pair.

## Model identities

- **EfficientNetV2S:** Keras `EfficientNetV2S(..., weights="imagenet", include_top=False, pooling="avg")`.
- **ResNet50:** Keras `ResNet50(..., weights="imagenet", include_top=False, pooling="avg")`.
- **DenseNet121-XRV:** TorchXRayVision `densenet121-res224-all` with the AURA two-logit head; 6,968,084 parameters.
- **EVA-X-S:** official `eva_x_small_patch16` initialized from `eva_x_small_patch16_merged520k_mim.pt`, followed by head replacement; 21,667,970 parameters.

## Branch-specific preprocessing

All branches use 224×224 inputs but not one shared normalization. Exact machine-readable settings are in `reproducibility/deep_01_branch_input_normalization.csv` and `reproducibility/deep_training_specification.json`.

Common RSNA DICOM handling includes RescaleSlope/Intercept, MONOCHROME1 inversion when required, per-image min-max normalization, and resize to 224×224. Branch-specific normalization is then applied for each backbone.

## Training and optimization

- TF candidates: Adam; frozen-backbone phase followed by controlled partial unfreezing; validation-AUC checkpointing/early stopping.
- DenseNet121-XRV: full fine-tuning with AdamW, class-balanced focal cross-entropy, gradient clipping, and cosine annealing.
- EVA-X-S: full fine-tuning with AdamW, class-balanced focal cross-entropy, gradient clipping, and cosine annealing.
- Outer OOF holdouts are never used for checkpoint selection or learning-rate scheduling.
- Every outer fold constructs an inner 90/10 stratified split using seed `42 + fold`.

## Final development refit

The selected XRV fold best epochs were `[1, 2, 2, 1, 2]`; EVA best epochs were `[2, 1, 1, 2, 3]`. Each has median 2, so the final development refit uses two epochs after fresh initialization from the corresponding upstream weights. The refit uses all 21,346 development cases, with no locked-test or Kermany input.

## Frozen deployment checkpoint identities

- XRV `xrv_development_refit.pt`: SHA256 `ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a`.
- EVA-X-S `eva_x_development_refit.pt`: SHA256 `83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6`.

The exact checkpoint bytes are not distributed here. The upstream EVA checkpoint file hash was not historically captured, so it is not fabricated.

## Seeds and determinism

Global seed is 42, with fold-dependent inner seeds 42–46. The original code seeds relevant Python/NumPy/framework/CUDA generators. Full deterministic-kernel enforcement was not globally locked; training is therefore procedurally reproducible rather than guaranteed bitwise-identical across arbitrary GPU/software stacks.

## Runtime provenance

Verified historical selected-branch runtime includes Python 3.13.12, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, TorchXRayVision 1.5.2, timm 1.0.28, and NVIDIA A100-SXM4-40GB. See `reproducibility/environment_lock.json` and `reproducibility/deep_training_specification.json` for machine-readable details.

## Verification

```bash
python aura_cxr.py verify-deep-config
```

Full training requires the original datasets and upstream/pretrained model assets.
