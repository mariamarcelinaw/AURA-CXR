#!/usr/bin/env bash
# AURA-CXR full pipeline, from raw images to the locked-test and external results.
#
# Requires the raw RSNA and Kermany images, a CUDA GPU, the retraining packages listed
# at the end of requirements.txt, the official EVA-X repository and the EVA-X-S
# checkpoint. The order below is the order of the study; each stage checks that its
# prerequisites exist and refuses to run on data it must not see (for example, every
# stage before the pair lock reads only train.csv and val.csv).
#
# Usage:   bash run_aura_cxr.sh            (paths below can be overridden by environment variables)
# Checking the reported numbers without images or a GPU: python aura_cxr.py verify && python aura_cxr.py reproduce
set -euo pipefail

RSNA_LABELS=${RSNA_LABELS:-data/raw/rsna/stage_2_train_labels.csv}
RSNA_DICOM=${RSNA_DICOM:-data/raw/rsna/stage_2_train_images}
KERMANY_DIR=${KERMANY_DIR:-data/raw/kermany/chest_xray}   # contains test/NORMAL and test/PNEUMONIA
EVAX_REPO=${EVAX_REPO:-checkpoints/EVA-X}                  # git clone https://github.com/hustvl/EVA-X
EVAX_CKPT=${EVAX_CKPT:-checkpoints/eva_x_small_patch16_merged520k_mim.pt}
RUN=${RUN:-runs/aura_cxr}

PY=${PY:-python}
EVA_ARGS=(--eva_x_repo "$EVAX_REPO" --pretrained_checkpoint "$EVAX_CKPT")

# 1. Released patient-level split and five-fold assignment (no new randomization).
$PY scripts/prepare_splits.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --results_dir "$RUN"

# 2. Development-only radiomics: features, learner screening/MRFO refinement, nested OOF.
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage development_features --gpu40
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage radiomics_nested_oof --gpu40

# 3. Development OOF probabilities of the four deep candidates.
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage development_oof --gpu40   # EfficientNetV2S, ResNet50
$PY scripts/train_xrv_backbone.py   --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode oof --gpu
$PY scripts/train_eva_x_backbone.py --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode oof --gpu "${EVA_ARGS[@]}"

# 4. Pair selection, stacking coefficients and threshold from development OOF only (selection lock).
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage pair_select

# 5. Full-development refit of the selected components (deployment lock).
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage selected_refit --gpu40
$PY scripts/train_xrv_backbone.py   --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode refit --gpu
$PY scripts/train_eva_x_backbone.py --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode refit --gpu "${EVA_ARGS[@]}"

# 6. Locked RSNA test: label-free inference, then the frozen A8 fusion (opens the test once).
#    --exploratory_test / --exploratory_all4_test add the non-selected models for the exploratory A9/A10 rows.
$PY scripts/train_xrv_backbone.py   --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode test_infer --gpu --exploratory_test
$PY scripts/train_eva_x_backbone.py --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --mode test_infer --gpu --exploratory_test "${EVA_ARGS[@]}"
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage selected_test_infer --exploratory_all4_test --gpu40
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage fusion
$PY scripts/train_aura.py --labels_csv "$RSNA_LABELS" --dicom_dir "$RSNA_DICOM" --output_dir "$RUN" --stage ablation
$PY scripts/q1_stats_report.py --results_dir "$RUN"

# 7. Frozen external evaluation on the Kermany pediatric test partition (no refit, no threshold change).
$PY scripts/external_validation_kermany_stage19.py --results_dir "$RUN" --image_dir "$KERMANY_DIR" --gpu \
    --train_script scripts/train_aura.py --xrv_script scripts/train_xrv_backbone.py \
    --eva_x_script scripts/train_eva_x_backbone.py --eva_x_repo "$EVAX_REPO" --eva_x_checkpoint "$EVAX_CKPT"

# 8. Branch-scoped quantitative XAI on the 1,203 GT-positive locked-test cases.
$PY scripts/eval_only_xai.py --results_dir "$RUN" --dicom_dir "$RSNA_DICOM" --labels_csv "$RSNA_LABELS" --run_mode final --gpu40 \
    --xrv_script scripts/train_xrv_backbone.py --eva_x_script scripts/train_eva_x_backbone.py \
    --eva_x_repo "$EVAX_REPO" --eva_x_checkpoint "$EVAX_CKPT"

echo "Pipeline complete. Post-hoc analyses: see the 'Post-hoc analyses' table in README.md."
