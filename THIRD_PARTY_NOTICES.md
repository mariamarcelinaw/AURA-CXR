# Third-party notices and data attribution

The code in this repository is released under the MIT License (`LICENSE`). The items below come from, or were derived from, third-party models and datasets and remain subject to their terms.

## Trained checkpoints (GitHub release v1.0)

| File | Derived from | Upstream license | Changes made for AURA-CXR |
|---|---|---|---|
| `xrv_development_refit.pt` | TorchXRayVision `densenet121-res224-all` (Cohen et al., https://github.com/mlmed/torchxrayvision) | Apache License 2.0 (TorchXRayVision main library, which includes `xrv.models`); full text in `LICENSES/Apache-2.0.txt` | A dropout layer and a two-class linear head were added (the original 18-label classifier is kept in the state dictionary but not used), and all weights were fine-tuned for two epochs on the 21,346-patient RSNA development cohort |
| `eva_x_development_refit.pt` | EVA-X-S `eva_x_small_patch16_merged520k_mim.pt` (Yao et al., hustvl; https://huggingface.co/MapleF/eva_x) | MIT License, as declared on the model card (notice below); the EVA-X code repository (https://github.com/hustvl/EVA-X) carries an Apache-2.0 `LICENSE.txt`, and its code is not redistributed here | The classification head was replaced by dropout and a two-class linear layer, and all weights were fine-tuned for two epochs on the RSNA development cohort |
| `radiomics_mrfo_development_refit.pkl` | No third-party weights (LightGBM classifier trained by the authors) | — | — |

All three checkpoints were trained on RSNA Pneumonia Detection Challenge data, so their use is also subject to the RSNA terms and attribution requirements below.

MIT License notice for the EVA-X-S weights (the model card names no copyright holder; copyright belongs to the EVA-X authors):

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Citations:

- J. P. Cohen et al., "TorchXRayVision: A library of chest X-ray datasets and models", Proc. of the 5th International Conf. on Medical Imaging with Deep Learning, PMLR, Vol. 172, pp. 231-249, 2022.
- J. Yao et al., "EVA-X: a foundation model for general chest x-ray analysis with self-supervised learning", npj Digital Medicine, Vol. 8, Art. no. 678, 2025, doi: 10.1038/s41746-025-02032-z.

## Datasets

No images are redistributed. `data/splits/` contains patient identifiers and labels derived from the RSNA challenge annotations, and `results/` contains per-case model outputs.

**RSNA Pneumonia Detection Challenge (2018).** As required by the RSNA Terms of Use and Attribution:

- RSNA download site: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
- G. Shih et al., "Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia", Radiology: Artificial Intelligence, Vol. 1, No. 1, e180041, 2019, doi: 10.1148/ryai.2019180041.
- NIH Chest X-ray dataset, the source of the challenge images: https://nihcc.app.box.com/v/ChestXray-NIHCC. The NIH Clinical Center is the data provider.
- X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers, "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", Proc. of IEEE Conf. on Computer Vision and Pattern Recognition, pp. 3462-3471, 2017.

**Kermany pediatric chest X-ray data** (external evaluation only): D. S. Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning", Cell, Vol. 172, No. 5, pp. 1122-1131, 2018, doi: 10.1016/j.cell.2018.02.010; data: Mendeley Data, doi: 10.17632/rscbjbr9sj.2.

## Upstream weights used but not redistributed

ConvNeXt V2 Tiny (`timm/convnextv2_tiny.fcmae_ft_in1k`, same-split comparator) and the Keras ImageNet weights of EfficientNetV2S and ResNet50 (development candidates) are downloaded from their providers under their own licenses. The Python packages in `requirements.txt` are installed from PyPI under their own licenses.
