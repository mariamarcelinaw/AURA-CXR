# Third-Party Software, Model, and Dataset Notice

This repository separates AURA-CXR code/reproducibility artifacts from upstream assets.

## Datasets

Raw RSNA and Kermany chest X-ray images are **not redistributed**. Users must obtain them from the official dataset sources and comply with the corresponding terms.

The released `splits/`, predictions, and analysis artifacts support reproducibility of the protocol; they do not replace the source datasets.

## Upstream pretrained models

The workflow relies on pretrained components including DenseNet121-XRV / TorchXRayVision, EVA-X-S, and ConvNeXt V2 Tiny. Their source code, pretrained weights, and model files remain governed by their own upstream licenses and terms.

This repository does not assume that third-party weight redistribution is permitted. Model identities/hashes are documented where available; users should obtain required assets from official sources.

## AURA-CXR source code

No explicit open-source license has yet been added to this repository. See `LICENSE_POLICY.md`. A paper publication license does not automatically grant a software license, and a future software license cannot override dataset/model licenses.
