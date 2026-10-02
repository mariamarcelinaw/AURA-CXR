# Model Checkpoints

Checkpoint metadata and known SHA-256 identities are listed in `configs/checkpoints.json` and the reproducibility specifications.

The public repository does **not** redistribute model-weight files (`.pt`, `.pth`, `.ckpt`, `.h5`, `.keras`, etc.) unless redistribution rights are confirmed. This separation applies both to third-party pretrained weights and AURA-CXR derived/frozen checkpoints.

For the selected deployment branches, the frozen checkpoint identities are:

- DenseNet121-XRV: `ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a`
- EVA-X-S: `83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6`

Users performing exact checkpoint-based evaluation must obtain or reconstruct the required assets lawfully and verify them against the documented identities before inference.
