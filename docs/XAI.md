# Branch-Scoped XAI Specification

## Frozen models and target
The XRV and EVA-X-S deployment checkpoints are SHA256 `ba0559aeba8afe500eeae3fc36825d3742e543d005a52d3cc35eeb732bab317a` and `83a46553743f12a8bbb64f900ece2efa312a6b75d5638e8fae519ec0d31418a6`. Both attribution branches differentiate the **pre-softmax pneumonia logit (class index 1)**.

## XRV hooks
1. `backbone.features.denseblock4.denselayer16.conv1` — `Conv2d`, activation/gradient `[1,128,7,7]`, weight 0.75.
2. `backbone.features.denseblock4.denselayer15.conv1` — `Conv2d`, activation/gradient `[1,128,7,7]`, weight 0.25.

For activation $A_{k,u,v}$ and gradient $G_{k,u,v}$, the implementation computes

`L(u,v) = ReLU( Σ_k A(k,u,v) · ReLU(G(k,u,v)) )`.

The single-layer map is max-normalized; each layer map is order-1 resized to 224×224, min-max normalized, combined with weights 0.75/0.25, and min-max normalized again.

## EVA-X-S hooks and token structure
1. `blocks.11.mlp` — `GluMlp`, activation/gradient `[1,197,384]`, weight 0.75.
2. `blocks.11` — `EvaBlock`, activation/gradient `[1,197,384]`, weight 0.25.

The exact model has one class token followed by 196 row-major patch tokens (16×16 patches on a 14×14 grid), embedding dimension 384, and no other prefix tokens. The historical one-token removal heuristic was dynamically confirmed equivalent for this checkpoint.

For retained patch token $i$ and embedding dimension $k$, the token-space LayerCAM analogue is

`s_i = Σ_k A(i,k) · ReLU(G(i,k))`,

followed by row-major reshape to 14×14 and `ReLU` of the spatial score grid. This is a **LayerCAM-style token attribution analogue**, not a convolutional feature map and not an attention-rollout method.

## Hybrid map fusion
Each model's two layer maps are fused 0.75/0.25. The normalized XRV and EVA maps are then combined as `0.25·H_XRV + 0.75·H_EVA`, Gaussian smoothed with sigma 1.12 pixels (`max(0.5, 0.005×224)`), and min-max normalized. The selected map policy is `raw`; IoU uses an absolute threshold of **0.25 on the normalized map**.

## Validity and instrumentation
Maps are invalid if the hooked tensor has wrong dimensionality, token count cannot form the verified patch grid, activations/gradients are unavailable, values are non-finite, or normalization collapses to a constant/zero map. On the full released cohort, LayerCAM and comparator validity were both 1203/1203. Forward hooks produced zero change in XRV and EVA probabilities at tolerance 1e-7.

## Scope
`XAI_SCOPE = HYBRID_CNN_VIT_BRANCH_ONLY`. Spatial maps do not explain the LightGBM radiomics branch, the Logistic Regression meta-learner, or the final A8 decision.


# Transformer token-space LayerCAM rationale

CNN LayerCAM uses a local positive-gradient weight at each spatial site before aggregation over channels. In the frozen EVA-X-S representation, the selected hook tensors have shape `[1,197,384]`: after removing the **verified** leading class token, each of the 196 remaining row-major patch tokens corresponds to one site of the 14×14 patch grid, while the 384 embedding dimensions provide the local feature axis.

The implemented EVA score is therefore `s_i = Σ_k A_ik ReLU(∂y/∂A_ik)`, followed by spatial reshape and ReLU. This preserves the local gradient×activation logic of LayerCAM under the correspondence `CNN spatial site ↔ patch token` and `CNN channel ↔ embedding dimension`. It is appropriately described as a **token-space LayerCAM analogue** or **LayerCAM-style token attribution**, not as original convolutional LayerCAM and not as an attention map.

The released analysis dynamically verified the exact token structure instead of inferring it: one class token, zero other prefix tokens, 196 patch tokens, embedding dimension 384, patch size 16×16, 14×14 grid, and row-major ordering. The historical `N-1` square heuristic happened to be equivalent for this exact frozen checkpoint, but the paper should state the verified structure explicitly.

The established post-hoc comparator was Integrated Gradients (Sundararajan, Taly, and Yan, 2017), selected and parameter-locked before full test execution. The comparison does **not** prove either method correct. It asks whether localization conclusions are stable to a materially different attribution mechanism. They were not: the paired sensitivity results show substantial method dependence, which is now treated as an explicit limitation.
