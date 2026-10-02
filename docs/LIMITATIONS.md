# Known Limitations

- Exact frozen checkpoint bytes are not bundled in the public repository.
- Raw RSNA and Kermany medical images are not redistributed.
- Full CUDA/cuDNN/driver behavior is not guaranteed bitwise identical across machines.
- Spatial XAI is scoped to the selected Hybrid CNN–ViT image branch and is not a complete explanation of A8.
- Transformer attribution results are method-dependent.
- The adaptive wavelet policy is not claimed to outperform all fixed-wavelet controls.
- The radiomics probability is not claimed to provide an incremental locked-test discrimination benefit over A4 under the frozen A8 integration.
- The contemporary baseline analysis covers one controlled comparator and does not establish universal or state-of-the-art superiority.
- Kermany supports transportability only to the evaluated pediatric cohort; adult, multicenter, prospective, subgroup-fairness, and clinical-deployment validity remain unestablished.
