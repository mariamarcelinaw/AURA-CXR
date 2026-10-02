# Provenance

The repository separates three provenance layers:

1. **Public scientific API** under `src/aura_cxr/` for stable verification and lightweight reproduction.
2. **Scientific source snapshot** under `reproducibility/source_snapshot/` preserving the implementation lineage used to generate the released experiment artifacts.
3. **Machine-readable evidence** under `configs/`, `splits/`, `predictions/`, `reproducibility/`, `analysis/`, and `publication_assets/`.

Private absolute paths, raw medical images, credentials, transient caches, and journal-workflow files are not part of the public release.

The public package records SHA-256 hashes of released files in `SHA256SUMS_PUBLIC.txt`. Scientific source and derived outputs should be compared using the published hashes/configuration rather than filenames alone.

The public API is intentionally non-invasive: audit reproduction does not refit models, retune thresholds, or access protected test data for optimization.
