# Local datasets (not bundled)

Keep original downloads unchanged under `raw/` or any configured external location. Derived metadata goes under `processed/`; verified supplemental annotation tables can live in `annotations/`. Everything under this directory except this README is ignored by Git.

See [ACRIMA setup and dataset contracts](../docs/datasets.md) before organizing your download. Only ACRIMA processing is implemented. Other dataset paths are reserved; no download is automatic and no missing labels or patient IDs are inferred.
