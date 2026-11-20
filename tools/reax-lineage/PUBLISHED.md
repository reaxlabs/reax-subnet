# What is published here

This is the REAX lineage checker, version 1.6.2, as run by the round-2 intake (`check_script_sha256`
`302932efe9a4d4f77a358e9bdcc098fff87665f02557e021fcac0858694cf3f1`; the README and the lineage page say how to recompute it).
The hash covers `reax_lineage/*.py`, `reax_lineage/thresholds.json` and the two schema files; those are published unchanged.

Left out of this copy (not covered by the hash):
- `tests/golden/*.npz`: sampled weight fingerprints of three third-party models. The golden tests skip when those fixtures are absent.
- `calibration/null_readmes/`, `calibration/c1_repo_files/`: copies of third-party model READMEs and repo files used for calibration.
Files the README or comments mention but that are not part of this tree (calibration notes, change log) are internal.

Notes:
- The schema's `$id` (`https://reax.example/schema/reax-lineage.schema.json`) is a placeholder that does not resolve. The schema files are part of the
  script hash, so they are published byte-identical to what the intake runs; the canonical location is
  https://github.com/reaxlabs/reax-subnet/blob/main/tools/reax-lineage/schema/reax-lineage.schema.json . A later checker version will carry the real `$id`.
