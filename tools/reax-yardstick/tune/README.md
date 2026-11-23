# REAX yardstick - optional public TUNE development panel

**Use this separate public development panel to check your local evaluation pipeline.** The yardstick calculates the preview composite from its four preview families.

The roster contains upstream IDs and hashes. Rebuild the rows with the [Decision Index reproduction kit](https://github.com/apolinario/decision-index) for edition 0.3. Follow its rebuild and import steps, then point this verifier at the populated `suite-0.3` directory.

Revert upstream commit `2d7791c` locally before the rebuild (upstream issue
[#274](https://github.com/apolinario/decision-index/issues/274)); otherwise the SGD rows do not match the frozen
edition and the verifier reports them as mismatches.

* 10,484 requests, 43 benchmarks (`benchmarks.csv`), keyed by upstream `run_id`.
* Roster: `tune-roster.csv` - `catalog_id, dataset, track, run_id, group_id, payload_sha256, upstream_file, upstream_row_sha256`.
* Verify (and optionally extract) against your copy: `python3 -I verify_tune.py --suite <dir with selected-rows/added-rows/gsm8k-rows>`
  (`--extract out.jsonl` writes the matching upstream rows verbatim; `--roster-only` checks the shipped files offline).
* Tests: `python3 -I -B -m unittest discover -s tests`. Integrity: `MANIFEST.sha256`.

## Rules
* **Evaluation only. Do not train, fine-tune, distil or tune on these rows or anything derived from them.**
* The rows are a mixed-licence collection; each benchmark keeps its own licence and attribution
  (`licences-attribution.csv`). Obtain and honour the original terms.
* Rows keep their original case grouping: score and resample by `group_id`.
* Read `PUBLIC-NOTES.md` for exactly what has and has not been verified.
