# Public notes

**Scope.** Separate optional public development panel (10,484 requests). Outside the preview composite.
Evaluation only; no training use.

**Counts.** 10,484 roster IDs; 43 benchmarks; 0 duplicate `run_id`; 0 duplicate (`run_id`,`group_id`).
Mapped to upstream edition 0.3 rows: 10,484 (8,005 in `selected-rows`, 2,281 in `added-rows`, 198 in `gsm8k-rows`);
unmapped: 0. Some distinct run IDs share one `payload_sha256` (20 hashes / 43 rows: BPoMP, CLadder, SGD) - `run_id` is the key.

**Hashes.** `payload_sha256` is the value published in each upstream row's `_evaluation` block; how upstream
derives it was NOT independently reproduced. `upstream_row_sha256` is SHA-256 of the upstream row's JSONL line
(without newline); upstream states rows are byte-identical lines across editions. Upstream file pins used (SHA-256
of the *uncompressed* files, from upstream `editions.py`, edition 0.3): selected-rows b2b56d6f...0792d5,
added-rows 7429f3c9...52f4ce89, gsm8k-rows 75933489...f4738a (full values in `verify_tune.py`).
Gzip bytes can differ between rebuilds; the verifier hashes uncompressed content.

**What was tested.** The verifier passed (10,484 mapped, 0 mismatches, pins matched) against a pre-existing
local copy of the three edition-0.3 files. It was NOT tested against a fresh download from the dataset host or a
fresh `suite rebuild`; no verified-rebuild claim is made. If your rebuild yields different bytes the verifier reports
which rows differ and exits non-zero; exact bytes cannot be supplied from this folder.

**Source terms.** The attribution table gives source links and licence evidence recorded by the pinned
[Decision Index methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/4ee57a1610e0547634ba0ec11990d10122672191/data/methodology.json). For sources without a recorded licence, consult the original terms.
Only upstream identifiers and hashes are shipped. Use the underlying rows for evaluation under their respective terms.
The reproduction code's MIT licence applies to its code.
