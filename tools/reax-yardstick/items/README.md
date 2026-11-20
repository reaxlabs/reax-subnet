# items/

| Dir | Family | Files | Text shipped? |
|---|---|---|---|
| `di/` | PV-HF (300) | `index.jsonl`: run id, source run id, benchmark, board payload hash, content hash | no: rebuild the Decision Index public suite and pass `--di-suite` |
| `hle/` | PV-HLE (509) | `index.jsonl`: HLE row id, decline option letter and wording, content hash | no: accept the `cais/hle` terms and pass `--hle` |
| `when2call/` | PV-W2C (3,423) | `requests.jsonl.gz`, `gold.jsonl.gz`, `index.jsonl` | yes (CC BY 4.0) |
| `onet/` | PV-ONET (1,000) | `requests.jsonl.gz`, `gold.jsonl.gz`, `index.jsonl` | yes (Apache-2.0) |

`content_sha256` = sha256 of `json.dumps({"state": ..., "questions": ..., "answers": ...}, separators=(",", ":"), ensure_ascii=True)`
with the key order kept, because the option order is part of the prompt. `MANIFEST.json` lists the sha256 of every file and
`manifest_sha256` over the rest of the manifest (sorted-key canonical JSON). Licences and attribution: `../THIRD-PARTY.md`.
