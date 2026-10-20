# Engine determinism GPU release validation (not yet run)

The CPU tests do not qualify a GPU class. Keep the engine track closed until the
release owner has performed and reviewed these steps on an approved, isolated GPU
host, with no sealed data, wallets, production credentials or unrelated workloads.
Use exact names for containers created by this work. Do not run the supplied miner
bundle until its source has passed the separate third-party code review.

## Reproduce the original failure

Use the pre-fix subnet checkout **b0ec95c**, because the patched DockerRunner correctly
refuses unpublished GPU configs. Egor's bundle is at
`~/jobs/sn92-fast-launch-20261015/untrusted/egor-repro-20261017/reax-reference-repro.tar.gz`.
After reviewing/extracting it into an isolated directory, from that pre-fix venv:

```bash
export OPENAI_API_KEY=
export BUNDLE=/path/to/reviewed/owner-bundle
export REF=ghcr.io/reaxlabs/reax-s1-fast-runtime@sha256:d857a90eaf168d1db3ca982c5111b10fb61a65cc5f7113ea9e391fe6fca17b71
PYTHONPATH="$BUNDLE" python "$BUNDLE/spike/e4d_refvsref.py" --gpu 0 \
  --weights /models/plumb-4b --workdir /tmp/engdet-e4d-1 --n 2000 --out E4d-1.json
```

Run four separately named trials, retain every output and idle/loaded GPU census.
The bundle README describes its 20 idle / 20 loaded fresh-process census and
aggregation commands. Its reported failing trial was 1,354/2,000 differing answers,
top-1 agreement 98.4%. These are the miner's results, not independently reproduced
measurements from this patch.

## Generate configs from measured winners

The following command is a **not yet GPU-tested** recipe based on the actual
`fla-core==0.5.2` wheel's `fla/ops/utils/cache.py` schema. It exports observed Triton
winners, never hand-authored tuning values. Run once on an idle GPU of each exact
class, with a fresh process/cache and the pinned original runtime. Mount the
reviewed subnet source at `$REPO` and create an empty, writable `$OUT` directory.
The public seeded workload is for calibration only, never sealed competition items.

```bash
docker run --rm -i --name engdet-config-idle-1 --gpus device=0 --network none \
  --read-only --tmpfs /tmp:rw,exec,size=16g --shm-size 2g --user 10002:10002 \
  -v /models/plumb-4b:/model:ro -v "$REPO":/repo:ro -v "$OUT":/out \
  -e OPENAI_API_KEY= -e HF_HUB_OFFLINE=1 -e HOME=/tmp -e TRITON_CACHE_DIR=/tmp/triton \
  -e FLA_CACHE_MODE=disabled -e PYTHONPATH=/repo/src --entrypoint python "$REF" - <<'PY'
import json, sys
from pathlib import Path
import torch, triton
from jevk5.runtime import JevK5, decision_options
from reax_subnet.s1items import generate_s1_items
from fla.ops.utils.cache import CachedAutotuner, AutotuneKey
model = JevK5('/model', graphs=True)
for item in generate_s1_items(2000, seed=0x52454158):
    author = item.question()
    options = decision_options(author)
    ids = model.encode(item.state, author['instructions'], [text for _, text in options])
    model.letter_logits(ids, len(options))
seen, kernels = set(), {}
for name, module in list(sys.modules.items()):
    if module is None or not (name == 'fla' or name.startswith('fla.')):
        continue
    for obj in list(vars(module).values()):
        chain = set()
        while obj is not None and id(obj) not in chain:
            chain.add(id(obj))
            if isinstance(obj, CachedAutotuner):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    entries = kernels.setdefault(obj.kernel_name, {})
                    for key, cfg in obj.cache.items():
                        config = {k: getattr(cfg, k) for k in
                                  ('kwargs', 'num_warps', 'num_stages', 'num_ctas', 'maxnreg', 'ir_override')
                                  if hasattr(cfg, k)}
                        entry = {'autotune_key': AutotuneKey.normalize_autotune_key(key), 'config': config}
                        digest = AutotuneKey.key_hash(key)
                        if digest in entries and entries[digest] != entry:
                            raise RuntimeError('conflicting measured config for ' + obj.kernel_name)
                        entries[digest] = entry
                break
            obj = getattr(obj, 'fn', None)
assert kernels and all(kernels.values()), 'no measured kernel configs'
for kernel, entries in kernels.items():
    Path('/out', kernel + '.json').write_text(json.dumps({
        'kernel_name': kernel, 'triton_version': triton.__version__, 'autotune_entries': entries
    }, sort_keys=True, indent=2))
print(torch.cuda.get_device_name(0), 'kernels', len(kernels),
      'keys', sum(map(len, kernels.values())))
PY
```

Startup + 2,000 short items do **not** prove full shape coverage. Extend calibration
to every admitted prompt/token length, question/options shape and CUDA-graph path.
The patched reference rejects missing keys even though FLA strict mode ordinarily
falls back to autotuning. Test the rebuilt reference with an intentionally omitted
key: it must fail startup/request and yield no miner score. Arbitrary candidate
images can ignore env vars; correctness comparisons remain mandatory.

Publish configs with SHA256s and exact GPU name/SM count, driver, CUDA, Triton, FLA,
weights revision, original image digest, source revision and workload provenance.
Copy configs into the matching image directory, rebuild the reference image and
record its new digest. Add only the measured class to `PUBLISHED_GPU_CLASSES`.
Never reuse the Workstation config for Server Edition or vice versa without measurement.

## Noise, answers and scoring

1. Across at least 20 fresh-process idle sessions and 20 loaded-start sessions,
   measure reference throughput with the released timed workload, concurrency and
   runs; preserve every result. Do not treat within-process repeats as fresh starts.
2. Compute `statistics.stdev(throughputs) / statistics.mean(throughputs)` for each
   arm/class. Publish the conservative maximum as `PINNED_REFERENCE_NOISE_STD`,
   update the frozen threshold digest, and verify both gain/margin use
   `max(0.03, 3 * measured_relative_std)`. Record the exact measurement definition.
3. Run reference-versus-reference on the same 2,000 items across fresh idle/loaded
   starts, at least four paired trials, against the new digest with strict configs.
   Keep bitwise differences, distance distribution and top-choice agreement.
4. Verify the per-session repeatability check, missing-config VOID events, no miner
   attempt/strike on VOID, two distinct invalid champion sessions before vacating,
   reset on a valid session, and worker restart between invalid sessions.
5. Check serving reference drift separately. The serving s1-fast sidecar also uses
   JevK5/FLA, but this patch does not activate strict mode in serving compose files.
   Preserve existing serving policy until that separate validation is reviewed.

Only after review: update reference digest/rules, miner notice and start prerequisites.
This worker neither ran these GPU commands nor built/pushed/deployed an image.
