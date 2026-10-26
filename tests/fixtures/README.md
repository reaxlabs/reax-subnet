# Engine slot regression fixtures

- `engine-accepted-pre-slots.json`: exact read-only copy of
  `/home/flori/jobs/sn92-fast-launch-20261015/engbox/hardening/state/history/cand-20261020-1950.json`.
  SHA256 `b2f7f0855d2ddc7916a30dab9caf09e6e47752ebc36d769329c5513fa740e303`.
- `candidate-batch.pre-engslots.sh`: exact read-only copy of
  `/home/flori/jobs/sn92-fast-launch-20261015/engbox/hardening/candidate-batch.sh` before this patch.
  SHA256 `8e7f09cc59b89ef01f21edbb5f54a49433d951f71f8c7a7f441c684b19bcb1ab`.
  Tests apply the proposed patch to a temporary copy; they never execute GPU phases, remote calls or production writes.

`engbatch-auto.pre-engslots.sh` is a read-only copy of the Sandy auto launcher for N1 shell-stub tests.
SHA256: `8164c984c0d89bff60af16c233c71cecf5bd9b4a3e7c076787e256b493aa840c`.
The generated `tools/engslots/engbatch-auto.patch` applies to this exact base.
