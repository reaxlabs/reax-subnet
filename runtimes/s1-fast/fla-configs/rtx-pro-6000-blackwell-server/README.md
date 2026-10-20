# Published FLA configs: NVIDIA RTX PRO 6000 Blackwell Server Edition

Measured, not invented. Do not edit any file here: `src/reax_subnet/engconfigs.py` pins the sha256 of
`SHA256SUMS`, and the engine runner verifies the whole directory before every session (mismatch = VOID session).

Provenance
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (engbox, idle), 20 Oct 2026.
- Image: `ghcr.io/reaxlabs/reax-engine-s1-fast@sha256:9bd41632e33bd90377f7e29bdb9410b18cfe1fb557200160b9ebe9222abcecea`
  (flash-linear-attention 0.5.2, Triton 3.7.1), the 2k-item engine workload (board #21985).
- Kernels (`@fla_cache_autotune`, 7 autotune keys): chunk_local_cumsum_scalar_kernel, l2norm_fwd_kernel (2 keys),
  chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_fwd_kernel_o, recompute_w_u_fwd_kernel,
  chunk_gated_delta_rule_fwd_kkt_solve_kernel. `_meta.json` records the autotuners and is not read by fla.
- Noise: 11 idle fresh-process reference samples, mean 65.70 decisions/s, sample std / mean = 0.00167
  (engconfigs.PINNED_REFERENCE_NOISE_STD). 15 fresh starts gave bit-identical answer dumps.
- l2norm_fwd_kernel, NB 3-256 (20 Oct 2026, board #22151/#22205): the measured run only hit NB 1-2 (rows <= 131072),
  so on a strict-mode miss a candidate engine without its own pin would autotune l2norm and could pick a minority
  bit-class (BT8/w8, BT16/w16, BT8/w16; ~1 in 10 starts). Entries for NB 3-256 (rows up to 16.7M) are added with
  BT=16/num_warps=4, the config the reference engine forces in `pin_fla_autotune()`; it is in the majority class at
  every measured row count. The two measured NB 1-2 entries are unchanged (both majority class).
- Coverage: only these six kernels consume the pins; plain `@triton.autotune` kernels would still autotune at run time
  (none ran in the workload). Strict mode falls back to autotune on an exact-key miss.
- The image ships no configs: the runner mounts this directory read-only at
  `/opt/reax/fla-configs/rtx-pro-6000-blackwell-server` with `FLA_CACHE_MODE=strict` and `FLA_CONFIG_DIR` pointing there.
- `SHA256SUMS` was generated from the measured files on Sandy (the engbox hand-back carried no checksum file);
  verify with `sha256sum -c SHA256SUMS`.
