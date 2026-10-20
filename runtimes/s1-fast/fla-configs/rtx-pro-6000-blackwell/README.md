# UNMEASURED — engine competition release blocked

Placeholder only; no Triton tuning values have been invented. Measure FLA 0.5.2
configs once on an idle GPU of this exact class using the pinned reference image,
weights, model revision, CUDA/Triton versions and complete engine workload shapes.
Publish the actual FLA configs here with checksums, hardware/software provenance,
commands and repeated fresh-process reference throughput measurements (idle and
loaded). Verify strict mode never falls back to timing autotune for any shape.

Only then add this class to engconfigs.PUBLISHED_GPU_CLASSES and replace the
UNMEASURED PINNED_REFERENCE_NOISE_STD with sample std(throughputs)/mean(throughputs).
Both min gain and displacement margin use max(3%, 3 * this relative noise std).
For a second GPU class, measure independently; the shared noise constant must
conservatively cover every published class (or release a per-class rule).

The image contains this directory at /opt/reax/fla-configs/rtx-pro-6000-blackwell;
both reference and candidate must contain the exact published configs there.
The read-only root filesystem prevents a runtime from updating them.
