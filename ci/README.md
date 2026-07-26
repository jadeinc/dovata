# ci — pinned build inputs

Two files that pin what the production model is made of. No code, no scripts — this is
data that other things read.

## Why it is separate

The model is a build artifact like any other, and it needs to be reproducible: the same
inputs must always yield the same GGUF. Scattering the source repo, the revision, the
quantization scheme, and the llama.cpp commit across a shell script and a workflow file
means they drift, and a drifting llama.cpp commit is the one that hurts — quantize with a
different commit than the daemon links and the GGUF can be rejected outright for
quant-type or GGUF-version skew.

So they live here in one flat `KEY=VALUE` file that both a Python script and a workflow
step can read without a parser. Bumping a value here is the intended way to cut a new
model release.

## Files

- `model-version.lock` — the single source of truth for the published model. Records the
  HF repo and a full commit SHA (not a branch, so "latest" cannot silently change what
  ships), the quant scheme and imatrix chunk count, the llama.cpp repo and commit, and
  the CUDA arch. Its comments carry the provenance of the llama.cpp pin and how to
  re-derive it when the engine is bumped.
- `calib-code.txt` — the calibration corpus for the importance matrix. Q4_K_M is driven
  by an imatrix calibrated on code, which meaningfully improves quality for a coder model.

## Who reads them

`src/scripts/quantize-model.py` reads `model-version.lock` for its env defaults and
`calib-code.txt` for the imatrix pass. `src/ec2-tests/quantize.py` drives that script on
the GPU box. The output lands at
`s3://jade-os-release-bucket/models/model.gguf`.

## The one rule

`LLAMA_CPP_COMMIT` must match the commit the `src/infer-engine/ffi/llama.cpp` submodule
is pinned at. Bumping the engine without bumping this — or the reverse — is how you ship
a model the daemon cannot load. Nothing enforces the pairing automatically; check it by
hand.
