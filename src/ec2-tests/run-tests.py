#!/usr/bin/env python3
"""JADE OS — run the inference evals against the real engine on the GPU box.

Designed to run on the EC2 instance, either directly or over SSH. REPO is derived
from the script's own location so it works regardless of $HOME.

Usage:
  python3 src/ec2-tests/run-tests.py           # library must already be built
  python3 src/ec2-tests/run-tests.py --build   # cargo-build the engine first

## There is no daemon to start

This script used to build a `dovata` binary, spawn it, wait for a
`dovata: listening on` line, and let the evals reach it over `~/.jade/llm.sock`.
None of that exists any more — dovata ships only as a provider `.so`, which a Jade
program dlopens and calls in-process. The whole start-and-wait block is gone, and
so is the cleanup that killed the daemon afterwards; there is no process to leak.

What replaces it is a staging step. The Jade runtime loads exactly one provider
library, from the active slot, and `JADE_PROVIDER_ACTIVE` overrides that slot's
path outright — the hook jadelang provides for dev and testing. So the engine is
copied into a scratch slot under `target/` and the evals are pointed at it. Note
this is *not* `JADE_PROVIDERS_DIR`: that one is CLI-side discovery for
`jade register`, and the runtime never reads it.

The evals themselves needed no changes. They speak `use llm` / `prompt` / `?p`,
which is language surface, not transport.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Per-eval ceiling. Generation on an A10G is seconds, not minutes; a test past
# this is wedged, and failing it beats hanging the suite.
TEST_TIMEOUT_S = 300


def info(msg):
    print(f"==> {msg}")


def ok(msg):
    print(f"    OK: {msg}")


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def find_jade():
    """The jade toolchain binary, however this box happens to have it installed."""
    found = shutil.which("jade") or "/usr/bin/jade"
    p = Path(found)
    if not (p.exists() and os.access(p, os.X_OK)):
        die(f"jade not found at {p} — run the deploy step")
    return p


def stage_provider(lib):
    """Copy the engine into a scratch active slot and return that directory.

    The runtime takes the single `.so` in the slot, so the directory is rebuilt
    from scratch each run: a leftover library from an earlier build would be
    picked up ahead of, or instead of, the one just compiled.
    """
    slot = REPO / "target" / "provider-active"
    if slot.exists():
        shutil.rmtree(slot)
    slot.mkdir(parents=True)
    shutil.copy2(lib, slot / "dovata.so")
    return slot


def assert_real_provider(lib):
    """Fail early if the cdylib is the non-Linux stub rather than a real provider.

    `export_provider!` is `#[cfg(target_os = "linux")]`, so a library built where
    the engine is compiled out still exists and still has a plausible size — it
    just exports nothing. Loading it fails far downstream with a null handle, so
    check the symbol here where the message can say why.
    """
    out = subprocess.run(["nm", "-D", "--defined-only", str(lib)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return  # nm missing or unhappy; not worth failing the run over
    if "ovata_provider_abi_version" not in out.stdout:
        die(f"{lib.name} exports no provider symbols — built with the engine "
            f"compiled out (is this a Linux build?)")


def main(argv):
    build = "--build" in argv[1:]

    jade = find_jade()
    lib = REPO / "target" / "release" / "libdovata.so"
    model_path = Path(os.environ.get("JADE_MODEL_PATH", "/jade/inference/model.gguf"))
    tests_dir = REPO / "src" / "ec2-jade-tests"

    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # ── Optional build ─────────────────────────────────────────────────────
    if build:
        info("Building the engine (dovata, cuda)...")
        # The local llama.cpp engine is compiled in by default on Linux; `cuda`
        # adds GPU offload (drop it for a CPU-only build).
        r = subprocess.run("cargo build --release -p dovata --features cuda",
                           shell=True, cwd=REPO)
        if r.returncode != 0:
            die("engine build failed")
        ok("Build done")

    # ── Preflight ──────────────────────────────────────────────────────────
    if not lib.is_file():
        die(f"engine library not found: {lib} — run with --build")
    assert_real_provider(lib)
    if not model_path.is_file():
        die(f"model not found: {model_path} (run first-run.py first)")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    for line in gpu.stdout.splitlines():
        if line.strip():
            ok(f"GPU: {line.strip()}")

    # ── Point the runtime at the engine ────────────────────────────────────
    slot = stage_provider(lib)
    ok(f"engine staged: {slot}/dovata.so")
    env = dict(os.environ,
               JADE_PROVIDER_ACTIVE=str(slot),
               JADE_MODEL_PATH=str(model_path))

    # ── Run test suite ─────────────────────────────────────────────────────
    info(f"Running test suite: {tests_dir}")
    info(f"Model: {model_path} (loaded in-process, per eval)")
    print()

    passed = 0
    failures = 0
    for test_file in sorted(tests_dir.glob("*.jde")):
        name = test_file.stem
        try:
            result = subprocess.run([str(jade), "run", str(test_file)],
                                    capture_output=True, text=True,
                                    env=env, timeout=TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            print(f"    FAIL: {name} — timed out after {TEST_TIMEOUT_S}s")
            failures += 1
            continue
        output = (result.stdout or "") + (result.stderr or "")
        pass_line = next((ln for ln in output.splitlines() if ln.startswith("PASS:")), "")
        if pass_line:
            ok(pass_line[len("PASS: "):])
            passed += 1
        else:
            non_empty = [ln for ln in output.splitlines() if ln.strip()]
            diag = non_empty[-1] if non_empty else "no output"
            print(f"    FAIL: {name} — {diag}")
            failures += 1

    print()
    info(f"Results: {passed} passed, {failures} failed")
    if failures != 0:
        print()
        die(f"{failures} test(s) failed")
    ok("All tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
