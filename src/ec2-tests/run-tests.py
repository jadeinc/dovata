#!/usr/bin/env python3
"""JADE OS — Run inference integration tests.

Designed to run on the EC2 instance, either directly or via the deploy step
over SSH. REPO is derived from the script's own location so it works regardless
of $HOME.

Usage:
  python3 src/ec2-tests/run-tests.py           # binaries must already be built
  python3 src/ec2-tests/run-tests.py --build   # cargo-build jadec + dovata first
"""
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def info(msg):
    print(f"==> {msg}")


def ok(msg):
    print(f"    OK: {msg}")


_daemon = {"proc": None}


def cleanup():
    proc = _daemon["proc"]
    if proc is not None and proc.poll() is None:
        info(f"Stopping dovata (pid {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    cleanup()
    sys.exit(code)


def main(argv):
    atexit.register(cleanup)

    build = "--build" in argv[1:]

    jade = Path("/usr/bin/jade")
    dovata = REPO / "target" / "release" / "dovata"
    model_path = Path(os.environ.get("JADE_MODEL_PATH", "/jade/inference/model.gguf"))
    tests_dir = REPO / "src" / "ec2-jade-tests"

    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    )

    # ── Optional build ─────────────────────────────────────────────────────
    if build:
        info("Building jadec...")
        subprocess.run("cargo build --release -p jade 2>&1 | tail -3",
                       shell=True, cwd=REPO)
        info("Building dovata...")
        # The local llama.cpp engine is compiled in by default on Linux; `cuda`
        # adds GPU offload (drop it for a CPU-only build).
        subprocess.run("cargo build --release -p dovata --features cuda 2>&1 | tail -3",
                       shell=True, cwd=REPO)
        ok("Build done")

    # ── Preflight ──────────────────────────────────────────────────────────
    if not (jade.exists() and os.access(jade, os.X_OK)):
        die(f"jade not found at {jade} — run the deploy step")
    if not dovata.is_file():
        die(f"dovata not found: {dovata}")
    if not model_path.is_file():
        die(f"model not found: {model_path} (run first-run.py first)")

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    for line in gpu.stdout.splitlines():
        if line.strip():
            ok(f"GPU: {line.strip()}")

    # ── Start dovata ───────────────────────────────────────────────────────
    # Kill any existing dovata first — a running repl instance holds the socket
    # path; starting a second clobbers it, leaving the original with an unlinked
    # socket fd (ECONNREFUSED on next connect).
    if subprocess.run(["pgrep", "-x", "dovata"],
                      stdout=subprocess.DEVNULL).returncode == 0:
        info("Stopping existing dovata before test run...")
        subprocess.run(["pkill", "-x", "dovata"])
        time.sleep(2)

    (Path.home() / ".jade").mkdir(parents=True, exist_ok=True)
    sock = Path.home() / ".jade" / "llm.sock"
    if sock.exists():
        sock.unlink()

    info(f"Starting dovata (model: {model_path})...")
    log_path = Path("/tmp/dovata.log")
    env = dict(os.environ, JADE_MODEL_PATH=str(model_path))
    log_file = open(log_path, "w")
    proc = subprocess.Popen([str(dovata)], stdout=log_file, stderr=subprocess.STDOUT,
                            env=env)
    _daemon["proc"] = proc

    info("Waiting for dovata to load model...")
    for _ in range(60):
        if log_path.exists() and "dovata: listening on" in log_path.read_text():
            ok("dovata ready")
            break
        if proc.poll() is not None:
            print("dovata log:")
            print(log_path.read_text())
            die("dovata exited unexpectedly")
        time.sleep(2)

    # ── Run test suite ─────────────────────────────────────────────────────
    info(f"Running test suite: {tests_dir}")
    print()

    passed = 0
    failures = 0
    for test_file in sorted(tests_dir.glob("*.jde")):
        name = test_file.stem
        result = subprocess.run([str(jade), "run", str(test_file)],
                                capture_output=True, text=True)
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
