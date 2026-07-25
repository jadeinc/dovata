#!/usr/bin/env python3
"""Produce the JADE OS production inference model: download (or take locally-merged)
weights, convert to GGUF f16, build a code-calibrated importance matrix, and
quantize to Q4_K_M — using the EXACT llama.cpp commit that this repo's dovata
links, so the GGUF can never be rejected for quant-type / GGUF-version skew.

Outputs (into --out-dir, default ./dist/model):
  model.gguf            the Q4_K_M quantized model
  model.gguf.sha256     `sha256sum model.gguf` output (jade-os reads field 1)
  model.gguf.buildinfo  provenance (source rev, llama.cpp commit, sizes) — NOT
                        uploaded to S3; for release notes / debugging only.

This is the permanent self-quantization path: post-LoRA-fine-tune there is no
community GGUF, so it accepts locally-merged weights too (--local-dir), not
just an HF download.

Heavy host required: peak ~150 GB disk (safetensors ~60 + f16 GGUF ~60 +
Q4_K_M ~18) and a CUDA GPU for the imatrix pass — it can't run on a laptop or a
GitHub-hosted runner. Run it on the EC2 GPU box; src/ec2-tests/quantize.py is the
one-command driver (start box -> sync repo -> run this -> upload in-region).

Usage:
  python3 src/scripts/quantize-model.py                       # pins from ci/model-version.lock
  python3 src/scripts/quantize-model.py --local-dir ./merged  # use locally-merged weights
  python3 src/scripts/quantize-model.py --upload              # also push to the S3 contract
  python3 src/scripts/quantize-model.py --out-dir /tmp/out

Key env overrides (defaults come from ci/model-version.lock):
  HF_REPO HF_REVISION QUANT IMATRIX_CHUNKS LLAMA_CPP_COMMIT CUDAARCHS
  HF_TOKEN                 # for gated/private repos (Qwen3-Coder is ungated)
  JADE_QUANTIZE_WORKDIR    # persistent build/download cache (default ~/jade-quantize)
  JADE_IMATRIX_OFFLOAD     # llama-imatrix offload args (default: -ngl 99 --cpu-moe)
  MIN_DISK_GB              # preflight free-space floor (default 150)
  S3_BUCKET S3_PREFIX      # upload target (default jade-os-release-bucket / models)
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import info, ok, die, have, du_sh, run, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pins loaded (as defaults) from the lock; env wins over the lock.
LOCK_KEYS = [
    "HF_REPO", "HF_REVISION", "QUANT", "IMATRIX_CHUNKS",
    "LLAMA_CPP_REPO", "LLAMA_CPP_COMMIT", "CUDAARCHS",
]


def load_lock(lock_path):
    """Load `KEY=VALUE` pins as DEFAULTS: a non-empty env value wins; the lock
    fills the rest. Returns the effective values and also exports them."""
    if not lock_path.is_file():
        die(f"missing {lock_path}")
    for raw in lock_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        if not os.environ.get(key):
            os.environ[key] = val.strip()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--local-dir", default="")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "dist" / "model"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args, unknown = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return 0
    if unknown:
        die(f"unknown arg: {unknown[0]} (try --help)")

    local_dir = args.local_dir
    out_dir = Path(args.out_dir)
    do_upload = args.upload

    # --- pinned inputs (ci/model-version.lock is the source of truth) --------
    load_lock(REPO_ROOT / "ci" / "model-version.lock")
    hf_repo = os.environ["HF_REPO"]
    hf_revision = os.environ["HF_REVISION"]
    quant = os.environ["QUANT"]
    imatrix_chunks = os.environ["IMATRIX_CHUNKS"]
    llama_repo = os.environ.get("LLAMA_CPP_REPO") or "https://github.com/ggml-org/llama.cpp"
    llama_commit = os.environ["LLAMA_CPP_COMMIT"]
    cuda_archs = os.environ.get("CUDAARCHS") or "86"

    calib = Path(os.environ.get("CALIB", REPO_ROOT / "ci" / "calib-code.txt"))
    if not calib.is_file():
        die(f"missing calibration corpus {calib}")

    workdir = Path(os.environ.get("JADE_QUANTIZE_WORKDIR", Path.home() / "jade-quantize"))
    imatrix_offload = os.environ.get("JADE_IMATRIX_OFFLOAD", "-ngl 99 --cpu-moe")
    min_disk_gb = int(os.environ.get("MIN_DISK_GB", "150"))
    s3_bucket = os.environ.get("S3_BUCKET", "jade-os-release-bucket")
    s3_prefix = os.environ.get("S3_PREFIX", "models")

    info("JADE OS model quantization")
    print(f"    HF_REPO          : {hf_repo}")
    print(f"    HF_REVISION      : {hf_revision}")
    print(f"    QUANT            : {quant}  (imatrix chunks cap: {imatrix_chunks})")
    print(f"    LLAMA_CPP_COMMIT : {llama_commit}")
    print(f"    CUDAARCHS        : {cuda_archs}")
    print(f"    workdir          : {workdir}")
    print(f"    out-dir          : {out_dir}")
    if local_dir:
        print(f"    local weights    : {local_dir} (skipping HF download)")
    print()

    workdir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- preflight: disk -----------------------------------------------------
    avail_gb = shutil.disk_usage(workdir).free // (1024 ** 3)
    info(f"free disk on {workdir} filesystem: {avail_gb} GB (need >= {min_disk_gb})")
    if avail_gb < min_disk_gb:
        die(f"insufficient disk: {avail_gb} GB < {min_disk_gb} GB. "
            "Attach a larger volume or set MIN_DISK_GB.")

    # --- preflight: tools ----------------------------------------------------
    for tool in ("git", "cmake", "python3"):
        if not have(tool):
            die(f"{tool} not found")
    if not have("nvcc"):
        info("WARNING: nvcc not on PATH — CUDA build of imatrix may fail (imatrix wants a GPU)")

    # --- 1. build llama.cpp tools at the pinned commit -----------------------
    llama_src = workdir / "llama.cpp"
    built_marker = llama_src / f".jade-built-{llama_commit}"
    if built_marker.is_file():
        ok(f"llama.cpp tools already built at {llama_commit} (cached)")
    else:
        info(f"fetching llama.cpp @ {llama_commit}")
        if not (llama_src / ".git").is_dir():
            run(["git", "init", "-q", str(llama_src)])
            run(["git", "-C", str(llama_src), "remote", "add", "origin", llama_repo])
        run(["git", "-C", str(llama_src), "fetch", "--depth", "1", "origin", llama_commit])
        run(["git", "-C", str(llama_src), "checkout", "-q", "-f", llama_commit])

        info(f"building llama-imatrix + llama-quantize (CUDA arch {cuda_archs})")
        run(["cmake", "-S", str(llama_src), "-B", str(llama_src / "build"),
             "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
             f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
             "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF"])
        run(["cmake", "--build", str(llama_src / "build"),
             "--target", "llama-imatrix", "llama-quantize",
             "-j", str(os.cpu_count() or 1)])
        built_marker.touch()
        ok("llama.cpp tools built")

    imatrix_bin = llama_src / "build" / "bin" / "llama-imatrix"
    quantize_bin = llama_src / "build" / "bin" / "llama-quantize"
    convert_py = llama_src / "convert_hf_to_gguf.py"
    for f in (imatrix_bin, quantize_bin, convert_py):
        if not f.exists():
            die(f"expected tool missing after build: {f}")

    # --- 2. python env for the converter -------------------------------------
    # The convert script's pinned deps only ship wheels for mainstream Python
    # (<=3.12); provision an isolated known-good CPython via uv — no apt/sudo.
    venv = workdir / "venv"
    pyver = os.environ.get("JADE_PYTHON_VERSION", "3.12")
    os.environ["PATH"] = os.pathsep.join([
        str(Path.home() / ".local" / "bin"),
        str(Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")) / "bin"),
        os.environ.get("PATH", ""),
    ])
    if not have("uv"):
        info("installing uv (python/dep manager)")
        run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True)
    if not have("uv"):
        die("uv not on PATH after install")

    info(f"creating Python {pyver} venv via uv")
    # --clear makes this idempotent across reruns (uv caches CPython + wheels).
    run(["uv", "venv", "--clear", "--python", pyver, str(venv)])
    venv_python = venv / "bin" / "python"

    def pyinstall(*pkg_args):
        run(["uv", "pip", "install", "--python", str(venv_python),
             "--index-strategy", "unsafe-best-match", *pkg_args])

    info("installing convert + huggingface deps (pinned via llama.cpp's requirements)")
    pyinstall("-r", str(llama_src / "requirements" / "requirements-convert_hf_to_gguf.txt"))
    pyinstall("huggingface_hub[cli,hf_transfer]")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # --- 3. obtain weights ---------------------------------------------------
    if local_dir:
        src_model = Path(local_dir)
        if not src_model.is_dir():
            die(f"--local-dir not a directory: {src_model}")
        ok(f"using local weights: {src_model}")
    else:
        src_model = workdir / "src-model"
        info(f"downloading {hf_repo} @ {hf_revision} -> {src_model}")
        hf_args = ["download", hf_repo, "--revision", hf_revision,
                   "--local-dir", str(src_model)]
        if os.environ.get("HF_TOKEN"):
            hf_args += ["--token", os.environ["HF_TOKEN"]]
        hf_cli = "hf" if have("hf") else "huggingface-cli"
        run([hf_cli, *hf_args])
        ok(f"weights present ({du_sh(src_model)})")

    # --- 4. convert -> f16 GGUF ---------------------------------------------
    f16 = workdir / "model-f16.gguf"
    info(f"convert_hf_to_gguf.py -> {f16} (f16)")
    run([str(venv_python), str(convert_py), str(src_model),
         "--outfile", str(f16), "--outtype", "f16"])
    ok(f"f16 GGUF ({du_sh(f16)})")

    # --- 5. importance matrix (code-calibrated) ------------------------------
    # The committed seed (ci/calib-code.txt) carries the deployment-specific
    # signal; unless disabled we deterministically augment it with real source
    # from the PINNED llama.cpp checkout to reach a solid chunk count.
    eff_calib = workdir / "calib-effective.txt"
    shutil.copyfile(calib, eff_calib)
    if os.environ.get("JADE_CALIB_AUGMENT", "1") != "0":
        budget = int(os.environ.get("JADE_CALIB_BUDGET_BYTES", "300000"))
        info(f"augmenting calibration from pinned llama.cpp source (budget {budget} B)")
        sources = []
        for sub in ("src", "common", "ggml/src"):
            base = llama_src / sub
            if base.is_dir():
                for f in base.iterdir():
                    if f.is_file() and f.suffix in (".cpp", ".c", ".cu", ".py"):
                        sources.append(f)
        with open(eff_calib, "ab") as out:
            for f in sorted(sources):
                out.write(f.read_bytes())
                out.write(b"\n")
                if eff_calib.stat().st_size >= budget:
                    break
    ok(f"effective calibration corpus: {du_sh(eff_calib)}")

    imat = workdir / "model.imatrix"
    info(f"llama-imatrix on code corpus ({imatrix_offload}, chunks<={imatrix_chunks})")
    run([str(imatrix_bin), "-m", str(f16), "-f", str(eff_calib), "-o", str(imat),
         "--chunks", str(imatrix_chunks), *shlex.split(imatrix_offload)])
    if not (imat.is_file() and imat.stat().st_size > 0):
        die("imatrix not produced")
    ok(f"imatrix ({du_sh(imat)})")

    # --- 6. quantize -> Q4_K_M ----------------------------------------------
    qout = workdir / f"model-{quant}.gguf"
    info(f"llama-quantize --imatrix -> {quant}")
    run([str(quantize_bin), "--imatrix", str(imat), str(f16), str(qout), quant])
    if not (qout.is_file() and qout.stat().st_size > 0):
        die("quantized model not produced")
    ok(f"quantized ({du_sh(qout)})")

    # --- 6b. GGUF sanity: valid v3 header + expected MoE arch ----------------
    info("verifying GGUF header")
    verify = r"""
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
ver = int(r.fields["GGUF.version"].parts[-1][0]) if "GGUF.version" in r.fields else r.version
arch_f = r.fields.get("general.architecture")
arch = bytes(arch_f.parts[arch_f.data[0]]).decode() if arch_f else "?"
assert ver == 3, f"unexpected GGUF version {ver} (dovata expects v3)"
assert "qwen3" in arch, f"unexpected architecture {arch!r}"
print(f"    OK: GGUF v{ver}, architecture={arch}, tensors={len(r.tensors)}")
"""
    run([str(venv_python), "-c", verify, str(qout)])

    # --- 7. finalize: canonical name + sidecar + buildinfo -------------------
    info(f"staging release artifacts in {out_dir}")
    final = out_dir / "model.gguf"
    shutil.copyfile(qout, final)
    digest = sha256_file(final)
    (out_dir / "model.gguf.sha256").write_text(f"{digest}  model.gguf\n")

    augment = os.environ.get("JADE_CALIB_AUGMENT", "1")
    size_bytes = final.stat().st_size
    (out_dir / "model.gguf.buildinfo").write_text(
        f"model_sha256={digest}\n"
        f"hf_repo={hf_repo}\n"
        f"hf_revision={hf_revision}\n"
        f"quant={quant}\n"
        f"imatrix=code-calibrated (ci/calib-code.txt seed + pinned llama.cpp src, "
        f"chunks<={imatrix_chunks}, augment={augment})\n"
        f"llama_cpp_repo={llama_repo}\n"
        f"llama_cpp_commit={llama_commit}\n"
        f"cuda_archs={cuda_archs}\n"
        f"size_bytes={size_bytes}\n"
    )

    ok(f"model.gguf      {du_sh(final)}  sha256={digest}")
    ok("model.gguf.sha256 + model.gguf.buildinfo written")
    print()
    print((out_dir / "model.gguf.buildinfo").read_text(), end="")

    # --- 8. optional upload to the cross-repo S3 contract --------------------
    if do_upload:
        # Prefer a system aws; otherwise install it into the build venv (rather
        # than depending on a system/monorepo aws) and call it by path.
        aws_cmd = shutil.which("aws")
        if not aws_cmd:
            info("aws CLI not found — installing awscli into the build venv")
            pyinstall("awscli")
            aws_cmd = str(venv / "bin" / "aws")
        dest = f"s3://{s3_bucket}/{s3_prefix}"
        info(f"uploading to {dest}/ (the jade-os contract)")
        run([aws_cmd, "s3", "cp", str(final), f"{dest}/model.gguf", "--no-progress"])
        run([aws_cmd, "s3", "cp", str(out_dir / "model.gguf.sha256"),
             f"{dest}/model.gguf.sha256", "--no-progress"])
        ok(f"published {dest}/model.gguf (+ .sha256)")

    print()
    ok(f"done. quantized with llama.cpp {llama_commit} "
       "(matches dovata's linked llama-cpp-sys-2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
