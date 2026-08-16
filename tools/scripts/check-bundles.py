#!/usr/bin/env python3
"""Check that every capability bundle still applies to the pinned llama.cpp.

A bundle is a capability's delta to llama.cpp: a patch series plus cmake defines,
living beside the capability instead of inside a fork. See
src/infer-engine/moe-driver/bundle/ for the shape and the sys crate's build.rs for
how it is staged and applied.

This script answers the one question that decides whether a build will work:
does the series still apply to the commit the submodule points at? It is fast,
needs no compiler, and runs in CI before the long CUDA build so a pin that moved
out from under a bundle fails in seconds rather than half an hour.

It also enforces the rule that keeps one pin enough for two consumers. The engine
links upstream *plus these patches*; `quantize-model.py` builds its tools from stock
upstream at the same commit. That is only sound while no patch touches how a GGUF is
produced or read, so a patch that edits the converter, gguf-py, or the quantizer is
rejected here — the alternative is discovering it as a published model.gguf the
engine refuses, after hours on a GPU box.

Usage:
  python3 tools/scripts/check-bundles.py            # check every bundle
  python3 tools/scripts/check-bundles.py --verbose  # name each patch as it is checked
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import die, info, ok, warn  # noqa: E402

try:
    import tomllib  # stdlib from 3.11
except ModuleNotFoundError:
    sys.exit(f"ERROR: needs Python 3.11+ for tomllib (running {sys.version.split()[0]})")

ROOT = Path(__file__).resolve().parents[2]
LLAMA_SUBMODULE = "src/infer-engine/ffi/llama.cpp"

# Mirrors FORBIDDEN_PATH_PREFIXES in src/infer-engine/bundle-kit/src/lib.rs. Two copies
# is a smell, and this one is deliberate: the Rust list gates the *build*, this one gates
# *review*, and a patch that reaches CI having skipped the build check is exactly the
# case worth catching. The bundle-kit tests pin the Rust side; if you add to one, add to
# both — a mismatch means CI passes something the build will refuse.
FORBIDDEN = (
    "convert_hf_to_gguf.py",
    "convert_hf_to_gguf_update.py",
    "gguf-py/",
    "tools/quantize/",
    "src/llama-quant.cpp",
)


def bundle_dirs():
    """Every directory under src/ holding a bundle.toml."""
    return sorted(p.parent for p in (ROOT / "src").rglob("bundle.toml"))


def pinned_commit():
    """The llama.cpp commit this repo pins.

    Read out of git rather than by asking the submodule what it has checked out, so it
    reports what the repo *pins* and works even where the submodule was never
    initialised. Same reasoning as quantize-model.py's `llama_cpp_pin`: one source, no
    hand-copied duplicate.

    From the *index* (`ls-files -s`) rather than HEAD (`ls-tree`). In CI the two are
    identical, because a fresh checkout has a clean index. Locally they differ exactly
    when someone has staged a pin bump and not committed it yet — which is the moment
    they most want to know whether the bundles still apply. Reading HEAD there would
    check the series against the commit being replaced and cheerfully pass.
    """
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", LLAMA_SUBMODULE],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        die(f"cannot read the {LLAMA_SUBMODULE} pin from git — is this a git checkout?")
    # "160000 <sha> 0\t<path>"; mode 160000 is a gitlink.
    parts = out.stdout.split()
    if len(parts) < 2 or parts[0] != "160000":
        die(f"{LLAMA_SUBMODULE} is not a submodule gitlink: {out.stdout.strip()}")
    return parts[1]


def patch_targets(text):
    """The files a unified diff writes to."""
    out = []
    for line in text.splitlines():
        if line.startswith("+++ "):
            p = line[4:].strip()
            if p == "/dev/null":
                continue
            out.append(p[2:] if p.startswith("b/") else p)
    return out


def check_bundle(bundle_dir, pin, workdir, verbose):
    """Check one bundle. Returns a list of problem strings (empty means fine)."""
    problems = []
    manifest = bundle_dir / "bundle.toml"
    with open(manifest, "rb") as f:
        m = tomllib.load(f)

    name = m.get("name", bundle_dir.parent.name)
    declared = m.get("upstream", "")
    patches = m.get("patches", [])

    if not patches:
        return [f"{name}: bundle.toml lists no patches"]

    if declared and declared != pin:
        # Not fatal on its own — the series may well still apply — but it means nobody
        # has re-verified the bundle since the pin moved, and the record now lies.
        warn(
            f"{name}: bundle.toml says upstream {declared[:12]}, the submodule pins "
            f"{pin[:12]}. If the checks below pass, update `upstream` in bundle.toml."
        )

    for rel in patches:
        p = bundle_dir / rel
        if not p.is_file():
            problems.append(f"{name}: bundle.toml names {rel}, which does not exist")
            continue

        text = p.read_text()
        for target in patch_targets(text):
            for bad in FORBIDDEN:
                if target.startswith(bad):
                    problems.append(
                        f"{name}: {p.name} edits {target}, which changes how a GGUF is "
                        f"produced or read. The engine links upstream+patches and the "
                        f"quantizer builds from stock upstream at the same pin; that is "
                        f"only sound while no patch touches these paths."
                    )

        r = subprocess.run(
            ["git", "apply", "--check", "--whitespace=nowarn", str(p)],
            cwd=str(workdir), capture_output=True, text=True,
        )
        if r.returncode != 0:
            problems.append(f"{name}: {p.name} does not apply\n      {r.stderr.strip()}")
        elif verbose:
            ok(f"{name}: {p.name}")

        # Apply as we go, so a later patch is checked against the state the earlier
        # ones leave behind. Checking each against pristine upstream would pass a
        # series whose patches conflict with each other.
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(p)],
            cwd=str(workdir), capture_output=True, text=True,
        )

    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", "-v", action="store_true", help="name each patch checked")
    args = ap.parse_args()

    dirs = bundle_dirs()
    if not dirs:
        ok("no capability bundles — the build uses stock llama.cpp unmodified")
        return 0

    pin = pinned_commit()
    info(f"llama.cpp pinned at {pin[:12]}")

    submodule = ROOT / LLAMA_SUBMODULE
    if not (submodule / ".git").exists():
        die(
            f"{LLAMA_SUBMODULE} is not checked out — run:\n"
            f"  git submodule update --init {LLAMA_SUBMODULE}"
        )

    all_problems = []
    with tempfile.TemporaryDirectory(prefix="check-bundles-") as tmp:
        work = Path(tmp) / "llama"
        # A worktree at the pinned commit: cheap (shares the object store) and it
        # cannot dirty the submodule the way applying in place would.
        r = subprocess.run(
            ["git", "worktree", "add", "--detach", "--force", str(work), pin],
            cwd=str(submodule), capture_output=True, text=True,
        )
        if r.returncode != 0:
            die(f"could not create a worktree at {pin[:12]}:\n{r.stderr.strip()}\n"
                f"If the commit is missing, fetch it: git -C {LLAMA_SUBMODULE} fetch origin")
        try:
            for d in dirs:
                info(f"checking bundle at {d.relative_to(ROOT)}")
                all_problems += check_bundle(d, pin, work, args.verbose)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(work)],
                           cwd=str(submodule), capture_output=True)

    if all_problems:
        print()
        for p in all_problems:
            print(f"ERROR: {p}", file=sys.stderr)
        die(f"{len(all_problems)} bundle problem(s) — the build would fail on these")

    ok(f"every bundle applies to {pin[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
