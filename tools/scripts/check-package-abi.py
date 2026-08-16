#!/usr/bin/env python3
"""Refuse to publish a Jade package built against an outdated value ABI.

The Jade runtime carries a value ABI version, bumped when the representation of a
value crossing the native boundary changes. It went 1 -> 2 in jade v1.1.31, when
structs began crossing. Every package this repo had published was built with the
pinned v1.1.23 toolchain, so they all carry ABI 1, and a program on v1.1.31 that
called one got "native function returned an unknown value tag" from somewhere
inside the call — naming neither the version nor the fix. jade v1.1.32 turns that
into a refusal at load with a real message.

The refusal makes the breakage legible; it does not prevent it. Preventing it is
this script's job: it reads the ABI back out of each freshly built `.so` and fails
the publish if it is below the floor, so the toolchain pin drifting behind the
runtime cannot quietly ship broken packages again.

A package declares its ABI one of two ways:

  * `jade_pkg_abi_version` — stamped into every package by `jade build --lib` from
    v1.1.32 on. Emitted rather than borrowed from the linked runtime, because the
    linker drops an unreferenced `jrt_abi_version`.
  * `jrt_abi_version` — the runtime symbol, re-exported. What packages built before
    the stamp existed carry, and how they are read.

A library with neither does not link the Jade runtime at all (a `jade pkg add
--c-abi` shim). Nothing to check, so it passes.

Usage:
    ./check-package-abi.py out/*.so              # floor of 2 (jade v1.1.31+)
    ./check-package-abi.py --min 2 out/*.so
"""
import argparse
import ctypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import die, info, ok, warn  # noqa: E402

# The value ABI jade v1.1.31 introduced. A package below this is refused at load by
# v1.1.32 and misbehaves at runtime on v1.1.31.
DEFAULT_MIN_ABI = 2

# In preference order: the stamp first, since a package carrying both was built by a
# toolchain that emits the stamp and that is the authoritative one.
ABI_SYMBOLS = ("jade_pkg_abi_version", "jrt_abi_version")


def read_abi(path):
    """(symbol, version) for `path`, or (None, None) if it links no Jade runtime."""
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as e:
        die(f"{path.name}: cannot load: {e}")
    for sym in ABI_SYMBOLS:
        try:
            fn = getattr(lib, sym)
        except AttributeError:
            continue
        fn.restype = ctypes.c_uint32
        fn.argtypes = []
        return sym, fn()
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("packages", nargs="+", help="built .so packages to check")
    ap.add_argument("--min", type=int, default=DEFAULT_MIN_ABI,
                    help=f"minimum acceptable value ABI (default {DEFAULT_MIN_ABI})")
    args = ap.parse_args()

    stale = []
    for name in args.packages:
        path = Path(name)
        if not path.exists():
            die(f"no such package: {path}")

        sym, abi = read_abi(path)
        if abi is None:
            # A C-ABI shim links no Jade runtime, so it has no value ABI to mismatch.
            warn(f"{path.name}: no Jade runtime linked, skipping")
        elif abi < args.min:
            warn(f"{path.name}: value ABI {abi} (from {sym}), need >= {args.min}")
            stale.append(path.name)
        else:
            ok(f"{path.name}: value ABI {abi} (from {sym})")

    if stale:
        info("A package below the floor was built by a toolchain older than the "
             "runtime it will be loaded into.")
        info("Raise JADE_VERSION in .github/workflows/publish-providers.yml to a "
             "release at or above jade v1.1.31, then rebuild.")
        die(f"{len(stale)} package(s) built against an outdated value ABI")


if __name__ == "__main__":
    main()
