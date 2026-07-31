#!/usr/bin/env python3
"""Generate the Jade-side model profile's constants from the engine-side TOML.

A model profile is two files that describe one model: `<model>.toml`, compiled
into the engine (`dovata::profile::Registry::builtin()`), and `<model>.jde`, a
Jade package published to jadelang.org so a program can parse tool calls out of
a reply. They ship on entirely separate pipelines and nothing at runtime links
them, so for a while they were two hand-written copies of the same three values.
They drifted: the TOML said the tool-name JSON key was `tool_name`, the Jade
package said `name`, and no build step could notice.

So the TOML is now the source of truth, and the `.jde`'s constants are generated
from it. Only the constants — the parsing body stays hand-written, because the
models genuinely differ in it. Qwen3-Coder wraps a JSON object; the qwen35moe
family wraps nested XML and deliberately has no name field at all. One template
cannot serve both, but they do share the delimiters, which is what drifted.

Usage:
    ./gen-profile-package.py              # rewrite every .jde in place
    ./gen-profile-package.py --check      # fail if any .jde is out of date (CI)
    ./gen-profile-package.py qwen3-coder-30b.toml   # just one

Profiles with no sibling `.jde` are skipped: not every model has a published
language-side package, and the engine works fine without one.
"""
import argparse
import sys
from pathlib import Path

try:
    import tomllib  # stdlib from 3.11; the only reason this script needs a floor
except ModuleNotFoundError:
    sys.exit(f"ERROR: needs Python 3.11+ for tomllib (running {sys.version.split()[0]})")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import die, info, ok, warn  # noqa: E402

# <root>/src/scripts/this.py → <root>
ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "src" / "dovata" / "profiles"

BEGIN = "// ── BEGIN GENERATED — from {toml}; run src/scripts/gen-profile-package.py"
END = "// ── END GENERATED"

# Mirrors `default_name_field()` in src/dovata/src/profile/mod.rs. A profile that
# omits the key gets the same value the engine's serde default gives it, so the
# two sides still agree on what the engine will actually look for.
DEFAULT_NAME_FIELD = "tool_name"


def jade_string(s):
    """A Jade double-quoted string literal for `s`."""
    out = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + out.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r") + '"'


def block_for(toml_path):
    """The generated constants block for one profile TOML."""
    with open(toml_path, "rb") as f:
        profile = tomllib.load(f)

    tool_call = profile.get("tool_call")
    if not tool_call:
        die(f"{toml_path.name}: no [tool_call] section")
    for key in ("open", "close"):
        if not tool_call.get(key):
            die(f"{toml_path.name}: [tool_call] {key} is missing or empty")

    model = profile.get("model", toml_path.stem)
    name_field = tool_call.get("name_field", DEFAULT_NAME_FIELD)

    lines = [
        BEGIN.format(toml=toml_path.name),
        f"// Model: {model}. Edit the TOML, not these three lines.",
        f"let TOOL_OPEN  = {jade_string(tool_call['open'])}",
        f"let TOOL_CLOSE = {jade_string(tool_call['close'])}",
        f"let NAME_FIELD = {jade_string(name_field)}",
        END,
    ]
    return "\n".join(lines)


def splice(text, block, jde_path):
    """Replace the marked region of `text` with `block`."""
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("// ── BEGIN GENERATED")]
    ends = [i for i, l in enumerate(lines) if l.startswith(END)]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        die(f"{jde_path.name}: expected exactly one BEGIN/END GENERATED region "
            f"(found {len(starts)} begin, {len(ends)} end)")
    body = lines[:starts[0]] + block.splitlines() + lines[ends[0] + 1:]
    return "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles", nargs="*",
                    help="profile TOMLs (default: every one with a sibling .jde)")
    ap.add_argument("--check", action="store_true",
                    help="do not write; exit 1 if any .jde is out of date")
    args = ap.parse_args()

    if args.profiles:
        tomls = [Path(p) if Path(p).exists() else PROFILES / p for p in args.profiles]
        for t in tomls:
            if not t.exists():
                die(f"no such profile: {t}")
    else:
        tomls = sorted(PROFILES.glob("*.toml"))

    stale = []
    handled = 0
    for toml_path in tomls:
        jde_path = toml_path.with_suffix(".jde")
        if not jde_path.exists():
            info(f"{toml_path.name}: no sibling .jde, skipping")
            continue
        handled += 1

        current = jde_path.read_text()
        updated = splice(current, block_for(toml_path), jde_path)

        if current == updated:
            ok(f"{jde_path.name} up to date")
        elif args.check:
            warn(f"{jde_path.name} is out of date with {toml_path.name}")
            stale.append(jde_path)
        else:
            jde_path.write_text(updated)
            info(f"{jde_path.name} regenerated from {toml_path.name}")

    if not handled:
        warn("no profiles had a sibling .jde — nothing to do")
    if stale:
        die(f"{len(stale)} profile package(s) out of date — "
            f"run src/scripts/gen-profile-package.py and commit the result")


if __name__ == "__main__":
    main()
