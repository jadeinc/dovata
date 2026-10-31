# model/ — the model's own vocabulary, as data

Tool-call delimiters, turn markers, chat templates, and the per-model limits the engine
reads. All of it TOML in [`profiles/`](profiles/README.md), compiled in with
`include_str!` and validated by a unit test.

## Why it exists

Because *model specifics are data*, the convention in the root `CLAUDE.md`. Swapping a
`.gguf` must not mean editing the engine, and the failure mode when it does is quiet:
tool calling breaks while generation looks perfect, because the delimiters the engine
looks for are not the ones the model emits.

A peer of [`load/`](../load/README.md) rather than part of it. The profile is chosen
during admission but used on every request — every prompt is rendered through the active
one — so it outlives the thing that picks it.

## What each file does

- `mod.rs` — the typed schema (`ModelProfile`, `ChatTemplate`, `Limits`) and the registry
  of bundled profiles, selectable by model name with architecture as the tiebreak.
- `profiles/` — the profiles themselves, plus the Jade-side packages generated from them.

## The two-file drift, and what stops it

A profile is two files describing one model: the `.toml` compiled into the engine, and
the `.jde` package published to jadelang.org so a Jade program can parse tool calls out
of a reply. They ship on separate pipelines with no runtime link, and they drifted — the
TOML said the tool-name key was `tool_name`, the Jade package said `name`, and no build
step could notice.

The TOML is the source of truth now. `tools/scripts/gen-profile-package.py` copies the
shared constants into the marked region of the `.jde`, and `--check` fails if one is
stale.

## What 3.0 changed here

The runtime parser that split a token stream into text and tool calls, `stream.rs`, went
with the rest of the runtime; it is on `main`. So did the workflow that published the
`.jde` packages and ran `--check`, which means nothing runs the drift check right now.
Whether 3.0 still publishes the `.jde` packages, and who parses tool calls at run time,
is an open question in the 3.0 plan.
