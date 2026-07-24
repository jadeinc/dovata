# Model profiles

A **profile** describes the model-specific token vocabulary the language layer
needs to interpret a model's output — tool-call delimiters, the tool-name field,
how a tool result is fed back, and any other special-token spans (reasoning, etc.).

This keeps model lingo out of Jade programs and out of the wire protocol. The
binary request/response protocol carries only opaque text + frames; `use llm`
loads the profile for whatever model the daemon reports (its `Meta` frame) and
uses it to build requests and parse the stream. **Swap the `.gguf` → drop in the
matching profile → no Jade program changes.**

## Packaging

- One TOML file per model, named for the model (`qwen3-coder-30b.toml`).
- Files in this directory are compiled into the daemon's `dovata::profile` module
  (`Registry::builtin()`), so the daemon ships with them.
- The daemon can also load external profiles at runtime and `Registry::push`
  them to override a builtin (e.g. ship a profile next to a custom model).
- Selection is by the daemon-read model name against each profile's `match`
  globs (first match wins; a pushed profile beats a builtin).

## Schema

See `../src/profile/mod.rs` for the typed schema. Required: `schema_version`, `model`,
`[tool_call]` (`open`, `close`, `name_field`), `[tool_result]` (`role`,
optional `wrap_prefix`/`wrap_suffix`). Optional: `match` globs and `[[span]]`
entries (`tag`, `open`, `close`) for extra special-token spans.

## Adding a model

1. Copy `qwen3-coder-30b.toml`, rename for the new model.
2. Set `model`, `match`, and the model's real tool/turn-taking tokens.
3. Add it to `BUILTIN_PROFILES` in `../src/profile/mod.rs` if it should ship by default.
4. `cargo test -p dovata` validates every bundled profile.
