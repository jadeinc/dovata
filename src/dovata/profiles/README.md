# Model profiles

A **profile** describes the model-specific token vocabulary the language layer
needs to interpret a model's output — tool-call delimiters, the tool-name field,
how a tool result is fed back, and any other special-token spans (reasoning, etc.).

This keeps model lingo out of Jade programs and out of the wire protocol. The
binary request/response protocol carries only opaque text + frames; `use llm`
loads the profile for whatever model the daemon reports (its `Meta` frame) and
uses it to build requests and parse the stream. **Swap the `.gguf` → drop in the
matching profile → no Jade program changes.**

## Two artifacts per model

- **`<model>.toml`** — the daemon-side profile, compiled into `dovata::profile`
  (below). Used by the daemon to build requests / parse the stream.
- **`<model>.jde`** — the language-side profile, a standalone **Jade package** a
  program installs to parse the model's tool calls out of a reply. It is published as
  a native `.so` to jadelang.org (by the publish-providers workflow) alongside the
  provider packages, and installed like any dependency:

  ```sh
  jade pkg add qwen3_coder_30b \
    --url 'https://www.jadelang.org/qwen3-coder-30b-{platform}.so' --version 0.1.0
  ```

  ```jade
  use qwen3_coder_30b
  let call = qwen3_coder_30b.find_tool_call(reply)   // { name, args } or nil
  ```

  (The import name is a valid identifier of your choice; the `.so` filename keeps the
  hyphenated model name.)

Both artifacts describe the same model, so they must agree on the delimiters and the
tool-name field. They are separate files because they are consumed by different
runtimes, not because they are allowed to differ.

## The files here

| File | What it is |
|---|---|
| `qwen3-coder-30b.toml` | The daemon-side profile: delimiters, tool-result wrapping, the ChatML template markers, and the `[limits]` block (native `context_window`, `rlm_max_context`). |
| `qwen3-coder-30b.jde` | The language-side profile: `find_tool_call` / `find_tool_calls` over a reply. |
| `qwen3-coder-30b.expert-order.json` | Not a profile — the per-layer MoE expert ranking, measured offline, loaded by `dovata::moe` and handed to the C driver. Lives here because it is model-specific data. |

## Who reads them

The **daemon** compiles the TOMLs in via `dovata::profile::Registry::builtin()` and uses
the selected one to render prompts (`dovata::prompt`) and to parse the token stream
(`dovata::profile::stream`). A **Jade program** installs the `.jde` package to parse tool
calls out of a reply it received over the socket. The **publish-providers workflow**
(`.github/workflows/publish-providers.yml`) builds the `.jde` to a native `.so` per
platform and serves it from jadelang.org.

## Packaging

- One TOML file per model, named for the model (`qwen3-coder-30b.toml`).
- Files in this directory are compiled into the daemon's `dovata::profile` module
  (`Registry::builtin()`), so the daemon ships with them.
- The daemon can also load external profiles at runtime and `Registry::push`
  them to override a builtin (e.g. ship a profile next to a custom model).
- Selection happens at boot in `dovata::preflight`, from the GGUF header, before the
  model loads. `general.name` is matched against each profile's `match` globs first
  (first match wins; a pushed profile beats a builtin); `general.architecture` is then
  matched against each profile's `arch` list as a family fallback. Matching neither is a
  refused boot unless `JADE_ALLOW_UNPROFILED=1`.

## Schema

See `../src/profile/mod.rs` for the typed schema. Required: `schema_version`, `model`,
`[tool_call]` (`open`, `close`, `name_field`), `[tool_result]` (`role`,
optional `wrap_prefix`/`wrap_suffix`). Optional: `match` globs and `[[span]]`
entries (`tag`, `open`, `close`) for extra special-token spans.

## Adding a model

1. Generate the skeleton from the model file itself, which fills in `model`, `match`,
   and `[limits] context_window` and appends the model's chat template as a comment:

   ```sh
   cargo run -p gguf --bin gguf-inspect -- new-model.gguf --profile-stub > new-model.toml
   ```

2. Fill in the tool-call and tool-result tokens. These cannot be read out of a GGUF —
   they are a training convention — so the chat template in the comments is the source.
3. Set `arch` to the file's `general.architecture` so later models in the same family
   fall back here instead of onto bare ChatML.
4. Add it to `BUILTIN_PROFILES` in `../src/profile/mod.rs` if it should ship by default.
5. `cargo test -p dovata` validates every bundled profile.
