# anthropic — Claude provider (Jade library)

A remote API like Claude is just HTTP, so this provider is a **pure-Jade library**,
not a daemon `.so`. It builds the Messages request, calls the API directly over
`std::http`, and parses the reply — no Rust, no dlopen, no daemon involvement. (The
daemon and the wire protocol are for *local* llama inference only.)

## Use it

Register the directory as a library in your project's `jade.toml`, then import:

```toml
[lib.anthropic]
path  = "path/to/dovata/anthropic"
files = ["anthropic.jde"]
```

```jade
use anthropic::anthropic

let reply = anthropic.complete(
    [{ "role": "user", "content": "What time is it in Tokyo?" }],
    { "temperature": 0.2 })
print(anthropic.text(reply))
```

The API key is read from **`ANTHROPIC_API_KEY`** in the calling program's
environment. Point **`ANTHROPIC_BASE_URL`** at a gateway/proxy to override the
endpoint (also how the mock-server tests run).

## Surface

**Call**
- `complete(messages, opts) -> reply` — the Messages call; returns the parsed response dict. `messages` is `[{ role, content }, ...]`; pass `{}` for no options.
- `ask(content, opts) -> str` — one-shot sugar: `complete` + `text`.

**opts** (only keys you set are sent; the rest take the API default)
`model`, `max_tokens`, `system`, `temperature` (validated 0–1), `top_p`, `top_k`,
`stop_sequences`, `tools`, `tool_choice`, `thinking`, `metadata`, `service_tier`.

**Tool helpers**
- `tool(name, description, schema)` — a tool definition.
- `choice_auto()` / `choice_any()` / `choice_tool(name)` — `tool_choice` values.
- `thinking(budget_tokens)` — extended-thinking config.

**Parse the reply**
- `text(reply) -> str` — concatenated text blocks.
- `tool_calls(reply) -> [{ id, name, args }]` / `tool_call(reply) -> {…} | nil`.
- `stop_reason(reply) -> str` — `end_turn` / `tool_use` / `max_tokens` / `stop_sequence`.
- `usage(reply) -> { input_tokens, output_tokens }`.
