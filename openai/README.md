# openai — OpenAI provider (Jade library)

A remote API like OpenAI is just HTTP, so this provider is a **pure-Jade library**,
not a daemon `.so`. It builds the Chat Completions request, calls the API directly
over `std::http`, and parses the reply — no Rust, no dlopen, no daemon involvement.
(The daemon and the wire protocol are for *local* llama inference only.)

## Use it

Register the directory as a library in your project's `jade.toml`, then import:

```toml
[lib.openai]
path  = "path/to/dovata/openai"
files = ["openai.jde"]
```

```jade
use openai::openai

let reply = openai.complete(
    [{ "role": "user", "content": "What time is it in Tokyo?" }],
    { "temperature": 0.2 })
print(openai.text(reply))
```

## API key

`api_key()` resolves the key in this order:

1. **`OPENAI_API_KEY`** in the environment (wins if set).
2. the `"api_key"` field of **`~/.jade/providers/openai.json`** — the config file
   written when you install the provider. Configure it with:

   ```sh
   scripts/configure-provider.sh openai             # prompts for the key (mode 0600)
   ```

Point **`OPENAI_BASE_URL`** at a gateway/proxy or an OpenAI-compatible host to
override the endpoint (also how the mock-server tests run).

## Shipping as a native `.so`

`jade build openai.jde --lib -o openai.so` compiles this library to a native package
(`jade_pkg_init` ABI) that the Jade runtime loads without the source. The `packaging`
xtask does this and drops `openai.so` into the release bundle (`dist/`, listed under
`jade_providers` in the manifest); register that `.so` as the `[lib]` file in
production instead of the `.jde`.

## Surface

**Call**
- `complete(messages, opts) -> reply` — the Chat Completions call; returns the parsed response dict. `messages` is `[{ role, content }, ...]`; pass `{}` for no options.
- `ask(content, opts) -> str` — one-shot sugar: `complete` + `text`.

**opts** (only keys you set are sent; the rest take the API default)
`model`, `max_tokens`, `system` (prepended as a system message), `temperature`
(validated 0–2), `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed`,
`n`, `tools`, `tool_choice`, `response_format`, `metadata`.

**Tool / format helpers**
- `tool(name, description, schema)` — a function-tool definition.
- `choice_auto()` / `choice_none()` / `choice_required()` / `choice_tool(name)` — `tool_choice` values.
- `json_object()` / `json_schema(name, schema)` — `response_format` values.

**Parse the reply**
- `text(reply) -> str` — assistant text (`""` when the turn is only tool calls).
- `tool_calls(reply) -> [{ id, name, args }]` / `tool_call(reply) -> {…} | nil`.
  `args` is the raw JSON **string** arguments — `json.parse` it to get a dict.
- `finish_reason(reply) -> str` — `stop` / `length` / `tool_calls` / `content_filter`.
- `usage(reply) -> { prompt_tokens, completion_tokens, total_tokens }`.
