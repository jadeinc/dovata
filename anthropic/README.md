# anthropic — Claude provider (Jade library)

A provider conforming to **ovata-infer-protocol**, written in Jade. `infer()` consumes
an ovata `InferenceRequest` and returns the response as protocol `Frame`s — it calls
the Messages API directly over `std::http`. The API knobs the wire protocol doesn't
carry (temperature, tools, system, …) live in a separate `params` value that
**complements** the request; nothing is added to the protocol.

## Install it

Published as a native `.so` package at `jadelang.org`; install with the package
manager (downloads, checksums, pins in `jade.lock`):

```sh
jade pkg add anthropic --url 'https://www.jadelang.org/anthropic-{platform}.so' --version 0.1.0
```

```jade
use anthropic

let request = { "prompt": "What time is it in Tokyo?", "model": "claude-sonnet-5", "max_tokens": 128 }
let p = anthropic.params({ "temperature": 0.2,
          "tools": [ anthropic.tool("get_time", "Current time in a city",
                       { "type": "object", "properties": { "city": { "type": "string" } } }) ],
          "tool_choice": anthropic.choice_auto() })

let frames = anthropic.infer(request, p)   // -> [ Token, Json(tool_calls)?, Done ] or [ Error ]
```

## Use from source (dev)

Register the directory as a library in your project's `jade.toml`:

```toml
[lib.anthropic]
path  = "path/to/dovata/anthropic"
files = ["anthropic.jde"]
```

## API key

`api_key()` resolves the key in this order:

1. **`ANTHROPIC_API_KEY`** in the environment (wins if set).
2. the `"api_key"` field of **`~/.jade/providers/anthropic.json`** — the config file
   written when you install the provider. Configure it with:

   ```sh
   scripts/configure-provider.sh anthropic          # prompts for the key (mode 0600)
   ```

Point **`ANTHROPIC_BASE_URL`** at a gateway/proxy to override the endpoint (also how
the mock-server tests run). A missing key surfaces as an `Error` frame, not a raise.

## Shipping as a native `.so`

`jade build anthropic.jde --lib -o anthropic.so` compiles this library to a native
package the Jade runtime loads without the source; the publish-providers workflow
builds it per platform and serves it at jadelang.org.

## Surface

**Inference (the wire-protocol interface)**
- `infer(request, params) -> [Frame]` — `request` is an ovata `InferenceRequest`
  (`prompt`, `model`, `max_tokens`, `grammar`, …); `params` is the complement (below).
  Returns `Token`(s) then `Done` on success, or a single `Error`; tool calls ride in a
  `Json` frame. The **host** emits the leading `Meta` frame, so this provider doesn't.
  `rlm` / `grammar` requests (protocol fields a remote API can't honor) return `Error`.

**Frame values** (mirror `ovata-infer-protocol` `response.rs`)
- `{ "type": "Token", "text": str }`
- `{ "type": "Json",  "json": str }` — `{ tool_calls: [{ id, name, input }], stop_reason }`
- `{ "type": "Done",  "tokens_used": int }`
- `{ "type": "Error", "message": str }`

**Params — the complement** (`params({...})`; only keys you set are sent)
`temperature` (validated 0–1), `top_p`, `top_k`, `stop_sequences`, `system`, `tools`,
`tool_choice`, `thinking`, `metadata`.

**Builders**
- `tool(name, description, schema)` — a tool definition.
- `choice_auto()` / `choice_any()` / `choice_tool(name)` — `tool_choice` values.
- `thinking(budget_tokens)` — extended-thinking config.
