# openai — OpenAI provider (Jade library)

A provider conforming to **ovata-infer-protocol**, written in Jade. `infer()` consumes
an ovata `InferenceRequest` and returns the response as protocol `Frame`s — it calls
the Chat Completions API directly over `std::http`. The API knobs the wire protocol
doesn't carry (temperature, tools, system, …) live in a separate `params` value that
**complements** the request; nothing is added to the protocol.

## Install it

Published as a native `.so` package at `jadelang.org`; install with the package
manager (downloads, checksums, pins in `jade.lock`):

```sh
jade pkg add openai --url 'https://www.jadelang.org/openai-{platform}.so' --version 0.1.0
```

```jade
use openai

let request = { "prompt": "What time is it in Tokyo?", "model": "gpt-4o", "max_tokens": 128 }
let p = openai.params({ "temperature": 0.2,
          "tools": [ openai.tool("get_time", "Current time in a city",
                       { "type": "object", "properties": { "city": { "type": "string" } } }) ],
          "tool_choice": openai.choice_auto() })

let frames = openai.infer(request, p)   // -> [ Token?, Json(tool_calls)?, Done ] or [ Error ]
```

## Use from source (dev)

Register the directory as a library in your project's `jade.toml`:

```toml
[lib.openai]
path  = "path/to/dovata/openai"
files = ["openai.jde"]
```

## API key

`api_key()` resolves the key in this order:

1. **`OPENAI_API_KEY`** in the environment (wins if set).
2. the `"api_key"` field of **`~/.jade/providers/openai.json`** — the config file
   written when you install the provider. Configure it with:

   ```sh
   scripts/configure-provider.sh openai             # prompts for the key (mode 0600)
   ```

Point **`OPENAI_BASE_URL`** at a gateway/proxy or an OpenAI-compatible host to override
the endpoint. A missing key surfaces as an `Error` frame, not a raise.

## Shipping as a native `.so`

`jade build openai.jde --lib -o openai.so` compiles this library to a native package
the Jade runtime loads without the source; the publish-providers workflow builds it per
platform and serves it at jadelang.org.

## Surface

**Inference (the wire-protocol interface)**
- `infer(request, params) -> [Frame]` — `request` is an ovata `InferenceRequest`
  (`prompt`, `model`, `max_tokens`, `grammar`, …); `params` is the complement (below).
  Returns `Token`(s) then `Done` on success, or a single `Error`; tool calls ride in a
  `Json` frame. The **host** emits the leading `Meta` frame, so this provider doesn't.
  `rlm` / `grammar` / `count_only` requests (protocol fields OpenAI can't honor) return
  `Error`.

**Frame values** (mirror `ovata-infer-protocol` `response.rs`)
- `{ "type": "Token", "text": str }`
- `{ "type": "Json",  "json": str }` — `{ tool_calls: [{ id, name, arguments }], finish_reason }` (`arguments` is a JSON **string**)
- `{ "type": "Done",  "tokens_used": int }`
- `{ "type": "Error", "message": str }`

**Params — the complement** (`params({...})`; only keys you set are sent)
`temperature` (validated 0–2), `top_p`, `frequency_penalty`, `presence_penalty`,
`stop`, `seed`, `n`, `system`, `tools`, `tool_choice`, `response_format`, `metadata`.

**Builders**
- `tool(name, description, schema)` — a function-tool definition.
- `choice_auto()` / `choice_none()` / `choice_required()` / `choice_tool(name)` — `tool_choice` values.
- `json_object()` / `json_schema(name, schema)` — `response_format` values.
