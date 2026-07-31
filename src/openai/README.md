# openai — OpenAI model-profile (Jade package)

A model-profile package conforming to **ovata-infer-protocol**, written in Jade. The
inference call is protocol-pure — `infer(request)` takes an ovata `InferenceRequest`
and returns protocol `Frame`s. The fixed per-request parameters OpenAI needs but the
protocol doesn't carry (`api_key`, `tools`, `temperature`, `system`, …) are held as the
profile's **config**, set at runtime via `configure()`; body-building and auth are
internal. The config is runtime-**mutable** — change `tools`/params between requests and
the next `infer()` uses them.

## Install it

Published as a native `.so` package at `jadelang.org`; install with the package
manager (downloads, checksums, pins in `jade.lock`):

```sh
jade pkg add openai --url 'https://www.jadelang.org/openai-{platform}.so' --version 0.1.0
```

```jade
use openai

openai.configure({ "temperature": 0.2, "system": "Be terse.",
    "tools": [ openai.tool("get_time", "Current time in a city",
                 { "type": "object", "properties": { "city": { "type": "string" } } }) ],
    "tool_choice": openai.choice_auto() })

let frames = openai.infer({ "prompt": "What time is it in Tokyo?" })
// -> [ Token?, Json(tool_calls)?, Done ]   or   [ Error ]

openai.set_tools([])                          // mutate at runtime; next infer() reflects it
let more = openai.infer({ "prompt": "Thanks!" })
```

## Use from source (dev)

```toml
[lib.openai]
path  = "path/to/dovata/src/openai"
files = ["openai.jde"]
```

## API key

`api_key()` resolves in order: `configure({api_key})`, then **`OPENAI_API_KEY`**, then
the `"api_key"` field of **`~/.jade/providers/openai.json`** (written by
`src/scripts/configure-provider.py openai`, mode 0600). A missing key surfaces as an `Error`
frame, not a raise. Set **`OPENAI_BASE_URL`** (or `configure({base_url})`) to point at a
gateway/proxy or OpenAI-compatible host.

## Shipping as a native `.so`

`jade build openai.jde --lib -o openai.so` compiles this to a native package the Jade
runtime loads without the source; the publish-providers workflow builds it per platform
and serves it at jadelang.org.

Build it with jade v1.1.32 or later. The runtime's value ABI went 1 to 2 in v1.1.31,
when structs began crossing the native boundary, and a package carrying the old ABI is
refused at load by v1.1.32 — it is not a source problem, so no edit here fixes it. The
workflow pin is the floor, and `src/scripts/check-package-abi.py` reads the ABI back out
of each built `.so` and fails the publish if it is short, so the pin cannot drift behind
the runtime unnoticed again.

## Surface

**Configure (the profile's fixed params — runtime-mutable)**
- `configure(opts)` — set any subset; validates (e.g. `temperature` 0–2). Keys: `api_key`,
  `base_url`, `model`, `max_tokens`, `temperature`, `top_p`, `frequency_penalty`,
  `presence_penalty`, `stop`, `seed`, `n`, `system`, `tools`, `tool_choice`,
  `response_format`, `metadata`.
- `set_tools(tools)` — convenience for the common runtime mutation.
- Builders: `tool(name, description, schema)`, `choice_auto()` / `choice_none()` /
  `choice_required()` / `choice_tool(name)`, `json_object()` / `json_schema(name, schema)`.

**Infer (the protocol-pure interface)**
- `infer(request) -> [Frame]` — `request` is an ovata `InferenceRequest` (`prompt`,
  `model`, `max_tokens`, `grammar`, …). Returns `Token`(s) then `Done`, tool calls in a
  `Json` frame, or a single `Error`. The **host** emits `Meta`. `rlm`/`grammar`/`count_only`
  requests return `Error`.

**Frame values** (mirror `ovata-infer-protocol` `response.rs`)
- `{ "type": "Token", "text": str }` · `{ "type": "Done", "tokens_used": int }`
- `{ "type": "Json",  "json": str }` (`{ tool_calls: [{ id, name, arguments }], finish_reason }`; `arguments` is a JSON **string**)
- `{ "type": "Error", "message": str }`
