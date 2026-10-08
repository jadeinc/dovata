# anthropic — Claude model-profile (Jade package)

A model-profile package conforming to **ovata-infer-protocol**, written in Jade. The
inference call is protocol-pure — `infer(request)` takes an ovata `InferenceRequest`
and returns protocol `Frame`s. The fixed per-request parameters Claude needs but the
protocol doesn't carry (`api_key`, `tools`, `temperature`, `system`, …) are held as the
profile's **config**, set at runtime via `configure()`; body-building and auth are
internal. The config is runtime-**mutable** — change `tools`/params between requests and
the next `infer()` uses them.

## Install it

Published as a native `.so` package at `jadelang.org`; install with the package
manager (downloads, checksums, pins in `jade.lock`):

```sh
jade pkg add anthropic --url 'https://www.jadelang.org/anthropic-{platform}.so' --version 0.1.0
```

```jade
use anthropic

anthropic.configure({ "temperature": 0.2, "system": "Be terse.",
    "tools": [ anthropic.tool("get_time", "Current time in a city",
                 { "type": "object", "properties": { "city": { "type": "string" } } }) ],
    "tool_choice": anthropic.choice_auto() })

let frames = anthropic.infer({ "prompt": "What time is it in Tokyo?" })
// -> [ Token, Json(tool_calls)?, Done ]   or   [ Error ]

anthropic.set_tools([])                       // mutate at runtime; next infer() reflects it
let more = anthropic.infer({ "prompt": "Thanks!" })
```

## Use from source (dev)

```toml
[lib.anthropic]
path  = "path/to/dovata/src/anthropic"
files = ["anthropic.jde"]
```

## API key

`api_key()` resolves in order: `configure({api_key})`, then **`ANTHROPIC_API_KEY`**,
then the `"api_key"` field of **`~/.jade/providers/anthropic.json`** (written by
`tools/scripts/configure-provider.py anthropic`, mode 0600). A missing key surfaces as an
`Error` frame, not a raise. Set **`ANTHROPIC_BASE_URL`** (or `configure({base_url})`) to
point at a gateway/proxy.

## Shipping as a native `.so`

`jade build anthropic.jde --lib -o anthropic.so` compiles this to a native package the
Jade runtime loads without the source; the publish-providers workflow builds it per
platform and serves it at jadelang.org.

Build it with the jade it will run under. The runtime refuses a package whose value
ABI is not exactly its own: it went 1 to 2 in v1.1.31, when structs began crossing the
native boundary, and reached 5 by v1.3.10. It is not a source problem, so no edit here
fixes it. The
workflow pin is the floor, and `tools/scripts/check-package-abi.py` reads the ABI back out
of each built `.so` and fails the publish if it is short, so the pin cannot drift behind
the runtime unnoticed again.

## Surface

**Configure (the profile's fixed params — runtime-mutable)**
- `configure(opts)` — set any subset; validates (e.g. `temperature` 0–1). Keys: `api_key`,
  `base_url`, `model`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`,
  `system`, `tools`, `tool_choice`, `thinking`, `metadata`.
- `set_tools(tools)` — convenience for the common runtime mutation.
- Builders: `tool(name, description, schema)`, `choice_auto()` / `choice_any()` /
  `choice_tool(name)`, `thinking(budget_tokens)`.

**Infer (the protocol-pure interface)**
- `infer(request) -> [Frame]` — `request` is an ovata `InferenceRequest` (`prompt`,
  `model`, `max_tokens`, `grammar`, …). Returns `Token`(s) then `Done`, tool calls in a
  `Json` frame, or a single `Error`. The **host** emits `Meta`. `rlm`/`grammar` requests
  return `Error`.

**Frame values** (mirror `ovata-infer-protocol` `response.rs`)
- `{ "type": "Token", "text": str }` · `{ "type": "Done", "tokens_used": int }`
- `{ "type": "Json",  "json": str }` (`{ tool_calls: [{ id, name, input }], stop_reason }`)
- `{ "type": "Error", "message": str }`
