# tests/ — the dovata inference eval suite

End-to-end evals for the `dovata` inference engine, written in Jade, plus the harness
that runs them. Each program exercises one engine capability through the real runtime: a
Jade program loads `dovata.so` as its provider and every `?p` dereference is served
in-process.

There is no Rust here. `cargo test` never looks in this directory — the pure-core unit
tests live next to the code they cover, under `src/dovata/src/`. This is the other kind
of testing: the engine as a black box, on a real GPU, driven by a real client.

These are **integration/correctness** evals — they assert the engine's *mechanics*
(grammar validity, type coercion, anchor handling, multi-turn context), not the
model's factual accuracy. Sampling is non-greedy, so phrase any value checks
loosely; structural checks (a grammar matched, JSON parsed, a needle found) are
deterministic by construction.

## Why they are written in Jade

The engine's own `cargo test` covers the pure core thoroughly, but it can only ever
test the engine against itself. These evals come in through the front door: a real
Jade runtime, the real provider ABI, a real model. If the engine and the language
disagree about what a frame means, or a profile's delimiters are wrong, or the grammar
never reaches the sampler, this is where it shows up and nothing else will catch it.

That is also why they live in the engine repo rather than the language repo — they are
the engine's acceptance criteria, expressed in the client's language.

## Who runs them

`run-tests.py`, right here, is the only harness. It builds the engine `.so`, stages it
into a scratch provider slot, runs each `.jde` beside it, and greps the output. The evals
depend on a built engine, a model on disk, and a current Jade toolchain — so they run on
the GPU box, not on your laptop.

The harness used to live in `tools/ec2/`. It moved here because it is the runner for this
suite rather than a piece of box plumbing: it takes no AWS credentials, resolves no
instance, and finds the repo from its own path. Getting *to* the box is
[`tools/ec2/`](../tools/ec2/README.md)'s job; this script assumes it is already there.

### There is no daemon to start

`run-tests.py` used to build a `dovata` binary, spawn it, wait for a
`dovata: listening on` line, and let the evals reach it over `~/.jade/llm.sock`. Dropping
the daemon deleted all of that, and the harness sat broken until it was ported: it died
at its own `dovata not found` check, because `cargo build -p dovata` emits `libdovata.so`
and no binary at all.

What replaces the daemon is a staging step. The Jade runtime loads exactly one provider
library from the active slot, and `JADE_PROVIDER_ACTIVE` overrides that slot's path
outright — the hook jadelang provides for dev and testing. So the harness copies the
freshly built engine into `target/provider-active/` and points the evals there. The slot
is rebuilt each run, since a leftover `.so` from an earlier build would be loaded ahead of
the one just compiled.

`JADE_PROVIDERS_DIR` is *not* the hook. That one feeds `jade register`'s discovery of
installable providers; the runtime never reads it.

The evals themselves needed no changes through any of it. They speak `use llm`, `prompt`,
and `?p` — language surface, not transport.

## Convention

Each `.jde` prints exactly one line `PASS: <description>` when its assertions hold.
The harness greps for `^PASS:`; any other outcome is a FAIL and the program's last
output line is shown as the diagnostic.

Each program is self-contained: `ok` / `detail` locals, the work inside `try`, one
`PASS:`/`FAIL:` line at the end. Follow that shape when adding one — the harness reads
nothing else, and a program that prints two PASS lines or none is a harness bug waiting
to happen.

## Coverage

| File                    | Engine path exercised                                             |
|-------------------------|-------------------------------------------------------------------|
| `01_smoke.jde`          | provider load + unconstrained generation                          |
| `02_primitives.jde`     | full-primitive grammars (int/bool/float coercion), no anchor       |
| `03_struct.jde`         | JSON object grammar (explicit GBNF) → parseable object             |
| `04_collections.jde`    | JSON array grammar (explicit GBNF) → non-empty parsed array        |
| `05_grammar.jde`        | explicit user GBNF via `Grammar.new`, enforced from token 1        |
| `06_multiturn.jde`      | `_ovata_infer_protocol: messages` history threading                |
| `07_tool_anchor.jde`    | anchored grammar: deferred enforcement, depth retire, stop-strip   |
| `08_latency.jde`        | coarse end-to-end latency / throughput probe                       |
| *(09 — removed)*        | *see below*                                                        |
| `10_special_tokens.jde` | chat-template markers in user content stay data, not structure     |
| `11_concurrency.jde`    | two derefs in flight against one in-process engine                 |
| `12_grammar_retire.jde` | grammar drops after the anchored span closes; free text follows    |
| `13_long_context.jde`   | a long prompt is prefilled whole, needle still retrievable         |
| `14_error_frame.jde`    | a failure surfaces as a catchable Jade error                       |
| `15_tool_result.jde`    | `[tool_result]` return leg: role + wrapper understood              |

`07` and `12` are the two halves of the `control.rs` state machine — one proves the
grammar switches on at the anchor, the other that it switches off at the span close.

`09` was a `max_tokens` test and is gone. It could not be written: the language has no
way to cap a reply, so there was nothing for an eval at this layer to assert. The number
is left as a gap rather than reused, so the removal stays visible instead of looking like
a renumbering. Whether the engine honours a token cap is a dovata-side question and
belongs in `cargo test -p dovata`.

`11` exists because of the daemon removal specifically. A socket serialised requests
whether or not the caller knew; an in-process library does not, and nothing else here
puts two requests in the engine at once.

## Notes

- **Imports use `::`**: `use std::json`, `use std::time`. Not `use std.json` (the dot
  form this suite was originally written against, correct for the jade v1.1.8 that used
  to be on the box) and not `use "std/json"` (quoted file imports were removed from the
  language).
- **`jade check` does not validate imports.** `use totally_made_up_module` passes it and
  then fails at run with "file not found". So checking a file proves it parses and
  nothing more. Before committing an eval, actually run it, or at minimum confirm each
  `use` resolves — the harness reports either failure as an ordinary FAIL, so a dead
  import looks like a failing assertion rather than a broken file.
- **There is no `llm` package, and no way to set `max_tokens`.** It was removed from the
  language; `InferenceRequest` in the jade repo records that `model`, `max_tokens`,
  `keep_anchors` and `trust` went with the daemon. A Jade program can express exactly
  four things — the prompt, a grammar, and the two anchors — so an eval cannot cap reply
  length. Ask for brevity in the prompt instead. Six evals called
  `llm.set_max_tokens(...)` and failed at load until they were rewritten.
- These assert engine *mechanics*, not the runtime's typed struct coercion
  (`?p |> SomeStruct`), which was flaky when the suite was written. `03`/`04` therefore
  use explicit JSON GBNF + `json.parse` instead of `|> Struct`.
- Several evals mix a `time.now_ms()` nonce into the prompt so a run measures real
  inference rather than replaying a cached response.
- `13_long_context.jde` stays a few thousand tokens deliberately. The profile advertises
  a 262144-token window, but that needs the MoE expert-offload driver enabled
  (`JADE_MOE_DRIVER=1`); without it the engine serves `JADE_CTX_SIZE`, default 32768.
  Keeping it small makes it a correctness test on any configuration. Sweeping the actual
  frontier is `JADE_FRONTIER`'s job.
- Baseline observed on a `g5.xlarge` (A10G) with the 18 GB model:
  **TTFT ≈ 57 ms, decode ≈ 165 tok/s** (coarse, via `08_latency.jde`, measured under the
  old daemon — worth re-taking now the engine is in-process).

## Running

On the EC2 GPU instance, with a model present:

```bash
python3 tests/run-tests.py           # engine already built
python3 tests/run-tests.py --build   # build the engine first
```

The harness builds `libdovata.so`, copies it into `target/provider-active/`, points
`JADE_PROVIDER_ACTIVE` there, runs every `*.jde` here, and reports pass/fail.
