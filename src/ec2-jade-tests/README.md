# dovata inference eval suite

End-to-end evals for the `dovata` inference engine, written in Jade. Each program
exercises one engine capability through the real runtime: a Jade program loads
`dovata.so` as its provider and every `?p` dereference is served in-process.

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

`src/ec2-tests/run-tests.py` is the only harness. It builds the engine `.so`, stages it
into a scratch provider slot, runs each `.jde` here, and greps the output. They depend on
a built engine, a model on disk, and a current Jade toolchain on the box — so they do not
run locally and are not part of any `cargo test`.

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
| `09_max_tokens.jde`     | `max_tokens` reaching the sampler's stop condition                 |
| `10_special_tokens.jde` | chat-template markers in user content stay data, not structure     |
| `11_concurrency.jde`    | two derefs in flight against one in-process engine                 |
| `12_grammar_retire.jde` | grammar drops after the anchored span closes; free text follows    |
| `13_long_context.jde`   | a long prompt is prefilled whole, needle still retrievable         |
| `14_error_frame.jde`    | a failure surfaces as a catchable Jade error                       |
| `15_tool_result.jde`    | `[tool_result]` return leg: role + wrapper understood              |

`07` and `12` are the two halves of the `control.rs` state machine — one proves the
grammar switches on at the anchor, the other that it switches off at the span close.

`11` exists because of the daemon removal specifically. A socket serialised requests
whether or not the caller knew; an in-process library does not, and nothing else here
puts two requests in the engine at once.

## Notes

- **Imports use `::`**: `use llm`, `use std::json`, `use std::time`. Not `use std.json`
  (the dot form this suite was originally written against, correct for the jade v1.1.8
  that used to be on the box) and not `use "std/json"` (quoted file imports were removed
  from the language). Run `jade check <file>` before committing an eval — the harness
  reports a parse error as an ordinary FAIL, so a syntax slip looks like a failing
  assertion rather than a broken file.
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
python3 src/ec2-tests/run-tests.py           # engine already built
python3 src/ec2-tests/run-tests.py --build   # build the engine first
```

The harness builds `libdovata.so`, copies it into `target/provider-active/`, points
`JADE_PROVIDER_ACTIVE` there, runs every `*.jde` here, and reports pass/fail.
