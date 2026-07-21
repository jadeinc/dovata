# jaded inference eval suite

End-to-end evals for the `jaded` inference daemon, written in Jade. Each program
exercises one daemon capability through the real runtime: the Jade VM connects to
`~/.jade/llm.sock` and every `?p` dereference is served by `jaded`.

These are **integration/correctness** evals — they assert the daemon's *mechanics*
(grammar validity, type coercion, anchor handling, multi-turn context), not the
model's factual accuracy. Sampling is non-greedy, so phrase any value checks
loosely; structural checks (a struct decoded, a grammar matched, JSON parsed) are
deterministic by construction.

## Convention

Each `.jde` prints exactly one line `PASS: <description>` when its assertions hold.
The harness (`scripts/ec2/run-tests.sh`) greps for `^PASS:`; any other outcome is a
FAIL and the program's last output line is shown as the diagnostic.

## Coverage

| File                  | Daemon path exercised                                            |
|-----------------------|------------------------------------------------------------------|
| `01_smoke.jde`        | socket connect + unconstrained generation                        |
| `02_primitives.jde`   | full-primitive grammars (int/bool/float coercion), no anchor     |
| `03_struct.jde`       | JSON object grammar (explicit GBNF) → parseable object           |
| `04_collections.jde`  | JSON array grammar (explicit GBNF) → non-empty parsed array      |
| `05_grammar.jde`      | explicit user GBNF via `Grammar.new`, enforced from token 1      |
| `06_multiturn.jde`    | `_ovata_infer_protocol: messages` history threading (basis of KV reuse) |
| `07_tool_anchor.jde`  | anchored grammar: deferred enforcement, depth retire, stop-strip |
| `08_latency.jde`      | coarse end-to-end latency / throughput probe (Phase 0 bench)     |

`07` maps directly onto the `control.rs` state machine refactored in Phase 0
(anchor span + `</tool>` stop-anchor + JSON depth tracking).

## Notes

- **Imports use dot notation, no quotes** in this runtime (v1.1.8): `use llm`,
  `use std.json`, `use std.time` — *not* `use "std/json"`.
- These assert daemon *mechanics* (grammar validity, anchors, multi-turn), not the
  runtime's typed struct coercion (`?p |> SomeStruct`), which is currently flaky on
  the deployed daemon — the project's own `programs/lib/tests/test_gbnf.jde` fails
  on struct decode identically. `03`/`04` therefore use explicit JSON GBNF +
  `json.parse` instead of `|> Struct`.
- Baseline observed on a `g5.xlarge` (A10G) with the 18 GB model:
  **TTFT ≈ 57 ms, decode ≈ 165 tok/s** (coarse, via `08_latency.jde`).

## Running

On the EC2 GPU instance, with `jaded` built (`--features cuda`) and a model present:

```bash
bash scripts/ec2/run-tests.sh           # binaries already built
bash scripts/ec2/run-tests.sh --build   # build jaded + jade runtime first
```

The harness starts `jaded`, waits for `jaded: listening on …`, runs every
`*.jde` here, and reports pass/fail.
