# Blended RAG-RLM: KV-Composition Engine — Design & Results

> Status: **prototype, validated end-to-end on EC2 (A10G), uncommitted** on branch
> `feat/rlm-phase0-fork-vendor`. All selftests are env-gated and live in
> `jaded/src/inference/compose.rs`. This document is the results snapshot to work from;
> the codebase needs a cleanup pass (see [Cleanup before implementation](#cleanup-before-implementation))
> before this becomes a production path.

All numbers below were measured on the EC2 `g5.xlarge` (NVIDIA A10G, 23 GB), serving the
deployed 30B-MoE Q4 model (~18.5 GB GGUF), `JADE_CTX_SIZE=32768`, `q4_0` K/V cache,
`JADE_KV_UNIFIED=1`. They are this-model/this-GPU figures, not universal.

---

## 1. What the engine is

The RLM path is no longer agentic (PICK→fan-out→COMBINE as N+2 prompted LLM calls — that
engine was removed). It is now **tensor-level KV composition**: the full corpus lives in
CPU RAM as token blocks; per query, a bounded set of blocks is selected, their precomputed
K/V is spliced into one attention window, and the answer comes from a **single decode**.
Recursion/decomposition lives in the KV/attention, not in a prompt loop.

Three building blocks:

1. **KV composition** (`compose_and_answer` / `compose_selected`)
   - Prefill a small prefix on seq 0; for each selected block, materialize its KV at the
     next offset via `materialize_block` (`kv_cache_seq_add` to relocate RoPE-correctly +
     same-stream `copy_kv_cache_seq` to splice into seq 0 — cheap under a **unified KV
     cache**, `kv_unified=true`); prefill the question; one decode.
   - Blocks computed in isolation; cross-block attention happens only at decode over seq 0.

2. **Host-RAM KV block cache** (`KvBlockCache`) — byte-bounded LRU keyed by block index;
   value = the block's serialized KV (`state_seq_get_data`, ~27 KB/token). Restoring is
   ~50× cheaper than re-prefilling (see §2). `cap_bytes == 0` disables caching. Per-handle
   cache cap via `JADE_KV_CACHE_MB` (default 2048). Persisted per uploaded context
   (`StoredContext` in `rlm.rs`), so repeat queries on a handle restore instead of
   re-prefilling.

3. **Blended RAG-RLM pipeline** (`blended_compose_and_answer`) — the PICK→FAN-OUT→COMBINE
   shape, but every phase a block/tensor mechanic, **zero agentic LLM calls**:
   - **PICK** = `retrieve` N candidates by lexical (IDF) similarity. Recall-oriented.
   - **FAN-OUT** = `rerank` — a cross-encoder relevance score per candidate =
     **query-likelihood `P(query | block)`**, the mean log-prob the model assigns to the
     query tokens with the block resident, read from `ctx.get_logits_ith` (no generation).
   - **COMBINE** = `compose_selected` over the top-k reranked blocks → single decode.
   - **Synergy:** rerank restores each candidate's block KV from the same cache COMBINE
     uses (the canonical `0..len` KV, saved *before* the compose offset-shift), so the two
     phases share work.

### Module map (`jaded/src/inference/`)
- `compose.rs` — the KV-composition engine, KV cache, blended pipeline, **and all the
  env-gated benches/selftests** (the part that needs splitting out).
- `rlm.rs` — socket dispatch (`handle_rlm`), the persistent handle store
  (`StoredContext` = `BlockStore` + `KvBlockCache`), and the batching-engine selftest.
- `concurrency.rs` + `sched.rs` — continuous-batching engine + pure scheduler. **Retained**
  as general infra for concurrent / mass prompting; currently only exercised by
  `selftest_rlm` (`JADE_RLM_SELFTEST`). Not on the compose path.
- `kv.rs` — KV helpers (cache-type map, multi-turn prefix reuse, shared-prefix prefill).
- `standard.rs` — single-shot inference (the production non-RLM path).

---

## 2. KV restore vs GPU re-prefill (`JADE_KV_BENCH`)

How expensive it is to materialize a block's KV two ways: recompute it on the GPU
(re-prefill) vs copy it from a host-RAM buffer (restore, the Phase-A `state_seq_set_data`).

| block tokens | re-prefill (GPU) | restore (PCIe) | speedup | save @ ingest | KB/tok |
|---:|---:|---:|---:|---:|---:|
| 64   | 51.5 ms  | 1.09 ms | 47× | 24.7 ms | 27.0 |
| 128  | 86.9 ms  | 1.33 ms | 65× | 40.8 ms | 27.0 |
| 256  | 105.3 ms | 2.05 ms | 51× | 53.3 ms | 27.0 |
| 512  | 147.6 ms | 3.03 ms | 49× | 75.7 ms | 27.0 |
| 1024 | 277.9 ms | 4.97 ms | 56× | 76.1 ms | 27.0 |
| 2048 | 547.4 ms | 9.09 ms | 60× | 96.2 ms | 27.0 |

- Restore is **~50–60× cheaper** than re-prefill across all sizes.
- Restore is **overhead-bound, not bandwidth-bound** (2048 tok = 54 MB in 9 ms ≈ 6.2 GB/s,
  well under PCIe Gen4 x16 ~25 GB/s) → the win grows with block size.
- KV footprint is **~27 KB/token** (6750× the token-id footprint) — caching block KV
  trades host RAM for compute. The right policy is an LRU of *hot* blocks, re-prefill cold.

---

## 3. Generation (decode) throughput baseline (`JADE_GEN_BENCH`)

Pure serial decode (prefill untimed, EOG ignored), 256 tokens, by **resident context size**:

| resident context | decode tok/s |
|---:|---:|
| 128   | 131.6 |
| 1,024 | 125.0 |
| 4,096 | 109.0 |
| 8,192 | 92.9 |

- Baseline **~130 tok/s** at short context, **~30% slower (~93)** by 8k resident (attention
  cost grows with KV size).
- This is the metric a user *feels*. The KV cache does **not** change it — caching only
  speeds prefill/TTFT. Faster generation needs a different lever (speculative decoding /
  draft model / shorter outputs), not composition.

---

## 4. Generation throughput PAST the 32k window (`JADE_RLM_OVERWINDOW`)

Corpus far larger than the window, held in RAM as blocks; a fixed `k=4` composed into a
bounded window; pure decode timed (256 tokens).

| corpus tokens | × window | blocks in RAM | GPU-resident | decode tok/s |
|---:|:--:|---:|---:|---:|
| 65,802  | 2× | 107 | 2,485 | 115.4 |
| 131,714 | 4× | 214 | 2,487 | 117.6 |
| 264,154 | 8× | 429 | 2,488 | 117.4 |

- **Decode tok/s is flat (~117) regardless of corpus size.** Generation past the window is
  decoupled from corpus size: it's set by the **resident composed window** (~2.5k here),
  not the corpus (which is RAM-bound; only k blocks touch the GPU).
- The knob is composed-window size (`k` × block size); decode slows along the §3 curve as
  the window grows. A standard model cannot run *any* of these (corpus > window).

---

## 5. KV-cache cold-vs-warm compose (`JADE_COMPOSE_CACHE`)

Same query twice over one block store + cache; warmup pass first to isolate CUDA graph
compile. 40 blocks, 1 block retrieved (~278 tok), 96-token generation.

- **COLD (re-prefill) 1561 ms / WARM (restore) 1452 ms** → ~109 ms of **prefill** saved per
  composed block.
- **`cold == warm` answers byte-identical** — restoring block KV reproduces the re-prefill
  composition exactly (greedy). This is the correctness guarantee for the cache path.
- Total-query speedup was small (1.08×) here because the **96-token decode dominates** and
  is unaffected by caching, and only 1 block was composed. Caching is a **prefill/TTFT**
  optimization; its payoff scales with `#composed blocks × per-block prefill` and shrinks
  against long generations.

---

## 6. Blended rerank — needle-in-haystack rescue (`JADE_BLENDED`)

Haystack with a needle block (`XQ-42`) plus **lexical distractor** blocks that echo the
query words ("secret access code") but contain no code.

**Rerank scores** (query-likelihood, higher = more relevant; well-separated, correct):

| block | score | |
|---:|---:|--|
| 20 | **-0.3591** | ← NEEDLE |
| 13 | -0.5024 | distractor |
| 30 | -0.6306 | distractor |
| 10 | -0.6492 | distractor |

**k = 1 (the decisive test):**

| pipeline | selected | needle found |
|--|--|--|
| PLAIN (lexical retrieval) | block `[10]` (a distractor) | ❌ false |
| BLENDED (rerank) | block `[20]` (the needle) | ✅ true |

Lexical recall was lured by a distractor and **missed**; the cross-encoder rerank scored
the true needle highest and **rescued** it. This is the needle-in-haystack quality win,
achieved with RAG mechanics — no agentic calls.

> Caveat: this is **one hand-built toy case** — it demonstrates the *mechanism*, not a
> benchmark accuracy number. A real figure needs a proper multi-hop / needle dataset
> (see [Next steps](#next-steps)).

---

## Reproduce (env-gated selftests)

Run on the box; all need `JADE_MODEL_PATH` set and (for compose paths) `JADE_KV_UNIFIED=1`:

```bash
# build (Linux/EC2 only)
cargo build --release -p jaded --features cuda

# run a selftest (it runs, prints to stderr, and exits)
LD_LIBRARY_PATH=/usr/local/cuda/lib64 JADE_MODEL_PATH=/jade/inference/model.gguf \
JADE_KV_UNIFIED=1 <FLAG>=1 ./target/release/jaded
```

| Flag | What it measures |
|--|--|
| `JADE_KV_BENCH` | re-prefill vs CPU-RAM KV restore (§2) |
| `JADE_GEN_BENCH` (+ `JADE_GEN_TOKENS`) | decode tok/s vs resident context (§3) |
| `JADE_RLM_OVERWINDOW` (+ `JADE_OW_BLOCKS`) | decode tok/s past the window (§4) |
| `JADE_COMPOSE_CACHE` | cold-vs-warm compose + correctness (§5) |
| `JADE_BLENDED` (+ `JADE_BLEND_N`, `JADE_BLEND_K`) | blended rerank rescue (§6) |
| `JADE_KV_ROUNDTRIP`, `JADE_KV_COMPOSE`, `JADE_COMPOSE_DEMO`, `JADE_COMPOSE_MULTI` | earlier composition spikes |
| `JADE_RLM_SELFTEST` | continuous-batching engine selftest |

Runtime knobs: `JADE_KV_CACHE_MB` (per-handle cache cap, default 2048),
`JADE_RLM_DEMO_SECTIONS` (haystack size).

---

## Caveats / not-yet-done

- **Toy evals, not benchmarks.** §5/§6 demonstrate mechanism and correctness, not accuracy
  numbers. Latency comparisons are confounded by first-call CUDA graph compilation.
- **Cross-block reasoning fidelity (CacheBlend) is deferred.** `recompute_frac` exists but
  the naive positional version didn't help; proper HKVD selection needs per-layer KV
  access (fork/ggml work). Callers pass `0.0`. Independent/RAG-style blocks work fine;
  multi-hop reasoning across composed blocks is unproven.
- **Retrieval is lexical IDF.** A semantic embedder for PICK is the obvious upgrade and
  matters most for recall quality.
- **Blended pipeline is selftest-only.** Not wired into the socket (`handle_rlm`) yet.
- **Decode output is rambly** (greedy, no stop conditions / chat-template framing) — a
  decode-polish item, not a correctness issue.
- **Rerank cost = N forwards** per query (cheap when the cache is warm; N prefills cold).

---

## Cleanup before implementation

The engine file has accumulated prototype scaffolding that should be tidied before this
becomes a real feature:

1. **Split benches/selftests out of `compose.rs`.** There are 8 env-gated selftests
   (`selftest_kv_roundtrip`, `selftest_kv_compose`, `selftest_compose_demo`,
   `selftest_compose_multi`, `selftest_kv_bench`, `selftest_compose_cache`,
   `selftest_rlm_overwindow`, `selftest_blended`) + their dispatch block in
   `inference.rs::run`. Move them to a dedicated bench module (or behind a `bench` cfg) so
   the engine file is just the engine.
2. **Prune dead/experimental knobs.** `recompute_frac` (CacheBlend) is effectively dead
   (always `0.0`) — either revive it properly or remove it. The `Block.text` field is
   unused (dead-code warning).
3. **Decide the engine surface.** Which is the production path — plain compose or blended?
   Wire the chosen one into `handle_rlm` (a `"blended": true` flag and/or default) with
   `N`/`k` defaults; drop the rest to selftests.
4. **Revisit the retained batching engine.** `concurrency.rs` + `sched.rs` are kept for
   mass/concurrent prompting but have no production caller yet — either wire them in or
   keep them clearly marked as infra (currently kept alive only by `selftest_rlm`).
5. **Naming.** The module is called `rlm` but the path is no longer recursive — consider
   renaming to reflect "retrieval + KV composition" to avoid confusion.

---

## Next steps (post-cleanup)

1. **Wire the blended pipeline into the socket** behind a request flag (or default), with
   `N`/`k` knobs.
2. **Real accuracy eval** — a proper needle / multi-hop QA dataset, measuring
   blended vs plain vs full-context-stuffing on this model, to turn "the mechanism works"
   into a measured "+X%."
3. **Semantic embedder for PICK** (recall quality).
4. **CacheBlend (cross-block fidelity)** — the real HKVD version, for multi-hop.
5. **Decode polish** — stop conditions / framing so answers aren't rambly.
6. (Optional) **yes/no relevance-logit reranker** as an alternative to query-likelihood;
   batch the rerank scoring across candidates to cut its cost.
