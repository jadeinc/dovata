//! Developer self-tests and benchmarks — NOT part of any request path.
//!
//! Every routine here is gated by an env var: [`dispatch`] is called once at the end
//! of daemon start-up; if a matching `JADE_*` var is set it runs that one routine
//! (printing to stderr), then the daemon EXITS without binding the socket. With no
//! var set, [`dispatch`] returns `false` and the normal server path proceeds. This
//! keeps all demo/bench code in one module instead of scattered through the engines.
//!
//! The routines split into three groups:
//! * **primitives** — KV save/restore round-trip and the two-block compose spike, the
//!   building blocks the [`super::compose`] engine rests on.
//! * **composition** — needle demo, cold-vs-warm cache, the blended rerank pipeline.
//! * **throughput** — KV-restore-vs-prefill, raw decode tok/s, and decode-past-window;
//!   plus the continuous-batching ([`super::concurrency`]) fan-out self-test.

use super::build_formatted_prompt;
use super::compose::{
    blended_compose_and_answer, compose_and_answer, gen_loop, load_seq, materialize_block,
    prefill, save_seq, BlockStore, KvBlockCache,
};
use super::concurrency::{run_concurrent, SubInfer};
use crate::profile::ChatTemplate;
use ovata_infer_protocol::InferenceRequest;
use crate::control::DecodeControl;
use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatTemplate, LlamaModel},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use std::time::Instant;

/// Run the env-selected self-test/bench against the loaded model, if any is requested.
/// Returns `true` when a routine ran (the caller should then exit instead of serving),
/// `false` when no `JADE_*` self-test var is set. Each var is checked in turn; the
/// first match wins.
pub(crate) fn dispatch(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) -> bool {
    // (env var, routine) pairs. `selftest_rlm` exercises the continuous-batching engine;
    // the rest exercise the KV-composition engine (all need JADE_KV_UNIFIED=1, the default).
    let routines: &[(&str, fn(&LlamaModel, &mut LlamaContext, &LlamaChatTemplate))] = &[
        ("JADE_RLM_SELFTEST", selftest_rlm),
        ("JADE_KV_ROUNDTRIP", selftest_kv_roundtrip),
        ("JADE_KV_COMPOSE", selftest_kv_compose),
        ("JADE_COMPOSE_DEMO", selftest_compose_demo),
        ("JADE_COMPOSE_MULTI", selftest_compose_multi),
        ("JADE_KV_BENCH", selftest_kv_bench),
        ("JADE_COMPOSE_CACHE", selftest_compose_cache),
        ("JADE_GEN_BENCH", selftest_gen_bench),
        ("JADE_RLM_OVERWINDOW", selftest_rlm_overwindow),
        ("JADE_BLENDED", selftest_blended),
        ("JADE_FRONTIER", selftest_frontier),
    ];
    for (var, run) in routines {
        if std::env::var(var).is_ok() {
            run(model, ctx, tmpl);
            return true;
        }
    }
    false
}

/// Decode one token with logits enabled (used to seed generation after a prefill that
/// didn't request logits). Local to the bench module — the engine doesn't need it.
fn feed_token(ctx: &mut LlamaContext, tok: LlamaToken, pos: i32, seq: i32) -> Result<(), String> {
    let mut b = LlamaBatch::new(1, 1);
    b.add(tok, pos, &[seq], true).map_err(|e| format!("feed add: {e}"))?;
    ctx.decode(&mut b).map_err(|e| format!("feed decode: {e}"))
}

// ── Primitives ─────────────────────────────────────────────────────────────────

/// KV state save → restore round-trip (`JADE_KV_ROUNDTRIP`). Prefill a block, save it,
/// generate a reference continuation; then clear, restore the block, generate again —
/// the two continuations must be byte-identical (greedy) iff the KV round-tripped.
fn selftest_kv_roundtrip(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let prompt = build_formatted_prompt("In one sentence, what is the capital of France?", &ChatTemplate::default())?;
        let toks = model.str_to_token(&prompt, AddBos::Always).map_err(|e| format!("tok: {e}"))?;
        if toks.len() < 2 {
            return Err("prompt too short".into());
        }
        let n = toks.len();
        let block = &toks[..n - 1];
        let last = toks[n - 1];
        let last_pos = (n - 1) as i32;

        // Reference: prefill block on seq 0, save it, then continue.
        ctx.clear_kv_cache();
        prefill(ctx, block, 0, 0, false)?;
        let bytes = save_seq(ctx, 0);
        eprintln!("dovata: [kv-roundtrip] saved {} bytes for {} block tokens", bytes.len(), n - 1);
        feed_token(ctx, last, last_pos, 0)?;
        let ref_text = gen_loop(model, ctx, 0, last_pos + 1, 16)?;

        // Restore the block into a clean cache and continue identically.
        ctx.clear_kv_cache();
        let ok = load_seq(ctx, &bytes, 0);
        eprintln!("dovata: [kv-roundtrip] restore ok={ok}");
        feed_token(ctx, last, last_pos, 0)?;
        let test_text = gen_loop(model, ctx, 0, last_pos + 1, 16)?;

        eprintln!("dovata: [kv-roundtrip] ref  = {:?}", ref_text.trim());
        eprintln!("dovata: [kv-roundtrip] test = {:?}", test_text.trim());
        eprintln!("dovata: [kv-roundtrip] MATCH = {}", ref_text.trim() == test_text.trim());
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("dovata: [kv-roundtrip] error: {e}");
    }
}

/// Two-block compose spike (`JADE_KV_COMPOSE`). Compose two independently-computed
/// blocks into one sequence's attention window via existing ops (no `is_full` surgery),
/// then answer a question that needs BOTH. Requires a unified KV cache so the seq_cp is
/// a cheap same-stream metadata op. Success = the answer contains both blocks' facts.
fn selftest_kv_compose(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        // Two independent facts as raw blocks (no chat template) + a question.
        let a = model.str_to_token("Note A: the secret animal is a platypus.\n", AddBos::Always)
            .map_err(|e| format!("tok a: {e}"))?;
        let b = model.str_to_token("Note B: the secret number is 42.\n", AddBos::Never)
            .map_err(|e| format!("tok b: {e}"))?;
        let q = model.str_to_token(
            "Using the notes above, answer: what is the secret animal and the secret number? Answer: ",
            AddBos::Never,
        ).map_err(|e| format!("tok q: {e}"))?;
        let la = a.len() as i32;
        let lb = b.len() as i32;

        ctx.clear_kv_cache();
        // Block A: native on seq 0 at [0, la).
        prefill(ctx, &a, 0, 0, false)?;
        // Block B: computed IN ISOLATION on seq 1 at [0, lb).
        prefill(ctx, &b, 0, 1, false)?;
        // Relocate B to [la, la+lb) (RoPE-correct), then splice its cells into seq 0
        // (same-stream metadata copy under a unified cache).
        ctx.kv_cache_seq_add(1, Some(0), Some(lb as u32), la).map_err(|e| format!("seq_add: {e}"))?;
        ctx.copy_kv_cache_seq(1, 0, Some(la as u32), Some((la + lb) as u32)).map_err(|e| format!("seq_cp: {e}"))?;
        eprintln!("dovata: [kv-compose] composed block A[0..{la}) + block B[{la}..{}) into seq 0", la + lb);

        // Append the question on seq 0 after both blocks and decode.
        let qbase = la + lb;
        prefill(ctx, &q, qbase, 0, true)?;
        let ans = gen_loop(model, ctx, 0, qbase + q.len() as i32, 40)?;
        let lc = ans.to_lowercase();
        let has_animal = lc.contains("platypus");
        let has_number = lc.contains("42");
        eprintln!("dovata: [kv-compose] ANSWER: {:?}", ans.trim());
        eprintln!("dovata: [kv-compose] platypus={has_animal} 42={has_number} -> {}",
            if has_animal && has_number { "COMPOSE OK (both blocks visible)" } else { "INCOMPLETE" });
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("dovata: [kv-compose] error: {e}");
    }
}

// ── Composition ──────────────────────────────────────────────────────────────--

/// Build a synthetic haystack: `n` sections of routine filler with the needle hidden in
/// the middle one. Returns (texts, needle, needle_index). Shared by the demo/cache benches.
fn needle_haystack(needle: &str, n: usize) -> (Vec<String>, usize) {
    let topics = ["Introduction","Hardware","Boot","Networking","Filesystem","Processes","Logging","Power","Updates","Telemetry","Scheduling","Memory","Drivers","Crypto","Backups","Glossary"];
    let filler = "This section documents routine operational details with no special codes or secrets. ".repeat(20);
    let needle_at = n / 2;
    let texts: Vec<String> = (0..n).map(|i| {
        if i == needle_at {
            format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly. {filler}")
        } else {
            let t = topics[i % topics.len()];
            format!("Section {i}: {t}. This section describes {t}. {filler}")
        }
    }).collect();
    (texts, needle_at)
}

/// End-to-end composition demo (`JADE_COMPOSE_DEMO`). A needle-in-haystack answered by
/// retrieve → compose → single decode — no agentic calls, no re-prompting.
fn selftest_compose_demo(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let n: usize = std::env::var("JADE_RLM_DEMO_SECTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let (texts, _) = needle_haystack(needle, n);
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("dovata: [compose-demo] build error: {e}"); return; }
    };
    let question = "What is the secret access code?";
    eprintln!("dovata: [compose-demo] {} blocks, question: {question:?}", store.blocks.len());
    let mut cache = KvBlockCache::new(0); // one-shot demo: caching disabled
    let t = Instant::now();
    match compose_and_answer(model, ctx, &store, &mut cache, question, 4) {
        Ok((ans, sel)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("dovata: [compose-demo] selected {sel:?}");
            eprintln!("dovata: [compose-demo] ANSWER: {:?}", ans.trim());
            eprintln!("dovata: [compose-demo] needle {needle} found: {}  ({ms}ms, ONE decode, zero agentic calls)", ans.contains(needle));
        }
        Err(e) => eprintln!("dovata: [compose-demo] error: {e}"),
    }
}

/// Multi-block composition test (`JADE_COMPOSE_MULTI`). The answer requires TWO different
/// blocks, so composition must place both in the window. Success = both facts present.
fn selftest_compose_multi(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let filler = "This section has routine details and no codes. ".repeat(20);
    let texts: Vec<String> = (0..20).map(|i| match i {
        7 => format!("Section 7: Launch. The launch code is ALPHA-9. {filler}"),
        13 => format!("Section 13: Abort. The abort code is OMEGA-3. {filler}"),
        _ => format!("Section {i}: Misc. {filler}"),
    }).collect();
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("dovata: [compose-multi] build error: {e}"); return; }
    };
    let question = "What are the launch code and the abort code?";
    let mut cache = KvBlockCache::new(0);
    match compose_and_answer(model, ctx, &store, &mut cache, question, 4) {
        Ok((ans, sel)) => {
            let lc = ans.to_lowercase();
            let both = lc.contains("alpha-9") && lc.contains("omega-3");
            eprintln!("dovata: [compose-multi] selected={sel:?} both_facts={both}");
            eprintln!("dovata: [compose-multi]   ANSWER: {:?}", ans.trim());
        }
        Err(e) => eprintln!("dovata: [compose-multi] error: {e}"),
    }
}

/// COLD-vs-WARM compose bench (`JADE_COMPOSE_CACHE`). Answers the same needle question
/// twice over one block store + KV cache: first COLD (retrieved blocks re-prefilled and
/// cached), then WARM (those blocks restored from host RAM). Reports both latencies and
/// asserts the two answers are IDENTICAL — restoring block KV must reproduce the
/// re-prefill composition exactly (greedy). The latency delta is the per-query cache win.
fn selftest_compose_cache(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let n: usize = std::env::var("JADE_RLM_DEMO_SECTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let (texts, _) = needle_haystack(needle, n);
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("dovata: [compose-cache] build error: {e}"); return; }
    };
    let mut cache = KvBlockCache::new(2 * 1024 * 1024 * 1024); // 2 GiB
    let question = "What is the secret access code?";
    eprintln!("dovata: [compose-cache] {} blocks, 2 GiB KV cache, question: {question:?}", store.blocks.len());

    // WARMUP: a throwaway compose (its own cache) so the COLD timing below isn't polluted
    // by one-time CUDA graph compilation on the first decode after model load.
    {
        let mut warm_cache = KvBlockCache::new(0);
        let _ = compose_and_answer(model, ctx, &store, &mut warm_cache, question, 4);
    }

    // COLD: cache empty → retrieved blocks are re-prefilled on the GPU and saved.
    let t = Instant::now();
    let (cold_ans, cold_sel) = match compose_and_answer(model, ctx, &store, &mut cache, question, 4) {
        Ok(r) => r,
        Err(e) => { eprintln!("dovata: [compose-cache] cold error: {e}"); return; }
    };
    let cold_ms = t.elapsed().as_millis();
    let (n_entries, bytes) = cache.stats();

    // WARM: the same blocks are now restored from host RAM instead of recomputed.
    let t = Instant::now();
    let (warm_ans, _) = match compose_and_answer(model, ctx, &store, &mut cache, question, 4) {
        Ok(r) => r,
        Err(e) => { eprintln!("dovata: [compose-cache] warm error: {e}"); return; }
    };
    let warm_ms = t.elapsed().as_millis();

    let same = cold_ans.trim() == warm_ans.trim();
    eprintln!("dovata: [compose-cache] selected {cold_sel:?}; cache now {n_entries} blocks / {:.1} MB",
        bytes as f64 / (1024.0 * 1024.0));
    eprintln!("dovata: [compose-cache] COLD (re-prefill) {cold_ms}ms   WARM (restore) {warm_ms}ms   speedup {:.2}x",
        cold_ms as f64 / warm_ms.max(1) as f64);
    eprintln!("dovata: [compose-cache] prefill saved by caching: {}ms over {} composed block(s) (decode time is identical & unaffected)",
        cold_ms.saturating_sub(warm_ms), cold_sel.len());
    eprintln!("dovata: [compose-cache] answers identical (cold==warm): {same}");
    eprintln!("dovata: [compose-cache] needle {needle} found: cold={} warm={}",
        cold_ans.contains(needle), warm_ans.contains(needle));
    if !same {
        eprintln!("dovata: [compose-cache]   COLD: {:?}", cold_ans.trim());
        eprintln!("dovata: [compose-cache]   WARM: {:?}", warm_ans.trim());
    }
}

/// Blended RAG-RLM eval (`JADE_BLENDED`). Builds a haystack with the needle block plus
/// lexical DISTRACTOR blocks (that echo the query's words but hold no answer), then runs
/// the plain pipeline (retrieve→compose) and the blended pipeline (retrieve N→rerank→
/// compose top-k) on the same store. Prints each pipeline's selection, the needle hit,
/// latency, and the rerank scores (needle/distractor tagged) — so we can see whether the
/// cross-encoder rerank reorders what lexical recall got wrong.
fn selftest_blended(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let n_sections: usize = std::env::var("JADE_RLM_DEMO_SECTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(40);
    let n_cand: usize = std::env::var("JADE_BLEND_N").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    let k: usize = std::env::var("JADE_BLEND_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let filler = "This section documents routine operational details. ".repeat(20);
    let needle_at = n_sections / 2;
    let distractors = [n_sections / 4, (3 * n_sections) / 4, n_sections / 3];
    let texts: Vec<String> = (0..n_sections).map(|i| {
        if i == needle_at {
            format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly. {filler}")
        } else if distractors.contains(&i) {
            format!("Section {i}: Policy. This section discusses the secret access code policy and what a secret access code is in general, but lists no actual code. {filler}")
        } else {
            format!("Section {i}: Misc. {filler}")
        }
    }).collect();
    let store = match BlockStore::from_texts(model, &texts) {
        Ok(s) => s,
        Err(e) => { eprintln!("dovata: [blended] build error: {e}"); return; }
    };
    let question = "What is the secret access code?";
    eprintln!("dovata: [blended] {} blocks (needle @ {needle_at}, distractors {distractors:?}); N={n_cand} k={k}", store.blocks.len());

    // PLAIN: retrieve top-k → compose.
    let mut cache = KvBlockCache::new(2 * 1024 * 1024 * 1024);
    let t = Instant::now();
    match compose_and_answer(model, ctx, &store, &mut cache, question, k) {
        Ok((ans, sel)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("dovata: [blended] PLAIN   sel={sel:?} needle={} ({ms}ms)", ans.contains(needle));
        }
        Err(e) => { eprintln!("dovata: [blended] plain error: {e}"); return; }
    }

    // BLENDED: retrieve N → rerank → compose top-k.
    let mut cache_b = KvBlockCache::new(2 * 1024 * 1024 * 1024);
    let t = Instant::now();
    match blended_compose_and_answer(model, ctx, &store, &mut cache_b, question, n_cand, k) {
        Ok((ans, sel, scored)) => {
            let ms = t.elapsed().as_millis();
            eprintln!("dovata: [blended] BLENDED sel={sel:?} needle={} ({ms}ms)", ans.contains(needle));
            eprintln!("dovata: [blended] rerank scores (top 8, higher=more relevant):");
            for (bi, s) in scored.iter().take(8) {
                let tag = if *bi == needle_at { " <- NEEDLE" }
                    else if distractors.contains(bi) { " <- distractor" } else { "" };
                eprintln!("dovata: [blended]   block {bi:>3}  score {s:>9.4}{tag}");
            }
        }
        Err(e) => { eprintln!("dovata: [blended] blended error: {e}"); }
    }
}

// ── Throughput ─────────────────────────────────────────────────────────────────

/// Transfer-vs-recompute benchmark (`JADE_KV_BENCH`). For several block sizes, compares
/// the two ways the compose engine could materialize a block's KV in the resident window:
///   (a) RE-PREFILL — recompute the block's K/V from its token ids on the GPU (pure
///       compute, no host traffic);
///   (b) RESTORE    — copy the block's precomputed K/V from a CPU-RAM byte buffer back
///       onto the device via `state_seq_set_data` (a host→device transfer, ~27 KB/token).
/// Also reports the one-time `state_seq_get_data` (device→host) save cost and bytes/token.
/// This decides whether caching block KV in host RAM beats recomputing it each query.
/// Each timing is the mean of `iters` runs after a warm-up.
fn selftest_kv_bench(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        // One long filler tokenized once; each block size is a prefix of it, so both
        // paths operate on byte-identical tokens.
        let filler = "This is routine filler text for the kv cache transfer benchmark. ".repeat(500);
        let all = model.str_to_token(&filler, AddBos::Never).map_err(|e| format!("tok: {e}"))?;
        let sizes = [64usize, 128, 256, 512, 1024, 2048];
        let iters = 5usize;

        eprintln!("dovata: [kv-bench] re-prefill (GPU compute) vs CPU-RAM KV restore (PCIe), mean of {iters}");
        eprintln!(
            "dovata: [kv-bench] {:>6} {:>11} {:>11} {:>9} {:>8} {:>9} {:>11}",
            "tokens", "prefill_ms", "restore_ms", "save_ms", "KB/tok", "save_MB", "restore_GB/s"
        );

        for &n in &sizes {
            if all.len() < n {
                break;
            }
            let block = &all[..n];

            // Warm-up (first GPU op of a size pays one-time costs we don't want timed).
            ctx.clear_kv_cache();
            prefill(ctx, block, 0, 0, false)?;

            // (a) RE-PREFILL: recompute the block's KV on seq 0 from token ids.
            let mut prefill_ms = 0f64;
            for _ in 0..iters {
                ctx.clear_kv_cache();
                let t = Instant::now();
                prefill(ctx, block, 0, 0, false)?;
                prefill_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            prefill_ms /= iters as f64;

            // Save the block's KV once (device→host) into a CPU-RAM buffer.
            ctx.clear_kv_cache();
            prefill(ctx, block, 0, 0, false)?;
            let t = Instant::now();
            let bytes = save_seq(ctx, 0);
            let save_ms = t.elapsed().as_secs_f64() * 1000.0;

            // (b) RESTORE: copy the saved KV back onto the device (host→device PCIe).
            let mut restore_ms = 0f64;
            for _ in 0..iters {
                ctx.clear_kv_cache();
                let t = Instant::now();
                let ok = load_seq(ctx, &bytes, 0);
                restore_ms += t.elapsed().as_secs_f64() * 1000.0;
                if !ok {
                    return Err("load_seq returned false".into());
                }
            }
            restore_ms /= iters as f64;

            let save_mb = bytes.len() as f64 / (1024.0 * 1024.0);
            let kb_tok = bytes.len() as f64 / 1024.0 / n as f64;
            let restore_gbs = (bytes.len() as f64 / 1e9) / (restore_ms / 1000.0);
            eprintln!(
                "dovata: [kv-bench] {n:>6} {prefill_ms:>11.3} {restore_ms:>11.3} {save_ms:>9.3} {kb_tok:>8.2} {save_mb:>9.3} {restore_gbs:>11.2}"
            );
        }
        ctx.clear_kv_cache();
        eprintln!("dovata: [kv-bench] restore wins where prefill_ms > restore_ms; that's the cache-KV-in-RAM crossover.");
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("dovata: [kv-bench] error: {e}");
    }
}

/// Pure generation-throughput baseline (`JADE_GEN_BENCH`). For each resident context
/// size, prefill that many tokens (UNTIMED), then time generating a fixed `JADE_GEN_TOKENS`
/// (default 128) tokens one-at-a-time (greedy, EOG ignored so the loop runs the full count
/// for clean timing). Reports decode tok/s per resident size — the metric a user feels,
/// and the one that scales with the composed window the RLM builds.
fn selftest_gen_bench(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let gen_n: usize = std::env::var("JADE_GEN_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let filler = "This is routine filler text for the generation throughput baseline. ".repeat(800);
        let all = model.str_to_token(&filler, AddBos::Never).map_err(|e| format!("tok: {e}"))?;
        let sizes = [128usize, 1024, 4096, 8192];

        // Decode the generation loop once (warm the single-token decode graph) so the
        // first sized measurement isn't polluted by graph compilation.
        ctx.clear_kv_cache();
        prefill(ctx, &all[..64.min(all.len())], 0, 0, true)?;
        let _ = gen_loop(model, ctx, 0, 64, 8)?;

        eprintln!("dovata: [gen-bench] pure decode throughput, generating {gen_n} tokens per resident size");
        eprintln!("dovata: [gen-bench] {:>10} {:>10} {:>12}", "resident", "decode_ms", "decode_tok/s");

        for &csz in &sizes {
            if all.len() < csz {
                break;
            }
            // Prefill the resident context (UNTIMED) with logits on the last token.
            ctx.clear_kv_cache();
            prefill(ctx, &all[..csz], 0, 0, true)?;

            // Time ONLY the serial generation of gen_n tokens (ignore EOG for clean timing).
            let mut sampler = LlamaSampler::greedy();
            let mut batch = LlamaBatch::new(1, 1);
            let mut pos = csz as i32;
            let t = Instant::now();
            for _ in 0..gen_n {
                let tok = sampler.sample(ctx, -1);
                batch.clear();
                batch.add(tok, pos, &[0], true).map_err(|e| format!("gen add: {e}"))?;
                ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
                pos += 1;
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let tok_s = gen_n as f64 / (ms / 1000.0);
            eprintln!("dovata: [gen-bench] {csz:>10} {ms:>10.1} {tok_s:>12.1}");
        }
        ctx.clear_kv_cache();
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("dovata: [gen-bench] error: {e}");
    }
}

/// Generation throughput PAST the context window (`JADE_RLM_OVERWINDOW`). Holds a corpus
/// FAR larger than the model's ctx window in RAM as blocks, composes a FIXED `k` of them
/// into a bounded resident window, then times pure decode. The point: total corpus size
/// does NOT touch resident KV or decode speed — RLM keeps the window bounded, so decode
/// tok/s past the window ≈ the rate for the composed-window size and stays FLAT as the
/// corpus grows. Sweeps corpus = 2×/4×/8× the ctx window.
fn selftest_rlm_overwindow(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let mut run = || -> Result<(), String> {
        let gen_n: usize = std::env::var("JADE_GEN_TOKENS").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let k: usize = std::env::var("JADE_OW_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
        let n_ctx = ctx.n_ctx() as usize;

        // One ~512-token block body, reused; measure its real token length.
        let block_text = format!("routine operational detail with no special codes or secrets {}",
            "and more routine detail ".repeat(150));
        let btok = model.str_to_token(&block_text, AddBos::Never).map_err(|e| format!("tok: {e}"))?.len();

        // Warm the single-token decode graph so the first measurement is clean.
        // (next_pos MUST equal the prefilled length — no position gap.)
        ctx.clear_kv_cache();
        let warm = model.str_to_token("Warm up the decode graph for the bench.", AddBos::Always)
            .map_err(|e| format!("tok: {e}"))?;
        prefill(ctx, &warm, 0, 0, true)?;
        let _ = gen_loop(model, ctx, 0, warm.len() as i32, 8)?;

        let prefix = model.str_to_token("You are given several notes. Use them to answer the question.\n\n", AddBos::Always)
            .map_err(|e| format!("tok prefix: {e}"))?;
        let qtoks = model.str_to_token("\nQuestion: What is the secret access code?\nAnswer: ", AddBos::Never)
            .map_err(|e| format!("tok q: {e}"))?;

        eprintln!("dovata: [overwindow] ctx window = {n_ctx} tokens; composing k={k} blocks (~{btok} tok each), generating {gen_n}");
        eprintln!("dovata: [overwindow] {:>12} {:>9} {:>9} {:>9} {:>10} {:>12}",
            "corpus_tok", "blocks", "x_window", "resident", "decode_ms", "decode_tok/s");

        for mult in [2usize, 4, 8] {
            let corpus = mult * n_ctx;
            let n_blocks = (corpus / btok.max(1)).max(k);
            // Corpus held in RAM as blocks (only k ever touch the GPU).
            let texts: Vec<String> = (0..n_blocks).map(|i| format!("Section {i}. {block_text}")).collect();
            let store = BlockStore::from_texts(model, &texts)?;
            let total: usize = store.blocks.iter().map(|b| b.ids.len()).sum();

            // Compose k evenly-spaced blocks into one bounded window (untimed).
            ctx.clear_kv_cache();
            prefill(ctx, &prefix, 0, 0, false)?;
            let mut off = prefix.len() as i32;
            let mut cache = KvBlockCache::new(0);
            let stride = (n_blocks / k).max(1);
            for j in 0..k {
                let bi = (j * stride).min(n_blocks - 1);
                off += materialize_block(ctx, &mut cache, bi, &store.blocks[bi].ids, off, (j + 1) as i32)?;
            }
            prefill(ctx, &qtoks, off, 0, true)?;
            let resident = off as usize + qtoks.len();

            // Time ONLY the serial decode (EOG ignored for clean timing).
            let mut sampler = LlamaSampler::greedy();
            let mut batch = LlamaBatch::new(1, 1);
            let mut pos = resident as i32;
            let t = Instant::now();
            for _ in 0..gen_n {
                let tok = sampler.sample(ctx, -1);
                batch.clear();
                batch.add(tok, pos, &[0], true).map_err(|e| format!("gen add: {e}"))?;
                ctx.decode(&mut batch).map_err(|e| format!("gen decode: {e}"))?;
                pos += 1;
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            let tok_s = gen_n as f64 / (ms / 1000.0);
            eprintln!("dovata: [overwindow] {total:>12} {n_blocks:>9} {mult:>8}x {resident:>9} {ms:>10.1} {tok_s:>12.1}");
        }
        ctx.clear_kv_cache();
        eprintln!("dovata: [overwindow] resident window stays bounded → decode tok/s flat vs corpus size (standard model: cannot run past {n_ctx}).");
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("dovata: [overwindow] error: {e}");
    }
}

// ── Efficient-frontier study (`JADE_FRONTIER`) ──────────────────────────────────
//
// A RULER-style synthetic sweep that locates where the blended KV-composition engine
// stops answering reliably — the "effective RLM context length" that becomes the
// model profile's `rlm_max_context`. For each task and each corpus size we score three
// arms: NATIVE full-context (the accuracy ceiling, only where the corpus fits the
// window), PLAIN compose (IDF retrieve→compose), and BLENDED (retrieve→rerank→compose).
// Tasks mirror RULER: single needle, multi-key needle, multi-hop variable tracking
// (the binding constraint — multi-hop degrades fastest), and distractor QA. All
// synthetic/templated → zero downloads.

/// Tiny deterministic LCG so cells vary across repeats but reproduce across runs
/// (the workflow can't use a real RNG; positions/values are seeded per task+length).
fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state >> 33
}
/// A random 6-digit value as a string (needle payload).
fn rand6(rng: &mut u64) -> String {
    format!("{}", 100_000 + (lcg(rng) % 900_000))
}
/// A ~500-token filler block with no codes/secrets/assignments (the haystack).
fn frontier_filler(i: usize) -> String {
    format!(
        "Section {i}. This section contains routine operational notes. {}",
        "There are no special codes, secrets, or variable assignments in this part of the document. ".repeat(24)
    )
}
/// Case-insensitive substring hit (the expected answer string appears in the output).
fn frontier_hit(answer: &str, expected: &str) -> bool {
    answer.to_lowercase().contains(&expected.to_lowercase())
}

/// Build one task instance: `n_blocks` filler blocks with the needle(s) placed at
/// distinct random positions. Returns (corpus blocks, question, expected answer string).
fn gen_frontier_task(task: &str, n_blocks: usize, rng: &mut u64) -> (Vec<String>, String, String) {
    let mut corpus: Vec<String> = (0..n_blocks).map(frontier_filler).collect();
    // pick `count` distinct block indices to host needles.
    let pick = |rng: &mut u64, used: &mut Vec<usize>| -> usize {
        loop {
            let p = (lcg(rng) as usize) % n_blocks;
            if !used.contains(&p) {
                used.push(p);
                return p;
            }
        }
    };
    let mut used = Vec::new();
    match task {
        "niah_single" => {
            let val = rand6(rng);
            let p = pick(rng, &mut used);
            corpus[p] = format!("{} The special magic number for the alpha project is {val}. Keep it safe.", corpus[p]);
            (corpus, "What is the special magic number for the alpha project?".into(), val)
        }
        "niah_multikey" => {
            let keys = ["alpha", "bravo", "charlie", "delta"];
            let vals: Vec<String> = (0..keys.len()).map(|_| rand6(rng)).collect();
            for (i, key) in keys.iter().enumerate() {
                let p = pick(rng, &mut used);
                corpus[p] = format!("{} The magic number for the {key} project is {}.", corpus[p], vals[i]);
            }
            let ask = (lcg(rng) as usize) % keys.len();
            (corpus, format!("What is the magic number for the {} project?", keys[ask]), vals[ask].clone())
        }
        "vt" => {
            // multi-hop chain: VAR_1 = value; VAR_2 = VAR_1; ... VAR_H = VAR_{H-1}.
            let hops = 4usize;
            let val = rand6(rng);
            for i in 1..=hops {
                let p = pick(rng, &mut used);
                let stmt = if i == 1 {
                    format!("Note: the variable VAR_1 is assigned the value {val}.")
                } else {
                    format!("Note: the variable VAR_{i} is assigned the same value as VAR_{}.", i - 1)
                };
                corpus[p] = format!("{} {stmt}", corpus[p]);
            }
            (corpus, format!("Tracing the assignments, what numeric value does VAR_{hops} hold?"), val)
        }
        _ /* "qa" */ => {
            // needle + lexically-similar distractors that echo the query but hold no code.
            let val = format!("XQ-{}", rand6(rng));
            let p = pick(rng, &mut used);
            corpus[p] = format!("{} The secret access code is {val}. Rotate it quarterly.", corpus[p]);
            for _ in 0..3 {
                let d = pick(rng, &mut used);
                corpus[d] = format!("{} This section discusses the secret access code policy and what a secret access code is, but lists no actual code.", corpus[d]);
            }
            (corpus, "What is the secret access code?".into(), val)
        }
    }
}

/// Native full-context answer (the accuracy ceiling), or `None` if the corpus + question
/// doesn't fit the window. Concatenates all blocks + the question and decodes once.
fn frontier_native(model: &LlamaModel, ctx: &mut LlamaContext, corpus: &[String], question: &str, n_ctx: usize) -> Option<String> {
    let text = format!("{}\n\nQuestion: {question}\nAnswer: ", corpus.join("\n\n"));
    let toks = model.str_to_token(&text, AddBos::Always).ok()?;
    if toks.len() + 128 > n_ctx {
        return None; // doesn't fit natively — this is the whole point of RLM
    }
    ctx.clear_kv_cache();
    prefill(ctx, &toks, 0, 0, true).ok()?;
    let ans = gen_loop(model, ctx, 0, toks.len() as i32, 96).ok()?;
    ctx.clear_kv_cache();
    Some(ans)
}

fn selftest_frontier(model: &LlamaModel, ctx: &mut LlamaContext, _tmpl: &LlamaChatTemplate) {
    let n_ctx = ctx.n_ctx() as usize;
    let reps: usize = std::env::var("JADE_FRONTIER_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mults: Vec<f64> = std::env::var("JADE_FRONTIER_MULTS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.5, 1.0, 2.0, 4.0, 8.0]);
    // Compose budget — tunable so a sweep can exploit a large native window. k =
    // retrieved blocks composed into the window; N = candidates retrieved before the
    // rerank. Defaults 4/12 preserve the historical 32k-window measurement. Clamp
    // N≥k≥1 so the composed set is well-defined. At n_ctx=262144 with ~500-tok blocks,
    // k can climb into the hundreds (k=128 ≈ 64k tok composed, still ≪ window) — the
    // lever that pushes the hard-task frontier far past today's 8×.
    let k: usize = std::env::var("JADE_FRONTIER_K").ok().and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    let n_cand: usize = std::env::var("JADE_FRONTIER_N").ok().and_then(|s| s.parse().ok()).unwrap_or(12).max(k);
    let tau = 0.85f64;
    // Task subset selector — restrict the sweep to a comma-separated subset (e.g.
    // JADE_FRONTIER_TASKS=niah_single to cheaply bracket the easy-case ceiling, or the
    // hard set niah_multikey,vt,qa to find the limiting frontier). Default = all four.
    const ALL_TASKS: [&str; 4] = ["niah_single", "niah_multikey", "vt", "qa"];
    let tasks: Vec<&str> = match std::env::var("JADE_FRONTIER_TASKS") {
        Ok(s) => {
            let sl = s.to_lowercase();
            let want: Vec<&str> = sl.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).collect();
            ALL_TASKS.into_iter().filter(|t| want.contains(t)).collect()
        }
        Err(_) => ALL_TASKS.to_vec(),
    };

    // measure the filler block's real token length (corpus sizing + reporting).
    let block_tok = model
        .str_to_token(&frontier_filler(0), AddBos::Never)
        .map(|t| t.len())
        .unwrap_or(500)
        .max(1);

    // warm the single-token decode graph so the first cell isn't polluted.
    ctx.clear_kv_cache();
    if let Ok(w) = model.str_to_token("Warm up the decode graph for the frontier study.", AddBos::Always) {
        let _ = prefill(ctx, &w, 0, 0, true).and_then(|_| gen_loop(model, ctx, 0, w.len() as i32, 8));
    }

    eprintln!("dovata: [frontier] RULER-style sweep; window={n_ctx} tok, block≈{block_tok} tok, reps={reps}, k={k}, N={n_cand}, τ={tau}");
    eprintln!("dovata: [frontier] accuracy per arm (native = full-context ceiling; n/a = doesn't fit window)");
    eprintln!("dovata: [frontier] {:>13} {:>6} {:>11} {:>7} {:>8} {:>8} {:>8}",
        "task", "xwin", "corpus_tok", "blocks", "native", "plain", "blended");

    // (task, corpus_tok, blended_accuracy) records for the frontier computation.
    let mut records: Vec<(&str, usize, f64)> = Vec::new();

    for &task in &tasks {
        for &mult in &mults {
            let target = (mult * n_ctx as f64) as usize;
            let n_blocks = (target / block_tok).max(k + 4);
            let corpus_tok = n_blocks * block_tok;
            // deterministic per-cell seed
            let mut rng = 0x9E3779B97F4A7C15u64
                ^ (task.bytes().fold(0u64, |a, b| a.wrapping_mul(131).wrapping_add(b as u64)))
                ^ ((mult * 1000.0) as u64).wrapping_mul(0x100000001B3);

            let (mut nat_hits, mut plain_hits, mut blend_hits) = (0usize, 0usize, 0usize);
            let mut native_ran = false;
            for _ in 0..reps {
                let (corpus, question, expected) = gen_frontier_task(task, n_blocks, &mut rng);
                if let Some(ans) = frontier_native(model, ctx, &corpus, &question, n_ctx) {
                    native_ran = true;
                    if frontier_hit(&ans, &expected) { nat_hits += 1; }
                }
                let store = match BlockStore::from_texts(model, &corpus) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("dovata: [frontier] build error: {e}"); continue; }
                };
                let mut c0 = KvBlockCache::new(0);
                if let Ok((ans, _)) = compose_and_answer(model, ctx, &store, &mut c0, &question, k) {
                    if frontier_hit(&ans, &expected) { plain_hits += 1; }
                }
                let mut c1 = KvBlockCache::new(0);
                if let Ok((ans, _, _)) = blended_compose_and_answer(model, ctx, &store, &mut c1, &question, n_cand, k) {
                    if frontier_hit(&ans, &expected) { blend_hits += 1; }
                }
            }
            let r = reps as f64;
            let blended_acc = blend_hits as f64 / r;
            let native_str = if native_ran { format!("{:.2}", nat_hits as f64 / r) } else { "   n/a".into() };
            eprintln!("dovata: [frontier] {task:>13} {mult:>5}x {corpus_tok:>11} {n_blocks:>7} {native_str:>8} {:>8.2} {:>8.2}",
                plain_hits as f64 / r, blended_acc);
            records.push((task, corpus_tok, blended_acc));
        }
    }

    // Frontier per task = the largest corpus where BLENDED accuracy ≥ τ. A task that
    // ALSO fails at the SMALLEST length (sub-window, where native succeeds) is a
    // length-INDEPENDENT capability gap, not a length frontier — it's flagged and
    // EXCLUDED from the length-frontier min (otherwise one unsupported task type would
    // drive the limit to zero). rlm_max_context = the min largest-passing-length over
    // the tasks the engine actually performs.
    eprintln!("dovata: [frontier] ── effective RLM length per task (largest corpus with blended ≥ {tau}) ──");
    let smallest_corpus = records.iter().map(|(_, tok, _)| *tok).min().unwrap_or(0);
    let mut overall = usize::MAX;
    for &task in &tasks {
        let rows = records.iter().filter(|(t, _, _)| *t == task);
        // a capability gap fails even at the smallest corpus size (where it fits the window).
        let passes_small = rows.clone().any(|(_, tok, acc)| *tok == smallest_corpus && *acc >= tau);
        let best = rows.clone().filter(|(_, _, acc)| *acc >= tau).map(|(_, tok, _)| *tok).max();
        match (passes_small, best) {
            (true, Some(tok)) => {
                eprintln!("dovata: [frontier]   {task:>13}: {tok} tok");
                overall = overall.min(tok);
            }
            _ => eprintln!("dovata: [frontier]   {task:>13}: CAPABILITY GAP (fails sub-window; excluded from frontier)"),
        }
    }
    if overall == usize::MAX { overall = 0; } // every task was a capability gap
    eprintln!("dovata: [frontier] SUGGESTED rlm_max_context = {overall} tok (min length-frontier over supported tasks; set in the model profile)");
}

// ── Continuous-batching engine self-test (`JADE_RLM_SELFTEST`) ──────────────────

/// Build one [`SubInfer`] from a plain prompt: apply the chat template, tokenize, and
/// attach a fresh `DecodeControl` (+ optional grammar). Used by [`selftest_rlm`].
fn build_sub(
    model: &LlamaModel,
    _tmpl: &LlamaChatTemplate,
    prompt: &str,
    grammar: Option<&str>,
    max_new: usize,
) -> Result<SubInfer, String> {
    let formatted = build_formatted_prompt(prompt, &ChatTemplate::default())?;
    let tokens = model
        .str_to_token(&formatted, AddBos::Always)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let req = InferenceRequest {
        prompt: prompt.to_owned(),
        max_tokens: max_new as u32,
        grammar: grammar.map(|g| g.to_owned()),
        ..Default::default()
    };
    Ok(SubInfer {
        tokens,
        base_pos: 0,
        grammar: grammar.map(|g| g.to_owned()),
        control: DecodeControl::new(&req),
        max_new,
    })
}

/// Continuous-batching fan-out self-test. Build a handful of sub-inferences, run them
/// both sequentially (baseline) and concurrently (one batch/tick), and print outputs +
/// wall-clock so the fan-out speedup and correctness are visible. Check `nvidia-smi`
/// alongside to confirm VRAM tracks live KV rather than scaling with the number of
/// sub-calls. Exercises the [`super::concurrency`] engine, retained for concurrent /
/// mass prompting independent of the RLM compose path.
fn selftest_rlm(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    let prompts = [
        "Reply with exactly one word: the capital of France.",
        "Reply with exactly one word: a primary color.",
        "Reply with exactly one word: the largest planet.",
        "Reply with exactly one word: the opposite of hot.",
        "Reply with exactly one word: a fruit that is yellow.",
        "Reply with exactly one word: the chemical symbol for water.",
        "Reply with exactly one word: the number after ten.",
        "Reply with exactly one word: a animal that barks.",
    ];
    let max_new = 24usize;

    // Build a SubInfer per prompt (tokenize through the model's chat template).
    let build_subs = || -> Result<Vec<SubInfer>, String> {
        prompts
            .iter()
            .map(|p| build_sub(model, tmpl, p, None, max_new))
            .collect()
    };

    eprintln!("dovata: [selftest] {} sub-inferences, max_new={max_new}", prompts.len());

    // Sequential baseline: one sub per run_concurrent call, summed wall-clock.
    let seq_start = Instant::now();
    let mut seq_outputs = Vec::new();
    match build_subs() {
        Ok(subs) => {
            for sub in subs {
                match run_concurrent(model, ctx, vec![sub]) {
                    Ok(mut o) => seq_outputs.push(o.pop().unwrap_or_default()),
                    Err(e) => { eprintln!("dovata: [selftest] sequential error: {e}"); return; }
                }
            }
        }
        Err(e) => { eprintln!("dovata: [selftest] build error: {e}"); return; }
    }
    let seq_ms = seq_start.elapsed().as_millis();

    // Concurrent: all subs in one run (one decode per tick advances all).
    let con_start = Instant::now();
    let con_outputs = match build_subs() {
        Ok(subs) => match run_concurrent(model, ctx, subs) {
            Ok(o) => o,
            Err(e) => { eprintln!("dovata: [selftest] concurrent error: {e}"); return; }
        },
        Err(e) => { eprintln!("dovata: [selftest] build error: {e}"); return; }
    };
    let con_ms = con_start.elapsed().as_millis();

    eprintln!("dovata: [selftest] ── outputs (concurrent) ──");
    for (i, (p, o)) in prompts.iter().zip(con_outputs.iter()).enumerate() {
        eprintln!("  [{i}] {p}\n       -> {:?}", o.trim());
    }
    eprintln!("dovata: [selftest] sequential={seq_ms}ms  concurrent={con_ms}ms  speedup={:.2}x",
        seq_ms as f64 / con_ms.max(1) as f64);
    // Sanity: concurrent outputs should match sequential (same greedy-ish sampling
    // path, modulo the shared free sampler's RNG — compare trimmed first words).
    let matches = seq_outputs.iter().zip(con_outputs.iter())
        .filter(|(a, b)| a.trim() == b.trim()).count();
    eprintln!("dovata: [selftest] concurrent==sequential on {matches}/{} outputs", prompts.len());

    // Concurrency sweep: latency + throughput vs N concurrent sub-inferences (the A10G
    // knee). Latency staying ~flat as N grows = continuous batching working.
    eprintln!("dovata: [selftest] ── concurrency sweep (N concurrent, max_new={max_new}) ──");
    for &nn in &[1usize, 2, 4, 8, 16] {
        let mut subs = Vec::with_capacity(nn);
        let mut ok = true;
        for k in 0..nn {
            match build_sub(model, tmpl, prompts[k % prompts.len()], None, max_new) {
                Ok(s) => subs.push(s),
                Err(_) => { ok = false; break; }
            }
        }
        if !ok { continue; }
        let t = Instant::now();
        if run_concurrent(model, ctx, subs).is_err() { continue; }
        let ms = t.elapsed().as_millis().max(1);
        eprintln!("  N={nn:2}  {ms:4}ms  throughput {:.1} sub-inferences/s", nn as f64 * 1000.0 / ms as f64);
    }
}
