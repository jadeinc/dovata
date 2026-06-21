//! Recursive Language Model: reason over a context far larger than the model
//! window / VRAM. The full context lives in CPU RAM as sliceable spans
//! ([`ContextStore`]); only the spans the ROOT model chooses to look at are ever
//! tokenized and put on the GPU. Model-driven recursion in three in-process steps
//! (zero socket roundtrips):
//!   PICK    — root sees a structural MAP of the spans + the question and emits
//!             (grammar-forced) a JSON array of the span ids to examine;
//!   FAN-OUT — those spans run as CONCURRENT sub-inferences, each answering from
//!             its excerpt alone (the latency win);
//!   COMBINE — root sees the per-span findings and writes the final answer.
//!
//! Three latency optimizations: a ROUTER that takes the standard single-shot path
//! when the whole context fits one sequence (`run_rlm`'s `force_rlm`), shared-prefix
//! KV reuse across fan-out spans, and skipping COMBINE for a single-span pick.
//! This module also owns the socket dispatch (`handle_rlm`) and context handles.

use super::concurrency::{run_concurrent, run_concurrent_inner, SubInfer};
use super::kv::common_prefix_len;
use super::{build_formatted_prompt, MAX_SEQ_TOKENS};
use jade_protocol::{Frame, InferenceRequest};
use jaded::control::DecodeControl;
use llama_cpp_2::{
    context::LlamaContext,
    model::{AddBos, LlamaChatTemplate, LlamaModel},
    token::LlamaToken,
};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// A large context held in CPU RAM, addressable by span. Each span carries a short
/// `summary` (shown to the root in the map) and its full `body` text (only fetched
/// into a prompt if the root selects it).
#[derive(Clone)]
pub(crate) struct ContextStore {
    summaries: Vec<String>,
    bodies: Vec<String>,
}

impl ContextStore {
    fn from_sections(sections: &[(String, String)]) -> Self {
        ContextStore {
            summaries: sections.iter().map(|(s, _)| s.clone()).collect(),
            bodies: sections.iter().map(|(_, b)| b.clone()).collect(),
        }
    }
    fn len(&self) -> usize {
        self.bodies.len()
    }
    /// The block texts — the corpus the KV-composition engine builds blocks from.
    pub(crate) fn bodies(&self) -> &[String] {
        &self.bodies
    }
    /// The structural map the root navigates: one line per span, `id: summary`.
    fn map_text(&self) -> String {
        self.summaries
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{i}: {s}"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// GBNF forcing a JSON array of integers, e.g. `[2, 5]`. Contains `[0-9]`, so
/// `DecodeControl` treats it as a full-output primitive grammar (never retired).
const SPAN_ARRAY_GBNF: &str = r#"root ::= "[" ws (num (ws "," ws num)*)? ws "]"
num ::= [0-9]+
ws ::= [ \t\n]*"#;

/// Pull the integers out of a model-emitted span array like `[3, 7]`.
fn parse_span_ids(s: &str) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            cur.push(ch);
        } else if !cur.is_empty() {
            if let Ok(n) = cur.parse::<usize>() {
                out.push(n);
            }
            cur.clear();
        }
    }
    if let Ok(n) = cur.parse::<usize>() {
        out.push(n);
    }
    out
}

/// Build one [`SubInfer`] from a plain prompt: apply the chat template, tokenize,
/// and attach a fresh `DecodeControl` (+ optional grammar).
fn build_sub(
    model: &LlamaModel,
    tmpl: &LlamaChatTemplate,
    prompt: &str,
    grammar: Option<&str>,
    max_new: usize,
) -> Result<SubInfer, String> {
    let formatted = build_formatted_prompt(prompt, model, tmpl)?;
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

/// Answer `question` over `store` by model-driven recursion (PICK → FAN-OUT →
/// COMBINE). Returns the final answer plus the span ids the root chose (for
/// observability). `max_spans` caps fan-out width (the concurrency ceiling).
/// `force_rlm` bypasses the router (used by the latency A/B to measure raw RLM).
pub(crate) fn run_rlm(
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &LlamaChatTemplate,
    question: &str,
    store: &ContextStore,
    max_spans: usize,
    force_rlm: bool,
) -> Result<(String, Vec<usize>), String> {
    // ── OPT 1: Router. If the whole document fits one sequence's window, answer it
    // directly with a single standard inference — RLM's three rounds are pure
    // overhead when the context already fits; only pay them when it can't.
    let budget = MAX_SEQ_TOKENS.load(Ordering::Relaxed) as usize;
    if !force_rlm && budget > 0 {
        let full: String = (0..store.len())
            .map(|i| format!("{}\n{}", store.summaries[i], store.bodies[i]))
            .collect::<Vec<_>>()
            .join("\n\n");
        let std_prompt = format!("Document:\n{full}\n\nQuestion: {question}\nAnswer concisely.");
        let std_toks = model.str_to_token(&std_prompt, AddBos::Always).map(|t| t.len()).unwrap_or(usize::MAX);
        let headroom = 200usize;
        if std_toks + headroom <= budget {
            eprintln!("jaded: [rlm] router: doc fits ({std_toks}+{headroom} <= {budget}) — standard single-shot");
            let sub = build_sub(model, tmpl, &std_prompt, None, headroom)?;
            let ans = run_concurrent(model, ctx, vec![sub])?.pop().unwrap_or_default();
            return Ok((ans, Vec::new())); // empty picked = router took the standard path
        }
    }

    // ── PICK: root chooses which spans to examine (grammar-forced array) ────────
    let pick_prompt = format!(
        "You are navigating a large document split into numbered sections:\n{}\n\n\
         Question: {question}\n\n\
         Reply with ONLY a JSON array of the section numbers most likely to contain \
         the answer (e.g. [2, 5]). Choose at most {max_spans}.",
        store.map_text()
    );
    let pick_sub = build_sub(model, tmpl, &pick_prompt, Some(SPAN_ARRAY_GBNF), 48)?;
    let t_pick = Instant::now();
    let pick_out = run_concurrent(model, ctx, vec![pick_sub])?.pop().unwrap_or_default();
    let pick_ms = t_pick.elapsed().as_millis();
    let mut picked: Vec<usize> = parse_span_ids(&pick_out)
        .into_iter()
        .filter(|&i| i < store.len())
        .collect();
    picked.dedup();
    picked.truncate(max_spans);
    // Fallback: if the model emitted nothing usable, scan the first few spans.
    if picked.is_empty() {
        picked = (0..store.len().min(max_spans)).collect();
    }
    eprintln!("jaded: [rlm] PICK -> spans {picked:?} (raw {:?})", pick_out.trim());

    // ── FAN-OUT: one concurrent sub-inference per chosen span. OPT 2: the shared
    // instruction/question preamble is the PREFIX and the unique excerpt body the
    // SUFFIX, so the preamble's KV is prefilled once and copied into every span —
    // each span then only prefills its own body.
    let preamble = format!(
        "You are answering a question using one document excerpt.\n\
         Question: {question}\n\
         Answer using ONLY the excerpt below. If the answer is not present, reply \
         exactly: NOT FOUND.\n\nExcerpt:\n"
    );
    let mut toks: Vec<Vec<LlamaToken>> = Vec::with_capacity(picked.len());
    for &i in &picked {
        let formatted = build_formatted_prompt(&format!("{preamble}{}", store.bodies[i]), model, tmpl)?;
        toks.push(model.str_to_token(&formatted, AddBos::Always).map_err(|e| format!("tokenize: {e}"))?);
    }
    // Shared-prefix reuse only helps with ≥2 spans; for a single span there's
    // nothing to share, and using the whole prompt as the prefix would leave an
    // EMPTY suffix — a sequence with no tokens to prefill never produces logits, so
    // its output would be empty. Cap to `min_len - 1` so every span keeps ≥1 suffix.
    let shared_len = if picked.len() >= 2 {
        let min_len = toks.iter().map(|t| t.len()).min().unwrap_or(0);
        common_prefix_len(&toks).min(min_len.saturating_sub(1))
    } else {
        0
    };
    let shared: Vec<LlamaToken> = toks[0][..shared_len].to_vec();
    let fan_subs: Vec<SubInfer> = toks
        .iter()
        .map(|t| SubInfer {
            tokens: t[shared_len..].to_vec(),
            base_pos: shared_len as i32,
            grammar: None,
            control: DecodeControl::new(&InferenceRequest { max_tokens: 80, ..Default::default() }),
            max_new: 80,
        })
        .collect();
    let t_fan = Instant::now();
    let findings = run_concurrent_inner(model, ctx, &shared, fan_subs)?;
    let fan_ms = t_fan.elapsed().as_millis();
    eprintln!("jaded: [rlm] fan-out: {shared_len} shared-prefix tokens reused across {} spans", picked.len());
    for (k, &i) in picked.iter().enumerate() {
        eprintln!("jaded: [rlm] span {i} -> {:?}", findings[k].trim());
    }

    // ── OPT 3: a single chosen span needs no COMBINE round — its finding IS the
    // answer. Skip the third inference entirely.
    if picked.len() == 1 {
        eprintln!("jaded: [rlm] timing: pick={pick_ms}ms fan-out(1 span)={fan_ms}ms combine=SKIPPED");
        return Ok((findings[0].trim().to_string(), picked));
    }

    // ── COMBINE: root writes the final answer from the findings ─────────────────
    let mut combine = format!("Question: {question}\n\nFindings from the document:\n");
    for (k, &i) in picked.iter().enumerate() {
        combine.push_str(&format!("- (section {i}) {}\n", findings[k].trim()));
    }
    combine.push_str("\nUsing only these findings, give the final answer concisely.");
    let combine_sub = build_sub(model, tmpl, &combine, None, 120)?;
    let t_comb = Instant::now();
    let answer = run_concurrent(model, ctx, vec![combine_sub])?.pop().unwrap_or_default();
    let comb_ms = t_comb.elapsed().as_millis();
    eprintln!(
        "jaded: [rlm] timing: pick={pick_ms}ms fan-out({} spans)={fan_ms}ms combine={comb_ms}ms",
        picked.len()
    );
    Ok((answer, picked))
}

// ── RLM over the socket: context handles + request dispatch ────────────────────

/// Server-side context store: a program uploads a large context ONCE and gets a
/// handle, then issues many cheap `rlm` queries against it (no re-sending the doc).
static RLM_STORES: OnceLock<Mutex<HashMap<u64, ContextStore>>> = OnceLock::new();
static RLM_NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn rlm_stores() -> &'static Mutex<HashMap<u64, ContextStore>> {
    RLM_STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Store an uploaded context, returning its handle.
fn store_context(store: ContextStore) -> u64 {
    let h = RLM_NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    rlm_stores().lock().unwrap().insert(h, store);
    h
}

/// Fetch a clone of a stored context (cloned so the lock isn't held during the
/// long-running inference that follows).
fn get_context(h: u64) -> Option<ContextStore> {
    rlm_stores().lock().unwrap().get(&h).cloned()
}

/// Parse a JSON array of `["summary","body"]` pairs into sections.
fn parse_sections(v: &serde_json::Value) -> Vec<(String, String)> {
    v.as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let pair = e.as_array()?;
                    Some((pair.get(0)?.as_str()?.to_owned(), pair.get(1)?.as_str()?.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Handle an `rlm` request: `prompt` is a JSON payload — an upload (store a context,
/// reply with its handle) or a query (answer a question over a stored/inline context
/// via `run_rlm`, streamed as one Token frame then Done).
pub(crate) fn handle_rlm(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
    tmpl: &LlamaChatTemplate,
    stream: &mut UnixStream,
    model_name: &str,
) -> std::io::Result<()> {
    let payload: serde_json::Value = match serde_json::from_str(&req.prompt) {
        Ok(v) => v,
        Err(e) => {
            stream.write_all(&Frame::Error(format!("rlm: malformed payload JSON: {e}")).encode())?;
            return Ok(());
        }
    };

    // UPLOAD: store the context, return its handle.
    if let Some(up) = payload.get("upload") {
        let sections = parse_sections(up);
        if sections.is_empty() {
            stream.write_all(&Frame::Error("rlm: upload had no [summary,body] sections".into()).encode())?;
            return Ok(());
        }
        let h = store_context(ContextStore::from_sections(&sections));
        eprintln!("jaded: [rlm] uploaded context handle={h} ({} sections)", sections.len());
        stream.write_all(&Frame::Json(format!("{{\"handle\":{h}}}")).encode())?;
        stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        return Ok(());
    }

    // QUERY: resolve the context (by handle or inline), then run the RLM.
    let question = payload.get("question").and_then(|q| q.as_str()).unwrap_or("");
    let max_spans = payload.get("max_spans").and_then(|m| m.as_u64()).unwrap_or(4) as usize;
    // `"compose": true` selects the tensor-level KV-composition engine (retrieve →
    // splice blocks → single decode); otherwise the agentic PICK→fan-out→COMBINE path.
    let compose = payload.get("compose").and_then(|c| c.as_bool()).unwrap_or(false);
    let store = if let Some(h) = payload.get("handle").and_then(|h| h.as_u64()) {
        match get_context(h) {
            Some(s) => s,
            None => {
                stream.write_all(&Frame::Error(format!("rlm: unknown context handle {h}")).encode())?;
                return Ok(());
            }
        }
    } else if let Some(secs) = payload.get("sections") {
        ContextStore::from_sections(&parse_sections(secs))
    } else {
        stream.write_all(&Frame::Error("rlm: query needs a \"handle\" or inline \"sections\"".into()).encode())?;
        return Ok(());
    };

    stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;
    let result = if compose {
        // Tensor-level KV composition (needs the daemon launched with JADE_KV_UNIFIED=1).
        match super::compose::BlockStore::from_texts(model, store.bodies()) {
            Ok(bs) => super::compose::compose_and_answer(model, ctx, &bs, question, max_spans.max(1), 0.0),
            Err(e) => Err(e),
        }
    } else {
        run_rlm(model, ctx, tmpl, question, &store, max_spans.max(1), false)
    };
    match result {
        Ok((answer, picked)) => {
            eprintln!("jaded: [rlm] query answered ({} {picked:?})",
                if compose { "compose blocks" } else { "agentic spans" });
            stream.write_all(&Frame::Token(answer).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        }
        Err(e) => {
            stream.write_all(&Frame::Error(format!("rlm: {e}")).encode())?;
        }
    }
    Ok(())
}

// ── Self-tests / demos (gated by env vars in `run`) ────────────────────────────

/// Self-test for the concurrent engine: build a handful of sub-inferences, run them
/// both sequentially (baseline) and concurrently (one batch/tick), and print
/// outputs + wall-clock so the fan-out speedup and correctness are visible. Gated
/// by `JADE_RLM_SELFTEST`. Check `nvidia-smi` alongside to confirm VRAM tracks
/// live KV rather than scaling with the number of sub-calls.
pub(crate) fn selftest_rlm(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
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

    eprintln!("jaded: [selftest] {} sub-inferences, max_new={max_new}", prompts.len());

    // Sequential baseline: one sub per run_concurrent call, summed wall-clock.
    let seq_start = Instant::now();
    let mut seq_outputs = Vec::new();
    match build_subs() {
        Ok(subs) => {
            for sub in subs {
                match run_concurrent(model, ctx, vec![sub]) {
                    Ok(mut o) => seq_outputs.push(o.pop().unwrap_or_default()),
                    Err(e) => { eprintln!("jaded: [selftest] sequential error: {e}"); return; }
                }
            }
        }
        Err(e) => { eprintln!("jaded: [selftest] build error: {e}"); return; }
    }
    let seq_ms = seq_start.elapsed().as_millis();

    // Concurrent: all subs in one run (one decode per tick advances all).
    let con_start = Instant::now();
    let con_outputs = match build_subs() {
        Ok(subs) => match run_concurrent(model, ctx, subs) {
            Ok(o) => o,
            Err(e) => { eprintln!("jaded: [selftest] concurrent error: {e}"); return; }
        },
        Err(e) => { eprintln!("jaded: [selftest] build error: {e}"); return; }
    };
    let con_ms = con_start.elapsed().as_millis();

    eprintln!("jaded: [selftest] ── outputs (concurrent) ──");
    for (i, (p, o)) in prompts.iter().zip(con_outputs.iter()).enumerate() {
        eprintln!("  [{i}] {p}\n       -> {:?}", o.trim());
    }
    eprintln!("jaded: [selftest] sequential={seq_ms}ms  concurrent={con_ms}ms  speedup={:.2}x",
        seq_ms as f64 / con_ms.max(1) as f64);
    // Sanity: concurrent outputs should match sequential (same greedy-ish sampling
    // path, modulo the shared free sampler's RNG — compare trimmed first words).
    let matches = seq_outputs.iter().zip(con_outputs.iter())
        .filter(|(a, b)| a.trim() == b.trim()).count();
    eprintln!("jaded: [selftest] concurrent==sequential on {matches}/{} outputs", prompts.len());

    // Concurrency sweep: latency + throughput vs N concurrent sub-inferences (the
    // A10G knee). Latency staying ~flat as N grows = continuous batching working.
    eprintln!("jaded: [selftest] ── concurrency sweep (N concurrent, max_new={max_new}) ──");
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

/// End-to-end RLM demo (gated by `JADE_RLM_DEMO`): a needle-in-haystack over a
/// document far larger than any single prompt would want, proving the root can
/// navigate by map, fan out concurrently into the right span, and answer — with the
/// full document held in RAM, never stuffed into one context.
pub(crate) fn selftest_rlm_demo(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    // Build a haystack large enough that the WHOLE document cannot fit in the
    // context window — proving RLM reasons beyond the model's window. Each section
    // body is padded with filler; the needle lives in the "Access Credentials" one.
    let needle = "XQ-42";
    let n_sections: usize = std::env::var("JADE_RLM_DEMO_SECTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let topics = [
        "Introduction", "Hardware Overview", "Boot Sequence", "Networking",
        "Filesystem Layout", "Process Model", "Logging", "Power Management",
        "Update Channels", "Telemetry", "Scheduling", "Memory Management",
        "Device Drivers", "Cryptography", "Backups", "Glossary",
    ];
    // ~1000 tokens of filler per section: a FEW big sections sum past the window
    // while each section's body — and the short map — still fit one sequence's KV
    // slot (n_ctx / n_seq_max). This is the RLM shape: large content, small map.
    let filler = "This section documents routine operational details with no special \
                  codes or secrets. ".repeat(60);
    let needle_at = n_sections / 2; // put the credentials section in the middle
    let mut sections: Vec<(String, String)> = Vec::new();
    for i in 0..n_sections {
        if i == needle_at {
            sections.push((
                format!("Section {i}: Access Credentials"),
                format!("Section {i}: Access Credentials. For maintenance, operators authenticate \
                         with the device. The secret access code is {needle}. Rotate it quarterly. {filler}"),
            ));
        } else {
            let t = topics[i % topics.len()];
            sections.push((
                format!("Section {i}: {t}"),
                format!("Section {i}: {t}. This section describes {t} in general terms. {filler}"),
            ));
        }
    }
    let store = ContextStore::from_sections(&sections);
    let question = "What is the secret access code?";

    // Report how big the whole document is vs the context window — the point is it
    // does NOT fit, yet RLM answers by selecting only the relevant span.
    let ctx_size = ctx.n_ctx();
    let full_doc: String = store.bodies.join("\n\n");
    let full_tokens = model.str_to_token(&full_doc, AddBos::Always).map(|t| t.len()).unwrap_or(0);
    eprintln!(
        "jaded: [rlm-demo] {} sections, ~{full_tokens} tokens total vs ctx_size={ctx_size} \
         ({}), question: {question:?}",
        store.len(),
        if full_tokens as u32 > ctx_size { "DOES NOT FIT — RLM required" } else { "fits" }
    );
    let start = Instant::now();
    match run_rlm(model, ctx, tmpl, question, &store, 4, false) {
        Ok((answer, picked)) => {
            let ms = start.elapsed().as_millis();
            eprintln!("jaded: [rlm-demo] picked spans {picked:?}");
            eprintln!("jaded: [rlm-demo] ANSWER: {:?}", answer.trim());
            let ok = answer.contains(needle);
            eprintln!("jaded: [rlm-demo] needle {needle} found in answer: {ok}  ({ms}ms)");
        }
        Err(e) => eprintln!("jaded: [rlm-demo] error: {e}"),
    }
}

/// Latency A/B (gated by `JADE_RLM_COMPARE`): standard single-shot (stuff the WHOLE
/// document into one prompt → one inference) vs RLM (PICK→fan-out→COMBINE), on a
/// SMALL document that fits one sequence's KV slot so both are even possible. Prints
/// the RLM overhead — the cost you pay on small contexts for the machinery that lets
/// large contexts work at all (where standard isn't an option).
pub(crate) fn selftest_rlm_compare(model: &LlamaModel, ctx: &mut LlamaContext, tmpl: &LlamaChatTemplate) {
    let needle = "XQ-42";
    let topics = [
        "Introduction", "Hardware", "Boot", "Networking", "Filesystem", "Processes",
        "Access Credentials", "Logging", "Power", "Updates", "Telemetry", "Glossary",
    ];
    let mut sections: Vec<(String, String)> = Vec::new();
    for (i, t) in topics.iter().enumerate() {
        let body = if *t == "Access Credentials" {
            format!("Section {i}: Access Credentials. The secret access code is {needle}. Rotate it quarterly.")
        } else {
            format!("Section {i}: {t}. Routine details about {t} with no secrets.")
        };
        sections.push((format!("Section {i}: {t}"), body));
    }
    let store = ContextStore::from_sections(&sections);
    let question = "What is the secret access code?";

    let full: String = sections.iter().map(|(s, b)| format!("{s}\n{b}")).collect::<Vec<_>>().join("\n\n");
    let full_toks = model.str_to_token(&full, AddBos::Always).map(|t| t.len()).unwrap_or(0);

    // STANDARD baseline: whole document + question in a single prompt, one inference.
    let std_prompt = format!("Document:\n{full}\n\nQuestion: {question}\nAnswer concisely.");
    let std_sub = match build_sub(model, tmpl, &std_prompt, None, 120) {
        Ok(s) => s,
        Err(e) => { eprintln!("jaded: [compare] build error: {e}"); return; }
    };
    let t = Instant::now();
    let std_ans = match run_concurrent(model, ctx, vec![std_sub]) {
        Ok(mut o) => o.pop().unwrap_or_default(),
        Err(e) => { eprintln!("jaded: [compare] standard error: {e}"); return; }
    };
    let std_ms = t.elapsed().as_millis().max(1);

    // RLM forced (raw machinery — router bypassed — with opts 2 & 3 applied).
    let t = Instant::now();
    let (rlm_ans, _) = match run_rlm(model, ctx, tmpl, question, &store, 4, true) {
        Ok(r) => r,
        Err(e) => { eprintln!("jaded: [compare] rlm error: {e}"); return; }
    };
    let rlm_ms = t.elapsed().as_millis().max(1);

    // RLM routed (OPT 1 active): since the doc fits, the router should take the
    // standard path → latency ≈ standard.
    let t = Instant::now();
    let (routed_ans, routed_picked) = match run_rlm(model, ctx, tmpl, question, &store, 4, false) {
        Ok(r) => r,
        Err(e) => { eprintln!("jaded: [compare] routed error: {e}"); return; }
    };
    let routed_ms = t.elapsed().as_millis().max(1);

    eprintln!("jaded: [compare] doc {} sections (~{full_toks} tokens, fits the window)", sections.len());
    eprintln!("jaded: [compare] STANDARD      {std_ms:>5}ms -> {:?}", std_ans.trim());
    eprintln!("jaded: [compare] RLM (forced)  {rlm_ms:>5}ms -> {:?}", rlm_ans.trim());
    eprintln!(
        "jaded: [compare] RLM (routed)  {routed_ms:>5}ms (router took {} path) -> {:?}",
        if routed_picked.is_empty() { "STANDARD" } else { "rlm" },
        routed_ans.trim()
    );
    eprintln!(
        "jaded: [compare] forced RLM vs standard: {:.2}x (+{}ms)   routed vs standard: {:.2}x",
        rlm_ms as f64 / std_ms as f64,
        rlm_ms.saturating_sub(std_ms),
        routed_ms as f64 / std_ms as f64
    );
}
