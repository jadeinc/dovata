//! Recursive Language Model: reason over a context far larger than the model
//! window / VRAM. The full context lives in CPU RAM ([`StoredContext`]); only the
//! blocks a query retrieves are ever tokenized and put on the GPU.
//!
//! The engine is tensor-level **KV-cache composition** (see [`super::compose`]): per
//! query, retrieve the top-k blocks, splice their precomputed KV into one attention
//! window, and answer in a single decode — no per-branch re-prompting, no agentic
//! decision calls. (The earlier agentic PICK→fan-out→COMBINE engine was removed once
//! composition proved both faster and simpler.) An uploaded context is tokenized into
//! blocks ONCE and carries a host-RAM [`KvBlockCache`](super::compose::KvBlockCache),
//! so after the first query touches a block its KV is RESTORED (cheap host→device
//! copy) rather than re-prefilled — ~50× cheaper per block on the A10G. This module
//! owns the socket dispatch ([`handle_rlm`]) and the upload context handles.
//!
//! The continuous-batching engine ([`super::concurrency`]) is retained as general
//! infrastructure for concurrent / mass prompting, independent of RLM; its self-test
//! lives in [`super::bench`].

use super::compose::{blended_compose_and_answer, BlockStore, KvBlockCache};
use ovata_infer_protocol::{Frame, InferenceRequest};
use llama_cpp_2::{context::LlamaContext, model::LlamaModel};
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

// ── RLM over the socket: context handles + request dispatch ────────────────────

/// A persistent uploaded context: its blocks tokenized ONCE plus a host-RAM KV cache,
/// both reused across every query on the handle — so retrieved blocks are RESTORED
/// (cheap) rather than re-prefilled after the first query that touches them.
struct StoredContext {
    blocks: BlockStore,
    kv_cache: KvBlockCache,
}

/// Server-side context store: a program uploads a large context ONCE and gets a
/// handle, then issues many cheap `rlm` queries against it (no re-sending the doc).
/// The daemon is single-threaded (connections are handled in series on one context),
/// so holding this lock across a query's inference is uncontended.
static RLM_STORES: OnceLock<Mutex<HashMap<u64, StoredContext>>> = OnceLock::new();
static RLM_NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

fn rlm_stores() -> &'static Mutex<HashMap<u64, StoredContext>> {
    RLM_STORES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Store an uploaded context, returning its handle.
fn store_context(sc: StoredContext) -> u64 {
    let h = RLM_NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
    rlm_stores().lock().unwrap().insert(h, sc);
    h
}

/// Host-RAM budget for a context's precomputed block KV (`JADE_KV_CACHE_MB`, default
/// 2048). At ~27 KB/token this caps how much block KV is held resident for restore.
fn kv_cache_cap_bytes() -> usize {
    std::env::var("JADE_KV_CACHE_MB")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2048)
        .saturating_mul(1024 * 1024)
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

/// Handle an `rlm` request: `prompt` is a JSON payload — an upload (tokenize the
/// context into blocks, store it with a fresh KV cache, reply with its handle) or a
/// query (answer a question over a stored/inline context via the KV-composition
/// engine, streamed as one Token frame then Done).
pub(crate) fn handle_rlm(
    req: &InferenceRequest,
    model: &LlamaModel,
    ctx: &mut LlamaContext,
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

    // UPLOAD: tokenize into blocks, store with a KV cache, return the handle.
    if let Some(up) = payload.get("upload") {
        let sections = parse_sections(up);
        if sections.is_empty() {
            stream.write_all(&Frame::Error("rlm: upload had no [summary,body] sections".into()).encode())?;
            return Ok(());
        }
        let n = sections.len();
        let bodies: Vec<String> = sections.into_iter().map(|(_, b)| b).collect();
        let blocks = match BlockStore::from_texts(model, &bodies) {
            Ok(b) => b,
            Err(e) => {
                stream.write_all(&Frame::Error(format!("rlm: upload tokenize failed: {e}")).encode())?;
                return Ok(());
            }
        };
        let cap = kv_cache_cap_bytes();
        let h = store_context(StoredContext { blocks, kv_cache: KvBlockCache::new(cap) });
        eprintln!("jaded: [rlm] uploaded context handle={h} ({n} sections, kv cache cap {} MB)", cap / (1024 * 1024));
        stream.write_all(&Frame::Json(format!("{{\"handle\":{h}}}")).encode())?;
        stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        return Ok(());
    }

    // QUERY: blended compose-and-answer (PICK→rerank→COMBINE) over a stored handle
    // (warm cache persists) or an inline one-shot context (ephemeral, no caching).
    let question = payload.get("question").and_then(|q| q.as_str()).unwrap_or("");
    // `max_spans` caps how many blocks COMBINE composes into the window (the k); the
    // reranker considers a wider candidate pool first.
    let k = (payload.get("max_spans").and_then(|m| m.as_u64()).unwrap_or(4) as usize).max(1);
    let n_cand = (k * 3).max(8);
    // blended returns (answer, composed top-k, rerank scores); drop the scores here.
    let result: Result<(String, Vec<usize>), String> =
        if let Some(h) = payload.get("handle").and_then(|h| h.as_u64()) {
            // Single-threaded daemon: holding the store lock across the inference is
            // uncontended and lets the persistent KV cache mutate in place.
            let mut map = rlm_stores().lock().unwrap();
            match map.get_mut(&h) {
                Some(sc) => blended_compose_and_answer(model, ctx, &sc.blocks, &mut sc.kv_cache, question, n_cand, k)
                    .map(|(a, picked, _)| (a, picked)),
                None => Err(format!("unknown context handle {h}")),
            }
        } else if let Some(secs) = payload.get("sections") {
            let bodies: Vec<String> = parse_sections(secs).into_iter().map(|(_, b)| b).collect();
            match BlockStore::from_texts(model, &bodies) {
                Ok(bs) => {
                    let mut cache = KvBlockCache::new(0); // one-shot inline: no caching
                    blended_compose_and_answer(model, ctx, &bs, &mut cache, question, n_cand, k)
                        .map(|(a, picked, _)| (a, picked))
                }
                Err(e) => Err(e),
            }
        } else {
            stream.write_all(&Frame::Error("rlm: query needs a \"handle\" or inline \"sections\"".into()).encode())?;
            return Ok(());
        };

    match result {
        Ok((answer, picked)) => {
            eprintln!("jaded: [rlm] query answered (compose blocks {picked:?})");
            stream.write_all(&Frame::Meta { model: model_name.to_owned() }.encode())?;
            stream.write_all(&Frame::Token(answer).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
        }
        Err(e) => {
            stream.write_all(&Frame::Error(format!("rlm: {e}")).encode())?;
        }
    }
    Ok(())
}

// ── Unit tests ──────────────────────────────────────────────────────────────
//
// These cover the PURE, model-free logic of this module: the upload payload parser.
// The compose engine itself needs a real model and is exercised by the selftests in
// `super::bench` + `examples/rlm_client.rs` on the box; its pure pieces (retrieval, KV
// cache LRU) are unit-tested in `super::compose`. This whole module is behind the
// Linux-only `inference` feature, so these run under `cargo test -p jaded --features
// inference` (not the macOS pure suite).
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sections_reads_summary_body_pairs() {
        let v = serde_json::json!([["Sum A", "Body A"], ["Sum B", "Body B"]]);
        assert_eq!(
            parse_sections(&v),
            vec![
                ("Sum A".to_string(), "Body A".to_string()),
                ("Sum B".to_string(), "Body B".to_string()),
            ]
        );
    }

    #[test]
    fn parse_sections_drops_malformed_entries() {
        // Entries that aren't [string, string] pairs are skipped; a non-array is empty.
        let v = serde_json::json!([["ok", "body"], ["missing body"], "not a pair", 7]);
        assert_eq!(parse_sections(&v), vec![("ok".to_string(), "body".to_string())]);
        assert!(parse_sections(&serde_json::json!({"not": "an array"})).is_empty());
    }
}
