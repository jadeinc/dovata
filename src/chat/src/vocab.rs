//! Checking that a `tokenizer.json` and a GGUF describe the same vocabulary.
//!
//! # Why this is a refusal and not a warning
//!
//! The model's embedding table is indexed by token id. A tokenizer from a different
//! model, or from a different revision of the same one, still produces ids, and they
//! still index rows of the table. Nothing fails; the model just reads the wrong tokens.
//! Qwen2.5 and Qwen3 share most of their vocabulary, for instance, so a mixed-up pair
//! is nearly right, which is the worst kind of wrong. Constructors refuse it.
//!
//! # What "the same" means
//!
//! Every id the GGUF has is compared, not a sample. It costs a few milliseconds for a
//! 150k vocabulary, and a sample can miss a pair that differs only in its added tokens,
//! which is how two revisions of a model most often differ.
//!
//! The sizes are allowed to differ in one direction. `convert_hf_to_gguf.py` sizes the
//! GGUF vocabulary to the model's embedding table, `vocab_size` in `config.json`, which
//! is often larger than the tokenizer: Qwen3's table has 151,936 rows and its tokenizer
//! 151,665 tokens. The converter fills the gap with `[PAD<id>]` entries of type UNUSED,
//! and those are accepted. A tokenizer with ids the GGUF lacks is not, since the model
//! has no row for them.
//!
//! The converter also rewrites some added tokens on the way in: one whose `normalized`
//! flag is off goes through an encode and decode round trip, and a user-defined one has
//! SentencePiece's `▁` replaced with a space. An added token that matches only after the
//! same rewrite is accepted.

use std::collections::HashSet;

use tokenizers::Tokenizer;

use crate::{ChatError, Mismatch};

/// `tokenizer.ggml.token_type` values, from gguf-py's `TokenType`.
const TYPE_USER_DEFINED: u64 = 4;
const TYPE_UNUSED: u64 = 5;

/// How many disagreeing ids an error lists. Enough to see the pattern, such as an
/// offset of one, without printing a vocabulary.
const EXAMPLES: usize = 5;

/// Refuse a tokenizer whose vocabulary disagrees with a GGUF's.
///
/// `gguf_tokens` is `tokenizer.ggml.tokens` and `gguf_types` is
/// `tokenizer.ggml.token_type`, when the file has it. The constructors that read a GGUF
/// call this; it is public for a caller who builds a [`Chat`](crate::Chat) with
/// [`Chat::new`](crate::Chat::new) and still has a GGUF to check against.
pub fn check(gguf_tokens: &[&str], gguf_types: Option<&[u64]>, tokenizer: &Tokenizer) -> Result<(), ChatError> {
    let added: HashSet<u32> = tokenizer.get_added_tokens_decoder().into_keys().collect();
    // One past the highest id, rather than the token count: a tokenizer's ids can have
    // gaps, and an id past the GGUF's end is a mismatch wherever it sits.
    let span: u32 = tokenizer.get_vocab(true).values().max().map_or(0, |&m| m.saturating_add(1));
    // Ids are u32 on both sides, so counting in u32 covers every id either can name.
    let end = u32::try_from(gguf_tokens.len()).unwrap_or(u32::MAX).max(span);

    let mut count = 0;
    let mut examples = Vec::new();
    for id in 0..end {
        let gguf = gguf_tokens.get(id as usize);
        let ours = tokenizer.id_to_token(id);
        let ok = match (gguf, &ours) {
            (Some(g), Some(t)) => agrees(id, g, t, gguf_types, &added, tokenizer),
            (Some(g), None) => is_padding(id, g, gguf_types),
            (None, _) => false,
        };
        if !ok {
            count += 1;
            if examples.len() < EXAMPLES {
                examples.push(Mismatch { id, gguf: gguf.map(|s| (*s).to_owned()), tokenizer: ours });
            }
        }
    }

    if count == 0 {
        Ok(())
    } else {
        Err(ChatError::VocabMismatch { gguf: gguf_tokens.len(), tokenizer: span as usize, count, examples })
    }
}

/// Whether the GGUF's string for `id` is the tokenizer's, allowing for what the
/// converter does to added tokens. The rewrites are only tried on a mismatch, so the
/// common case is one string compare.
fn agrees(id: u32, gguf: &str, ours: &str, types: Option<&[u64]>, added: &HashSet<u32>, tokenizer: &Tokenizer) -> bool {
    if gguf == ours {
        return true;
    }
    if !added.contains(&id) {
        return false;
    }
    if type_of(types, id) == Some(TYPE_USER_DEFINED) && ours.replace('\u{2581}', " ") == gguf {
        return true;
    }
    tokenizer
        .encode_fast(ours, false)
        .ok()
        .and_then(|e| tokenizer.decode(e.get_ids(), false).ok())
        .is_some_and(|round_trip| round_trip == gguf)
}

/// Whether a GGUF entry the tokenizer has no id for is converter padding.
fn is_padding(id: u32, gguf: &str, types: Option<&[u64]>) -> bool {
    match type_of(types, id) {
        Some(t) => t == TYPE_UNUSED,
        // A file without token types: fall back to the name the converter writes.
        None => gguf == format!("[PAD{id}]"),
    }
}

fn type_of(types: Option<&[u64]>, id: u32) -> Option<u64> {
    types?.get(id as usize).copied()
}
