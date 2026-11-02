//! Author: Joe Ricks
//! File: src/chat_template/src/vocab.rs
//! Description: Checks that a `tokenizer.json` and a GGUF file have the same vocabulary.

use std::collections::HashSet;

use tokenizers::Tokenizer;

use crate::{ChatError, Mismatch};

/// `tokenizer.ggml.token_type` values, from gguf-py's `TokenType`.
const TYPE_USER_DEFINED: u64 = 4;
const TYPE_UNUSED: u64 = 5;

/// How many disagreeing ids an error lists. Enough to see the pattern, such as an
/// offset of one, without printing a vocabulary.
const EXAMPLES: usize = 5;

/// Purpose: This function checks that a tokenizer and a GGUF file give every token id
/// the same text.
///
/// Inputs:
/// - `gguf_tokens`: the GGUF's token list.
/// - `gguf_types`: the GGUF's token types, if it has them.
/// - `tokenizer`: the tokenizer from `tokenizer.json`.
///
/// Outputs: `Ok(())`, or a `ChatError::VocabMismatch` listing the ids that differ.
pub fn check(
    gguf_tokens: &[&str],
    gguf_types: Option<&[u64]>,
    tokenizer: &Tokenizer,
) -> Result<(), ChatError>
{
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

/// Purpose: This function tells whether the GGUF and the tokenizer agree on one token
/// id.
///
/// Inputs:
/// - `id`: the token id.
/// - `gguf`: the GGUF's text for it.
/// - `ours`: the tokenizer's text for it.
/// - `types`: the GGUF's token types, if any.
/// - `added`: the ids of the tokenizer's added tokens.
/// - `tokenizer`: the tokenizer.
///
/// Outputs: `true` if they agree.
fn agrees(
    id: u32,
    gguf: &str,
    ours: &str,
    types: Option<&[u64]>,
    added: &HashSet<u32>,
    tokenizer: &Tokenizer,
) -> bool
{
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

/// Purpose: This function tells whether a GGUF token is padding added by the converter.
///
/// Inputs:
/// - `id`: the token id.
/// - `gguf`: the GGUF's text for it.
/// - `types`: the GGUF's token types, if any.
///
/// Outputs: `true` for padding.
fn is_padding(
    id: u32,
    gguf: &str,
    types: Option<&[u64]>,
) -> bool
{
    match type_of(types, id) {
        Some(t) => t == TYPE_UNUSED,
        // A file without token types: fall back to the name the converter writes.
        None => gguf == format!("[PAD{id}]"),
    }
}

/// Purpose: This function looks up one token's type in the GGUF's token types.
///
/// Inputs:
/// - `types`: the GGUF's token types, if any.
/// - `id`: the token id.
///
/// Outputs: the type, or `None`.
fn type_of(
    types: Option<&[u64]>,
    id: u32,
) -> Option<u64>
{
    types?.get(id as usize).copied()
}
