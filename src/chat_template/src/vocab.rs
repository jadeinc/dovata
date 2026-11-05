//! Author: Joe Ricks
//! File: src/chat_template/src/vocab.rs
//! Description: Checks that a `tokenizer.json` and a GGUF file have the same vocabulary.

use tokenizers::Tokenizer;

use crate::ChatError;

/// Purpose: This function checks that a tokenizer and a GGUF file give every token id
/// the same text. The check is dovata's, `dovata::tokenize::check_vocab`, which dovata
/// also runs when a plan takes a tokenizer; its README explains the rules.
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
    dovata::tokenize::check_vocab(gguf_tokens, gguf_types, tokenizer)
        .map_err(|v| ChatError::VocabMismatch { gguf: v.gguf, tokenizer: v.tokenizer, count: v.count, examples: v.examples })
}
