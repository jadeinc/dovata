//! Author: Joe Ricks
//! File: src/chat/src/tests/vocab.rs
//! Description: Tests the vocabulary check.

use super::fixture::{self, Header, WORDS};
use crate::{vocab, Chat, ChatError, Mismatch};

/// Purpose: This function builds a `Chat` from a test header and the test tokenizer.
///
/// Inputs:
/// - `header`: the test header.
///
/// Outputs: the `Chat`, or the error the vocabulary check returned.
fn build(header: Header) -> Result<Chat, ChatError> {
    Chat::from_gguf(&header.parse(), fixture::tokenizer())
}

#[test]
fn vocab_mismatch_refused() {
    // Two ordinary words trade places: same size, same strings, wrong ids. This is the
    // shape of a sibling model's tokenizer, and the reason the check compares strings.
    let mut swapped = WORDS.to_vec();
    swapped.swap(4, 5);
    let err = build(Header::chatml().tokens(&swapped)).unwrap_err();
    let ChatError::VocabMismatch { gguf, tokenizer, count, examples } = &err else {
        panic!("expected VocabMismatch, got {err:?}");
    };
    assert_eq!((*gguf, *tokenizer, *count), (12, 12, 2));
    assert_eq!(
        examples[0],
        Mismatch { id: 4, gguf: Some("world".into()), tokenizer: Some("hello".into()) }
    );
    assert!(err.to_string().contains("id 4: GGUF \"world\", tokenizer.json \"hello\""), "{err}");
}

#[test]
fn special_token_mismatch_refused() {
    let mut renamed = WORDS.to_vec();
    renamed[2] = "<|endoftext|>";
    let err = build(Header::chatml().tokens(&renamed)).unwrap_err();
    assert!(matches!(&err, ChatError::VocabMismatch { count: 1, examples, .. } if examples[0].id == 2), "{err:?}");
}

#[test]
fn shorter_gguf_vocab_refused() {
    // The tokenizer has ids the model has no embedding row for.
    let err = build(Header::chatml().tokens(&WORDS[..10])).unwrap_err();
    let ChatError::VocabMismatch { gguf: 10, tokenizer: 12, count: 2, examples } = &err else {
        panic!("expected VocabMismatch, got {err:?}");
    };
    assert_eq!(examples[0], Mismatch { id: 10, gguf: None, tokenizer: Some("brief.".into()) });
}

#[test]
fn padded_gguf_vocab_accepted() {
    // convert_hf_to_gguf.py sizes the vocabulary to the embedding table and fills the
    // tail with `[PAD<id>]` of type UNUSED, as Qwen3's 151,936 tokens over 151,665 show.
    let mut padded: Vec<String> = WORDS.iter().map(|s| (*s).to_owned()).collect();
    padded.extend((12..16).map(|i| format!("[PAD{i}]")));
    let tokens: Vec<&str> = padded.iter().map(String::as_str).collect();
    let mut types = vec![3, 3, 3, 3];
    types.extend([1; 8]);
    types.extend([5; 4]);

    assert!(build(Header::chatml().tokens(&tokens).token_types(&types)).is_ok());
    // Without token types, the name alone is enough.
    assert!(build(Header::chatml().tokens(&tokens)).is_ok());
}

#[test]
fn extra_gguf_tokens_that_are_not_padding_refused() {
    let mut extended = WORDS.to_vec();
    extended.push("<|extra|>");
    let err = build(Header::chatml().tokens(&extended)).unwrap_err();
    assert!(
        matches!(&err, ChatError::VocabMismatch { count: 1, examples, .. }
            if examples[0] == Mismatch { id: 12, gguf: Some("<|extra|>".into()), tokenizer: None }),
        "{err:?}"
    );
    // Typed as UNUSED, the same entry is padding whatever it is called.
    let mut types = vec![1; 12];
    types.push(5);
    assert!(build(Header::chatml().tokens(&extended).token_types(&types)).is_ok());
}

#[test]
fn check_is_callable_on_its_own() {
    // For a caller who built with Chat::new and still has a GGUF.
    let tokenizer = fixture::tokenizer();
    assert!(vocab::check(WORDS, None, &tokenizer).is_ok());
    assert!(vocab::check(&WORDS[1..], None, &tokenizer).is_err());
}

#[test]
fn missing_vocab_is_a_clear_error() {
    let err = build(Header::chatml().without("tokenizer.ggml.tokens")).unwrap_err();
    assert!(matches!(err, ChatError::MissingVocab(_)), "{err:?}");
    assert!(err.to_string().contains("Chat::new"), "{err}");
}
