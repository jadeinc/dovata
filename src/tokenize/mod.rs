//! Author: Joe Ricks
//! File: src/tokenize/mod.rs
//! Description: Describes a model's tokenizer for its plan, finds its `tokenizer.json`,
//! and runs it: text to token ids and back.

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[cfg(feature = "tokenize")]
use crate::gguf::{Gguf, Value};

mod find;
#[cfg(feature = "tokenize")]
mod engine;
#[cfg(feature = "tokenize")]
mod vocab;

pub use find::{find, TOKENIZER_JSON};
#[cfg(feature = "tokenize")]
pub use engine::{describe, Stream, Tokenizer};
#[cfg(feature = "tokenize")]
pub use vocab::check_vocab;

/// The GGUF key naming the tokenizer's kind, such as `gpt2` for byte-level BPE.
#[cfg(feature = "tokenize")]
const KEY_MODEL: &str = "tokenizer.ggml.model";
/// The GGUF key holding the vocabulary, one string per token id.
#[cfg(feature = "tokenize")]
pub(crate) const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
/// The GGUF key holding each token's type, from gguf-py's `TokenType`.
#[cfg(feature = "tokenize")]
pub(crate) const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
/// The GGUF key holding the BOS token's id.
#[cfg(feature = "tokenize")]
const KEY_BOS: &str = "tokenizer.ggml.bos_token_id";
/// The GGUF key holding the EOS token's id.
#[cfg(feature = "tokenize")]
const KEY_EOS: &str = "tokenizer.ggml.eos_token_id";
/// The GGUF key holding the end-of-turn token's id, which llama.cpp's converter writes
/// for models that have one apart from EOS.
#[cfg(feature = "tokenize")]
const KEY_EOT: &str = "tokenizer.ggml.eot_token_id";
/// The GGUF key holding the end-of-message token's id, for tool calls in Llama 3.1.
#[cfg(feature = "tokenize")]
const KEY_EOM: &str = "tokenizer.ggml.eom_token_id";
/// The GGUF key holding the base model's Hugging Face link, which Unsloth's files carry.
pub(crate) const KEY_BASE_URL: &str = "general.base_model.0.repo_url";

/// Purpose: This struct holds a model's tokenizer as its plan carries it: what a program
/// needs to turn text into the model's token ids and back, with nothing read from the
/// model's files at run time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerPlan {
    /// The GGUF's `tokenizer.ggml.model`: `gpt2` for byte-level BPE, as Qwen, Llama 3
    /// and DeepSeek use, or `llama` for SentencePiece.
    pub kind: String,
    /// The tokens the tokenizer defines: one past its highest id.
    pub tokens: u32,
    /// The rows of the model's token embedding table. Every id a program gives the model
    /// must be below this. It can be more than `tokens`, since converters pad the
    /// embedding table.
    pub embedding_rows: u32,
    /// The BOS token's id, when the model has one. Qwen models have none.
    pub bos: Option<u32>,
    /// The EOS token's id.
    pub eos: Option<u32>,
    /// The ids that end the model's turn: EOS, and the end-of-turn and end-of-message
    /// tokens the GGUF names. A program stops generating at any of them.
    pub stop: Vec<u32>,
    /// The SHA-256 of `definition`, in lowercase hex, so two plans' tokenizers can be
    /// compared without the text.
    pub sha256: String,
    /// The model's `tokenizer.json`, exactly as the file has it. [`Tokenizer`] runs it.
    /// It is kept as text, not as parsed JSON, so its bytes, and the plan's table hash
    /// over them, never depend on how a JSON library orders keys.
    pub definition: String,
}

/// Purpose: This struct holds one token id on which a GGUF and a `tokenizer.json`
/// disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The token id.
    pub id: u32,
    /// The GGUF's string for it, or `None` when the GGUF's vocabulary ends before it.
    pub gguf: Option<String>,
    /// `tokenizer.json`'s string for it, or `None` when it has no such id.
    pub tokenizer: Option<String>,
}

/// Purpose: This struct holds the result of a failed vocabulary check: how far the two
/// vocabularies go and the first ids that disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabMismatch {
    /// Tokens in the GGUF's `tokenizer.ggml.tokens`.
    pub gguf: usize,
    /// One past the highest id `tokenizer.json` defines.
    pub tokenizer: usize,
    /// How many ids disagree in all.
    pub count: usize,
    /// The first few that disagree.
    pub examples: Vec<Mismatch>,
}

/// Purpose: This enum lists the errors describing, finding or running a tokenizer can
/// return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizeError {
    /// This build of dovata cannot run a tokenizer: the `tokenize` feature is off.
    NotBuilt,
    /// No `tokenizer.json` was found for the model.
    NotFound(String),
    /// A `tokenizer.json`, or a GGUF's tokenizer metadata, could not be read.
    Unreadable(String),
    /// The `tokenizer.json` and the GGUF have different vocabularies.
    Vocab(VocabMismatch),
    /// A GGUF key names a token id past the end of the vocabulary.
    BadTokenId {
        /// The key.
        key: &'static str,
        /// The id it holds.
        id: u64,
        /// The vocabulary's length.
        tokens: usize,
    },
    /// Fetching a `tokenizer.json` from the Hub failed.
    Fetch(String),
    /// Encoding or decoding failed, or gave an id the model has no row for.
    Run(String),
}

impl fmt::Display for TokenizeError {
    /// Purpose: This function formats tokenizer errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        match self {
            TokenizeError::NotBuilt => write!(f, "this build of dovata cannot tokenize: it was compiled without the `tokenize` cargo feature"),
            TokenizeError::NotFound(m) | TokenizeError::Unreadable(m) | TokenizeError::Fetch(m) | TokenizeError::Run(m) => write!(f, "{m}"),
            TokenizeError::Vocab(v) => {
                write!(
                    f,
                    "tokenizer.json does not match the GGUF: the GGUF has {} tokens, tokenizer.json defines ids up to {}, and {} ids disagree",
                    v.gguf, v.tokenizer, v.count
                )?;
                let show = |s: &Option<String>| s.as_ref().map_or_else(|| "no such id".to_owned(), |s| format!("{s:?}"));
                for m in &v.examples {
                    write!(f, "; id {}: GGUF {}, tokenizer.json {}", m.id, show(&m.gguf), show(&m.tokenizer))?;
                }
                Ok(())
            }
            TokenizeError::BadTokenId { key, id, tokens } => write!(f, "{key} is {id}, past the end of a vocabulary of {tokens} tokens"),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// Purpose: This function reads a GGUF's vocabulary.
///
/// Inputs:
/// - `gguf`: the parsed GGUF header.
///
/// Outputs: one string per token id, or [`TokenizeError::Unreadable`] when the header
/// has no vocabulary.
#[cfg(feature = "tokenize")]
pub(crate) fn gguf_tokens(
    gguf: &Gguf,
) -> Result<Vec<&str>, TokenizeError>
{
    match gguf.get(KEY_TOKENS) {
        Some(Value::Array(items)) if !items.is_empty() => items
            .iter()
            .map(Value::as_str)
            .collect::<Option<Vec<&str>>>()
            .ok_or_else(|| TokenizeError::Unreadable(format!("{KEY_TOKENS} holds a value that is not a string"))),
        Some(_) => Err(TokenizeError::Unreadable(format!("{KEY_TOKENS} is not a list of tokens"))),
        None => Err(TokenizeError::Unreadable(format!("the GGUF has no vocabulary: {KEY_TOKENS} is absent"))),
    }
}

/// Purpose: This function reads a GGUF's token types, which mark the padding entries a
/// converter appends.
///
/// Inputs:
/// - `gguf`: the parsed GGUF header.
///
/// Outputs: one type per token id, or `None` when the header has none.
#[cfg(feature = "tokenize")]
pub(crate) fn gguf_token_types(
    gguf: &Gguf,
) -> Option<Vec<u64>>
{
    match gguf.get(KEY_TOKEN_TYPE) {
        Some(Value::Array(items)) => items.iter().map(Value::as_u64).collect(),
        _ => None,
    }
}

/// Purpose: This struct holds a GGUF's tokenizer kind and special token ids.
#[cfg(feature = "tokenize")]
pub(crate) struct Specials {
    /// `tokenizer.ggml.model`.
    pub(crate) kind: String,
    /// The BOS id, if any.
    pub(crate) bos: Option<u32>,
    /// The EOS id, if any.
    pub(crate) eos: Option<u32>,
    /// EOS, end-of-turn and end-of-message ids, each once.
    pub(crate) stop: Vec<u32>,
}

/// Purpose: This function reads the GGUF's tokenizer kind and special token ids.
///
/// Inputs:
/// - `gguf`: the parsed GGUF header.
/// - `tokens`: the vocabulary's length, which every id must be below.
///
/// Outputs: the [`Specials`], or an error for a missing kind or an id past the
/// vocabulary.
#[cfg(feature = "tokenize")]
pub(crate) fn specials(
    gguf: &Gguf,
    tokens: usize,
) -> Result<Specials, TokenizeError>
{
    let kind = gguf.string(KEY_MODEL).ok_or_else(|| TokenizeError::Unreadable(format!("the GGUF does not say what tokenizer it uses: {KEY_MODEL} is absent")))?;
    let id = |key: &'static str| -> Result<Option<u32>, TokenizeError> {
        match gguf.u64(key) {
            None => Ok(None),
            Some(id) => u32::try_from(id).ok().filter(|i| (*i as usize) < tokens).map(Some).ok_or(TokenizeError::BadTokenId { key, id, tokens }),
        }
    };
    let (bos, eos) = (id(KEY_BOS)?, id(KEY_EOS)?);
    let mut stop = Vec::new();
    for key in [KEY_EOS, KEY_EOT, KEY_EOM] {
        if let Some(i) = id(key)? {
            if !stop.contains(&i) {
                stop.push(i);
            }
        }
    }
    Ok(Specials { kind: kind.to_owned(), bos, eos, stop })
}

/// Purpose: This function computes the SHA-256 of a text.
///
/// Inputs:
/// - `text`: the text.
///
/// Outputs: the hash in lowercase hex.
pub fn sha256_hex(
    text: &str,
) -> String
{
    Sha256::digest(text.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// Purpose: This function returns `NotBuilt` in a build without the `tokenize` feature.
///
/// Inputs:
/// - `model`: the model; unused.
/// - `definition`: the `tokenizer.json` text; unused.
///
/// Outputs: always [`TokenizeError::NotBuilt`].
#[cfg(not(feature = "tokenize"))]
pub fn describe(
    model: &crate::load::Model,
    definition: String,
) -> Result<TokenizerPlan, TokenizeError>
{
    let _ = (model, definition);
    Err(TokenizeError::NotBuilt)
}

#[cfg(all(test, feature = "tokenize"))]
mod tests;

#[cfg(all(test, not(feature = "tokenize")))]
mod not_built {
    #[test]
    fn the_refusal_names_the_feature()
    {
        assert!(crate::tokenize::TokenizeError::NotBuilt.to_string().contains("`tokenize` cargo feature"));
    }
}
