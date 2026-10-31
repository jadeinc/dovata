//! Reading the chat template, the vocabulary and the special tokens out of a GGUF header.
//!
//! Everything this package needs from the GGUF is metadata, which sits at the front of
//! the file ahead of the tensor index. So the reader starts small and doubles until the
//! metadata parses, capped at dovata's `HEADER_BUDGET`. For Qwen3-Coder-30B the metadata,
//! with its 151,936-token vocabulary and as many merges, ends near 6 MB of an 18 GB file.
//! The same loop serves a local file and a range-fetched one from the Hub.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use dovata::gguf::{Gguf, Value};
use dovata::load::preflight::HEADER_BUDGET;

use crate::ChatError;

/// The first read. Enough for every model checked so far, so the loop rarely runs twice.
pub(crate) const FIRST_READ: u64 = 8 * 1024 * 1024;

const KEY_TEMPLATE: &str = "tokenizer.chat_template";
const KEY_TOKENS: &str = "tokenizer.ggml.tokens";
const KEY_TOKEN_TYPE: &str = "tokenizer.ggml.token_type";
const KEY_BOS: &str = "tokenizer.ggml.bos_token_id";
const KEY_EOS: &str = "tokenizer.ggml.eos_token_id";

/// Read a local GGUF's header.
pub(crate) fn read(path: &Path) -> Result<Gguf, ChatError> {
    let io = |source| ChatError::Io { path: path.to_owned(), source };
    let mut file = File::open(path).map_err(io)?;
    grow(|buf, want| {
        let need = want.saturating_sub(buf.len() as u64);
        let before = buf.len();
        (&mut file).take(need).read_to_end(buf).map_err(io)?;
        Ok(((buf.len() - before) as u64) < need)
    })
}

/// Parse a GGUF header from a source read in growing prefixes.
///
/// `more(buf, want)` extends `buf` toward `want` bytes in all and returns whether the
/// source ended first. Any parse error before the end of the source and before the
/// budget means "read more": a truncated buffer can fail as `Truncated`, or as
/// `Implausible` when a long array claims more elements than the bytes so far hold.
/// Only the error from the last, largest read is reported.
pub(crate) fn grow(mut more: impl FnMut(&mut Vec<u8>, u64) -> Result<bool, ChatError>) -> Result<Gguf, ChatError> {
    let mut buf = Vec::new();
    let mut want = FIRST_READ.min(HEADER_BUDGET);
    loop {
        let ended = more(&mut buf, want)?;
        match Gguf::parse(&buf) {
            Ok(gguf) => return Ok(gguf),
            Err(_) if !ended && want < HEADER_BUDGET => want = (want * 2).min(HEADER_BUDGET),
            Err(e) if ended => return Err(ChatError::Header(e.to_string())),
            Err(e) => {
                return Err(ChatError::Header(format!(
                    "{e}; the metadata does not end within the first {HEADER_BUDGET} bytes"
                )))
            }
        }
    }
}

/// What a [`Chat`](crate::Chat) takes from a header, borrowed where it is large.
pub(crate) struct Pieces<'a> {
    pub template: String,
    pub tokens: Vec<&'a str>,
    pub token_types: Option<Vec<u64>>,
    pub bos: Option<String>,
    pub eos: Option<String>,
}

impl<'a> Pieces<'a> {
    pub(crate) fn of(gguf: &'a Gguf) -> Result<Pieces<'a>, ChatError> {
        let template = match gguf.string(KEY_TEMPLATE) {
            Some(t) => t.to_owned(),
            None => {
                let prefix = format!("{KEY_TEMPLATE}.");
                let named = gguf
                    .metadata
                    .iter()
                    .filter_map(|(k, _)| k.strip_prefix(&prefix).map(str::to_owned))
                    .collect();
                return Err(ChatError::MissingTemplate { named });
            }
        };

        let tokens = match gguf.get(KEY_TOKENS) {
            Some(Value::Array(items)) => items
                .iter()
                .map(Value::as_str)
                .collect::<Option<Vec<&str>>>()
                .ok_or_else(|| ChatError::MissingVocab(format!("{KEY_TOKENS} holds a non-string")))?,
            Some(other) => {
                return Err(ChatError::MissingVocab(format!("{KEY_TOKENS} is a {}, not an array", other.type_name())))
            }
            None => return Err(ChatError::MissingVocab(format!("{KEY_TOKENS} is absent"))),
        };
        if tokens.is_empty() {
            return Err(ChatError::MissingVocab(format!("{KEY_TOKENS} is empty")));
        }

        // Optional: only used to recognise the padding entries a converter appends.
        let token_types = match gguf.get(KEY_TOKEN_TYPE) {
            Some(Value::Array(items)) => items.iter().map(Value::as_u64).collect::<Option<Vec<u64>>>(),
            _ => None,
        };

        let bos = special(gguf, KEY_BOS, &tokens)?;
        let eos = special(gguf, KEY_EOS, &tokens)?;
        Ok(Pieces { template, tokens, token_types, bos, eos })
    }
}

/// A special token's string: the id under `key`, looked up in the vocabulary. Absent is
/// fine, since Qwen ships no BOS. An id past the end is not: the header is lying about
/// something, and the template would print the wrong marker.
fn special(gguf: &Gguf, key: &'static str, tokens: &[&str]) -> Result<Option<String>, ChatError> {
    let Some(id) = gguf.u64(key) else {
        return Ok(None);
    };
    usize::try_from(id)
        .ok()
        .and_then(|i| tokens.get(i))
        .map(|s| Some((*s).to_owned()))
        .ok_or(ChatError::BadTokenId { key, id, vocab: tokens.len() })
}
