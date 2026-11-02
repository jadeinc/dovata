//! Author: Joe Ricks
//! File: src/chat/src/header.rs
//! Description: Reads the chat template, token list and special tokens from a GGUF
//! file's header.

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

/// Purpose: This function reads the header of a GGUF file on disk.
///
/// Inputs:
/// - `path`: the GGUF file, whole or just its first megabytes.
///
/// Outputs: the parsed header, or an error if the file cannot be read or parsed.
pub(crate) fn read(
    path: &Path,
) -> Result<Gguf, ChatError>
{
    let io = |source| ChatError::Io { path: path.to_owned(), source };
    let mut file = File::open(path).map_err(io)?;
    grow(|buf, want| {
        let need = want.saturating_sub(buf.len() as u64);
        let before = buf.len();
        (&mut file).take(need).read_to_end(buf).map_err(io)?;
        Ok(((buf.len() - before) as u64) < need)
    })
}

/// Purpose: This function parses a GGUF header, reading more of the file until the
/// header is complete.
///
/// Inputs:
/// - `more`: a reader that adds bytes to a buffer and says if the file ended.
///
/// Outputs: the parsed header, or an error if it does not parse within dovata's
/// `HEADER_BUDGET`.
pub(crate) fn grow(
    mut more: impl FnMut(&mut Vec<u8>, u64) -> Result<bool, ChatError>,
) -> Result<Gguf, ChatError>
{
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

/// Purpose: This struct holds what a `Chat` needs from a GGUF header.
pub(crate) struct Pieces<'a> {
    pub template: String,
    pub tokens: Vec<&'a str>,
    pub token_types: Option<Vec<u64>>,
    pub bos: Option<String>,
    pub eos: Option<String>,
}

impl<'a> Pieces<'a> {
    /// Purpose: This function takes the chat template, token list, token types and BOS
    /// and EOS strings out of a GGUF header.
    ///
    /// Inputs:
    /// - `gguf`: the parsed GGUF header.
    ///
    /// Outputs: those pieces, or an error if the template or the token list is missing
    /// or a token id is out of range.
    pub(crate) fn of(
        gguf: &'a Gguf,
    ) -> Result<Pieces<'a>, ChatError>
    {
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

/// Purpose: This function looks up the text of the BOS or EOS token.
///
/// Inputs:
/// - `gguf`: the parsed GGUF header.
/// - `key`: the header key that holds the token's id.
/// - `tokens`: the header's token list.
///
/// Outputs: the token's text, `None` if the key is absent, or an error if the id is out
/// of range.
fn special(
    gguf: &Gguf,
    key: &'static str,
    tokens: &[&str],
) -> Result<Option<String>, ChatError>
{
    let Some(id) = gguf.u64(key) else {
        return Ok(None);
    };
    usize::try_from(id)
        .ok()
        .and_then(|i| tokens.get(i))
        .map(|s| Some((*s).to_owned()))
        .ok_or(ChatError::BadTokenId { key, id, vocab: tokens.len() })
}
