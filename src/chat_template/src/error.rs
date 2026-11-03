//! Author: Joe Ricks
//! File: src/chat_template/src/error.rs
//! Description: Defines the errors dovata-chat-template returns.

use std::fmt;
use std::path::PathBuf;

/// Purpose: This enum lists the errors dovata-chat-template can return.
#[derive(Debug)]
#[non_exhaustive]
pub enum ChatError {
    /// A file could not be read.
    Io {
        /// The file.
        path: PathBuf,
        /// What the OS said.
        source: std::io::Error,
    },
    /// The GGUF header could not be parsed, or its metadata runs past the bytes read.
    Header(String),
    /// The GGUF carries no `tokenizer.chat_template`.
    MissingTemplate {
        /// Named templates the file does carry, under `tokenizer.chat_template.<name>`.
        /// Some models ship a `tool_use` or `rag` template beside the default.
        named: Vec<String>,
    },
    /// The GGUF carries no usable `tokenizer.ggml.tokens`, so there is no vocabulary to
    /// check the tokenizer against.
    MissingVocab(String),
    /// A special-token id in the header points past the end of the vocabulary.
    BadTokenId {
        /// The metadata key, such as `tokenizer.ggml.eos_token_id`.
        key: &'static str,
        /// The id it holds.
        id: u64,
        /// How many tokens the vocabulary has.
        vocab: usize,
    },
    /// `tokenizer.json` could not be loaded, or encoding or decoding failed.
    Tokenizer(String),
    /// `tokenizer.json` and the GGUF do not describe the same vocabulary.
    VocabMismatch {
        /// Tokens in the GGUF's `tokenizer.ggml.tokens`.
        gguf: usize,
        /// One past the highest id `tokenizer.json` defines.
        tokenizer: usize,
        /// How many ids disagree in all.
        count: usize,
        /// The first few that disagree.
        examples: Vec<Mismatch>,
    },
    /// The template does not compile, or failed while rendering.
    Template(minijinja::Error),
    /// The template refused the conversation through `raise_exception`, for example
    /// because roles do not alternate. The message is the template's own.
    Raised(String),
    /// A message or the `extra` value has a shape the template context cannot take.
    BadInput(String),
    /// A Hugging Face Hub request failed. Only `Chat::from_hub` returns this.
    Hub(String),
}

/// Purpose: This struct holds one token id where `tokenizer.json` and the GGUF disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The token id.
    pub id: u32,
    /// The GGUF's string for it, or `None` when the GGUF's vocabulary ends before it.
    pub gguf: Option<String>,
    /// `tokenizer.json`'s string for it, or `None` when it has no such id.
    pub tokenizer: Option<String>,
}

impl ChatError {
    /// Purpose: This function converts a template render error into a `ChatError`.
    ///
    /// Inputs:
    /// - `err`: the error minijinja returned.
    ///
    /// Outputs: `ChatError::Raised` if the template called `raise_exception`, else
    /// `ChatError::Template`.
    pub(crate) fn from_render(
        err: minijinja::Error,
    ) -> ChatError
    {
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&err);
        while let Some(e) = source {
            if let Some(raised) = e.downcast_ref::<crate::template::Raised>() {
                return ChatError::Raised(raised.0.clone());
            }
            source = e.source();
        }
        ChatError::Template(err)
    }
}

impl fmt::Display for ChatError {
    /// Purpose: This function formats dovata-chat-template errors.
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
            ChatError::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            ChatError::Header(detail) => write!(f, "cannot read the GGUF header: {detail}"),
            ChatError::MissingTemplate { named } => {
                write!(f, "the GGUF has no tokenizer.chat_template")?;
                if !named.is_empty() {
                    write!(f, ", only the named templates {}", named.join(", "))?;
                }
                write!(f, "; pass a template to Chat::new, taken from the model card or tokenizer_config.json")
            }
            ChatError::MissingVocab(detail) => write!(
                f,
                "the GGUF has no usable tokenizer.ggml.tokens ({detail}), so tokenizer.json cannot be \
                 checked against it; use Chat::new to skip the check"
            ),
            ChatError::BadTokenId { key, id, vocab } => {
                write!(f, "{key} is {id}, but the vocabulary has only {vocab} tokens")
            }
            ChatError::Tokenizer(detail) => write!(f, "tokenizer: {detail}"),
            ChatError::VocabMismatch { gguf, tokenizer, count, examples } => {
                write!(
                    f,
                    "tokenizer.json does not match the GGUF: the GGUF has {gguf} tokens, tokenizer.json \
                     defines ids up to {tokenizer}, and {count} ids disagree"
                )?;
                let show = |s: &Option<String>| s.as_ref().map_or_else(|| "no such id".to_owned(), |s| format!("{s:?}"));
                for m in examples {
                    write!(f, "; id {}: GGUF {}, tokenizer.json {}", m.id, show(&m.gguf), show(&m.tokenizer))?;
                }
                write!(f, ". Token ids from this pair would be wrong; use the tokenizer.json published with this model")
            }
            // minijinja's message already ends with the line, as `(in chat:12)`.
            ChatError::Template(e) => write!(f, "chat template: {e}"),
            ChatError::Raised(msg) => write!(f, "the chat template refused the conversation: {msg}"),
            ChatError::BadInput(detail) => write!(f, "bad render input: {detail}"),
            ChatError::Hub(detail) => write!(f, "Hugging Face Hub: {detail}"),
        }
    }
}

impl std::error::Error for ChatError {
    /// Purpose: This function returns the lower-level error inside a dovata-chat-template error.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the I/O or minijinja error it wraps, or `None`.
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)>
    {
        match self {
            ChatError::Io { source, .. } => Some(source),
            ChatError::Template(e) => Some(e),
            _ => None,
        }
    }
}
