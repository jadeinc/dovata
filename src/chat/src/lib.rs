//! dovata-chat: turning a conversation into the prompt string and the token ids a model
//! expects, for a model planned with dovata.
//!
//! # Why it is a separate, optional package
//!
//! dovata plans where a model's tensors live. It never runs the model, so it has no use
//! for a tokenizer and does not depend on one. A Jade program that runs the model still
//! has to build its prompts, and that is where a mistake is quiet: a prompt with the
//! wrong turn markers, or token ids from the wrong vocabulary, still runs. It only
//! answers worse. Up to 2.x the engine did all of this itself. 3.0 moves it here, so it
//! is help a Jade user can take when they want it, and a user who would rather read the
//! model card and write the prompt string by hand loses nothing.
//!
//! It is Rust rather than a Jade package because the Jade language is about to change,
//! and the two hard parts, the tokenizer and the template engine, are Rust crates
//! already.
//!
//! # Where each piece comes from
//!
//! * The chat template and the BOS and EOS strings come from the GGUF header:
//!   `tokenizer.chat_template`, and `tokenizer.ggml.bos_token_id` and
//!   `tokenizer.ggml.eos_token_id` looked up in `tokenizer.ggml.tokens`. The GGUF is the
//!   file the program runs, so its template is the one that goes with the weights.
//! * The tokenizer is the model's `tokenizer.json`, run by Hugging Face's `tokenizers`
//!   crate. llama.cpp tokenizes from the GGUF's own vocabulary with its own code. This
//!   package does not reimplement that; it uses the reference tokenizer the model was
//!   published with.
//!
//! Two files means two chances to pair the wrong ones. A `tokenizer.json` from a sibling
//! model gives plausible token ids that are wrong, with no error anywhere, so every
//! constructor that has both files checks one against the other. See [`vocab`].
//!
//! # Templates
//!
//! Chat templates are Jinja written for Python's `jinja2`, the way Hugging Face's
//! `transformers` renders them, and models are trained on that output. This package
//! renders them with `minijinja`, set up the way `transformers` sets up `jinja2`:
//!
//! * `trim_blocks` and `lstrip_blocks` on, and `{% break %}` and `{% continue %}`.
//! * `raise_exception(message)`, which surfaces as [`ChatError::Raised`].
//! * `strftime_now(format)`, in UTC rather than local time. See [`Chat::with_time`].
//! * `tojson`, printing what Python's `json.dumps` prints and taking its `indent`,
//!   `separators`, `sort_keys` and `ensure_ascii` keyword arguments. Jinja's own
//!   `tojson` escapes HTML characters and prints compact JSON, which is not what the
//!   model saw in training.
//! * Python's string and dict methods, such as `strip`, `split`, `startswith` and
//!   `items`, from minijinja-contrib's `pycompat`.
//! * `tools` and `documents` always defined, as none when the caller passes none, and
//!   `bos_token` and `eos_token` from the GGUF.
//!
//! The one construct not supported is `{% generation %}`, which marks assistant tokens
//! for training. No inference template needs it.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//!
//! use dovata_chat::{Chat, Message};
//! use serde_json::json;
//!
//! # fn main() -> Result<(), dovata_chat::ChatError> {
//! let chat = Chat::from_files(Path::new("model.gguf"), Path::new("tokenizer.json"))?;
//!
//! let messages = [Message::system("Be brief."), Message::user("What is a KV cache?")];
//! let prompt = chat.render(&messages, true)?;
//! // `false`: the template already wrote any BOS token into the prompt.
//! let ids = chat.encode(&prompt, false)?;
//!
//! // Values a template reads go in `extra`, such as Qwen3's thinking switch.
//! let prompt = chat.render_with(&messages, true, &json!({ "enable_thinking": false }))?;
//!
//! // The program runs the model on `ids` until it produces `chat.eos()`, then:
//! # let generated: Vec<u32> = Vec::new();
//! let reply = chat.decode(&generated, true)?;
//! # let _ = (ids, prompt, reply);
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]

use std::path::Path;
use std::time::SystemTime;

use minijinja::value::{merge_maps, Serde, ValueKind};
use minijinja::{context, Environment, Value};
use serde::Serialize;

mod error;
mod header;
#[cfg(feature = "hub")]
mod hub;
mod message;
mod template;
pub mod vocab;

pub use error::{ChatError, Mismatch};
pub use message::Message;

/// The GGUF header type [`Chat::from_gguf`] takes, re-exported so a caller does not need
/// to depend on dovata just to name it.
pub use dovata::gguf::Gguf;

/// The `tokenizers` crate, re-exported so a caller building a [`Chat`] with [`Chat::new`]
/// uses the same version this package was built against.
pub use tokenizers;
/// The tokenizer [`Chat::new`] takes. See [`tokenizers`].
pub use tokenizers::Tokenizer;

/// Shorthand for this package's results.
pub type Result<T> = std::result::Result<T, ChatError>;

/// The name the chat template is registered under. It has no extension on purpose:
/// minijinja turns on HTML escaping for names ending in `.html`, and a prompt must
/// never be escaped.
const TEMPLATE_NAME: &str = "chat";

/// A model's chat template and tokenizer, ready to turn messages into a prompt and a
/// prompt into token ids.
///
/// Build one with [`Chat::from_files`] when both files are on disk, `Chat::from_hub` to
/// fetch them, which needs the `hub` feature, [`Chat::from_gguf`] when the GGUF header
/// is already parsed, or [`Chat::new`] when the caller has the pieces from somewhere
/// else.
#[derive(Clone)]
pub struct Chat {
    template: String,
    env: Environment<'static>,
    tokenizer: Tokenizer,
    bos: Option<String>,
    eos: Option<String>,
}

impl std::fmt::Debug for Chat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The template is kilobytes of Jinja and the tokenizer is the whole vocabulary,
        // so print their sizes rather than their contents.
        f.debug_struct("Chat")
            .field("template_bytes", &self.template.len())
            .field("vocab_size", &self.tokenizer.get_vocab_size(true))
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish()
    }
}

impl Chat {
    /// Build from a GGUF file and the model's `tokenizer.json`.
    ///
    /// Only the GGUF's header is read, never the tensor data, and only as much of it as
    /// the metadata needs, up to dovata's
    /// [`HEADER_BUDGET`](dovata::load::preflight::HEADER_BUDGET). A header fetched with an
    /// HTTP range request works as well as the whole file.
    ///
    /// Refuses a `tokenizer.json` whose vocabulary disagrees with the GGUF's. See
    /// [`vocab::check`].
    pub fn from_files(gguf: &Path, tokenizer_json: &Path) -> Result<Chat> {
        let header = header::read(gguf)?;
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| ChatError::Tokenizer(format!("{}: {e}", tokenizer_json.display())))?;
        Chat::from_gguf(&header, tokenizer)
    }

    /// Build from a GGUF repo on the Hugging Face Hub, the same repo id dovata fetches
    /// the model from.
    ///
    /// The template and the special tokens come from a GGUF header in the repo: one
    /// already in the local hf-hub cache if there is one, and otherwise the first GGUF by
    /// name, read with HTTP range requests so only its header is downloaded.
    /// `tokenizer.json` comes from the same repo, or, since GGUF repos rarely have one,
    /// from the base model, at its `main` revision. The base model is the one the
    /// header names in `general.base_model.0.repo_url`, or else the `base_model` of the
    /// repo's model card. The two files are then checked against each other as
    /// [`Chat::from_files`] checks them, which is what makes the base-model fallback
    /// safe.
    ///
    /// `revision` is a branch, tag or commit of `repo`, and `None` means `main`. A Hub
    /// token comes from `HF_TOKEN` or from `huggingface-cli login`; a gated base model
    /// needs one.
    #[cfg(feature = "hub")]
    pub fn from_hub(repo: &str, revision: Option<&str>) -> Result<Chat> {
        hub::from_hub(repo, revision)
    }

    /// Build from a GGUF header that is already parsed, and a tokenizer.
    ///
    /// For a caller that read the header with dovata already, such as the Jade compiler
    /// after preflight, so the file is not read twice. Checks the vocabularies agree the
    /// same way [`Chat::from_files`] does.
    pub fn from_gguf(header: &Gguf, tokenizer: Tokenizer) -> Result<Chat> {
        let pieces = header::Pieces::of(header)?;
        vocab::check(&pieces.tokens, pieces.token_types.as_deref(), &tokenizer)?;
        Chat::new(pieces.template, tokenizer, pieces.bos, pieces.eos)
    }

    /// Build from pieces the caller already has: a chat template, a tokenizer, and the
    /// BOS and EOS strings the template may print.
    ///
    /// The escape hatch for a model whose GGUF has no template, or for a caller who wants
    /// a different template than the one the model shipped. It checks nothing against a
    /// vocabulary, because there is none to check against. Fails only when the template
    /// does not compile.
    pub fn new(template: String, tokenizer: Tokenizer, bos: Option<String>, eos: Option<String>) -> Result<Chat> {
        let env = template::environment(template.clone(), None).map_err(ChatError::Template)?;
        Ok(Chat { template, env, tokenizer, bos, eos })
    }

    /// Fix the time `strftime_now` reports, for a build that must be reproducible.
    ///
    /// Some templates print today's date into the system prompt. Llama 3.1 and later do,
    /// through `strftime_now("%d %b %Y")`. Rendered at build time, that bakes the build
    /// date into the program, so two builds of the same source differ. Pinning the time
    /// makes them match. Without it, `strftime_now` reads the system clock.
    ///
    /// Either way the time is formatted in UTC. `transformers` uses local time, but a
    /// build machine's time zone says nothing about where the program runs, and UTC
    /// needs no time zone database.
    #[must_use]
    pub fn with_time(mut self, now: SystemTime) -> Chat {
        template::set_clock(&mut self.env, Some(now));
        self
    }

    /// Turn text into token ids.
    ///
    /// `add_special` runs the tokenizer's post-processor, which for many models adds a
    /// BOS token. Pass `false` for a prompt that came from [`Chat::render`]: a template
    /// that prints `bos_token` already put it in the text, and adding it again gives the
    /// model two. Special tokens written in the text, such as `<|im_start|>`, become
    /// their single ids either way.
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode_fast(text, add_special)
            .map_err(|e| ChatError::Tokenizer(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Turn token ids back into text. `skip_special` drops special tokens such as turn
    /// markers and EOS, which is what a caller showing a reply to a person wants.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.tokenizer.decode(ids, skip_special).map_err(|e| ChatError::Tokenizer(format!("decode: {e}")))
    }

    /// Render a conversation into the model's prompt string.
    ///
    /// `add_generation_prompt` appends the opening of an assistant turn, so the model's
    /// next tokens are the reply. Leave it off to render a finished conversation.
    pub fn render(&self, messages: &[Message], add_generation_prompt: bool) -> Result<String> {
        self.render_with(messages, add_generation_prompt, &())
    }

    /// Render a conversation, with extra values the template reads.
    ///
    /// `extra` must serialize to a map, or to nothing. Its keys become template
    /// variables: `tools` for function definitions, `enable_thinking` for Qwen3,
    /// `date_string` for Llama 3, and whatever else a model card names. It takes any
    /// `Serialize` value rather than only a `serde_json::Value` because key order
    /// reaches the prompt: `tojson` prints a tool's keys in the order they arrive, the
    /// way Python does, and a struct keeps its field order where a default
    /// `serde_json::Value` sorts its keys.
    ///
    /// `messages` and `add_generation_prompt` cannot be overridden through `extra`; a
    /// map that tries is refused, since one of the two values would be silently lost.
    pub fn render_with<E: Serialize + ?Sized>(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        extra: &E,
    ) -> Result<String> {
        for (i, m) in messages.iter().enumerate() {
            m.check().map_err(|e| ChatError::BadInput(format!("message {i}: {e}")))?;
        }

        let extra = Value::from(Serde(extra));
        match extra.kind() {
            ValueKind::Map => {
                for reserved in ["messages", "add_generation_prompt"] {
                    if extra.get_attr(reserved).is_ok_and(|v| !v.is_undefined()) {
                        return Err(ChatError::BadInput(format!(
                            "extra sets `{reserved}`, which render_with takes as its own argument"
                        )));
                    }
                }
            }
            ValueKind::None | ValueKind::Undefined => {}
            other => {
                return Err(ChatError::BadInput(format!("extra must serialize to a map, not a {other}")));
            }
        }

        let given = context! {
            messages => Value::from(Serde(messages)),
            add_generation_prompt => add_generation_prompt,
        };
        // What transformers passes when the caller does not. It always defines `tools`
        // and `documents`, as None, and templates test `tools is defined` and
        // `tools is not none` as different things. A missing BOS or EOS is left
        // undefined rather than none, which is also what transformers does: it passes
        // only the special tokens the tokenizer has.
        let mut defaults = vec![("tools", Value::from(())), ("documents", Value::from(()))];
        if let Some(bos) = &self.bos {
            defaults.push(("bos_token", Value::from(bos.as_str())));
        }
        if let Some(eos) = &self.eos {
            defaults.push(("eos_token", Value::from(eos.as_str())));
        }
        // merge_maps takes a key from the last map that has it, whatever its doc example
        // says; the tests pin that. So `extra` overrides the defaults, and nothing
        // overrides the arguments.
        let ctx = merge_maps([Value::from_pairs(defaults), extra, given]);

        let template = self.env.get_template(TEMPLATE_NAME).map_err(ChatError::Template)?;
        template.render(ctx).map_err(ChatError::from_render)
    }

    /// The BOS string, when the GGUF names a BOS token. Qwen models do not.
    pub fn bos(&self) -> Option<&str> {
        self.bos.as_deref()
    }

    /// The EOS string: the token that ends the model's turn, which a caller stops
    /// generating at.
    pub fn eos(&self) -> Option<&str> {
        self.eos.as_deref()
    }

    /// The chat template source, as the GGUF carries it.
    pub fn template(&self) -> &str {
        &self.template
    }

    /// The tokenizer, for what [`Chat::encode`] and [`Chat::decode`] do not cover, such
    /// as offsets or the vocabulary itself.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests;
