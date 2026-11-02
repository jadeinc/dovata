//! Author: Joe Ricks
//! File: src/chat/src/lib.rs
//! Description: Turns a conversation into a model's prompt and token ids, and back.

#![warn(missing_docs)]

use std::path::Path;
use std::time::SystemTime;

use minijinja::value::{merge_maps, Serde, ValueKind};
use minijinja::{context, Environment, Value};
use serde::Serialize;

mod error;
mod header;
#[cfg(feature = "hf-hub")]
mod hf_hub;
mod message;
mod template;
pub mod vocab;

pub use error::{ChatError, Mismatch};
pub use message::Message;

/// The GGUF header type [`Chat::from_gguf`] takes, re-exported so a program using
/// dovata-chat does not need to depend on dovata just to name it.
pub use dovata::gguf::Gguf;

/// The `tokenizers` crate, re-exported so a program building a [`Chat`] with [`Chat::new`]
/// uses the same version this package was built against.
pub use tokenizers;
/// The tokenizer [`Chat::new`] takes. See [`tokenizers`].
pub use tokenizers::Tokenizer;

/// Purpose: This type is the result type dovata-chat's functions return.
pub type Result<T> = std::result::Result<T, ChatError>;

/// The name the chat template is registered under. It has no extension on purpose:
/// minijinja turns on HTML escaping for names ending in `.html`, and a prompt must
/// never be escaped.
const TEMPLATE_NAME: &str = "chat";

/// Purpose: This struct holds a model's chat template and tokenizer.
#[derive(Clone)]
pub struct Chat {
    template: String,
    env: Environment<'static>,
    tokenizer: Tokenizer,
    bos: Option<String>,
    eos: Option<String>,
}

impl std::fmt::Debug for Chat {
    /// Purpose: This function prints a short debug summary of a `Chat`.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result
    {
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
    /// Purpose: This function builds a `Chat` from a GGUF file and a `tokenizer.json` on
    /// disk.
    ///
    /// Inputs:
    /// - `gguf`: the GGUF file, whole or just its header.
    /// - `tokenizer_json`: the model's `tokenizer.json`.
    ///
    /// Outputs: the `Chat`, or an error if a file cannot be read or the two vocabularies
    /// differ, as [`vocab::check`] decides.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    ///
    /// use dovata_chat::{Chat, Message};
    /// use serde_json::json;
    ///
    /// # fn main() -> Result<(), dovata_chat::ChatError> {
    /// let chat = Chat::from_files(Path::new("model.gguf"), Path::new("tokenizer.json"))?;
    ///
    /// let messages = [Message::system("Be brief."), Message::user("What is a KV cache?")];
    /// let prompt = chat.render(&messages, true)?;
    /// // `false`: the template already wrote any BOS token into the prompt.
    /// let ids = chat.encode(&prompt, false)?;
    ///
    /// // Values a template reads go in `extra`, such as Qwen3's thinking switch.
    /// let prompt = chat.render_with(&messages, true, &json!({ "enable_thinking": false }))?;
    ///
    /// // The program runs the model on `ids` until it produces `chat.eos()`, then:
    /// # let generated: Vec<u32> = Vec::new();
    /// let reply = chat.decode(&generated, true)?;
    /// # let _ = (ids, prompt, reply);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_files(
        gguf: &Path,
        tokenizer_json: &Path,
    ) -> Result<Chat>
    {
        let header = header::read(gguf)?;
        let tokenizer = Tokenizer::from_file(tokenizer_json)
            .map_err(|e| ChatError::Tokenizer(format!("{}: {e}", tokenizer_json.display())))?;
        Chat::from_gguf(&header, tokenizer)
    }

    /// Purpose: This function builds a `Chat` from a GGUF repo on the Hugging Face Hub.
    ///
    /// Inputs:
    /// - `repo`: the GGUF repo id, such as `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`.
    /// - `revision`: a branch, tag or commit of `repo`, or `None` for `main`.
    ///
    /// Outputs: the `Chat`, or an error if a download fails or the files do not match.
    #[cfg(feature = "hf-hub")]
    pub fn from_hub(
        repo: &str,
        revision: Option<&str>,
    ) -> Result<Chat>
    {
        hf_hub::from_hub(repo, revision)
    }

    /// Purpose: This function builds a `Chat` from a GGUF header that is already parsed,
    /// and a tokenizer.
    ///
    /// Inputs:
    /// - `header`: the parsed GGUF header.
    /// - `tokenizer`: the model's tokenizer.
    ///
    /// Outputs: the `Chat`, or an error if the header lacks a piece or the vocabularies
    /// differ.
    pub fn from_gguf(
        header: &Gguf,
        tokenizer: Tokenizer,
    ) -> Result<Chat>
    {
        let pieces = header::Pieces::of(header)?;
        vocab::check(&pieces.tokens, pieces.token_types.as_deref(), &tokenizer)?;
        Chat::new(pieces.template, tokenizer, pieces.bos, pieces.eos)
    }

    /// Purpose: This function builds a `Chat` from a chat template, a tokenizer and the
    /// BOS and EOS strings.
    ///
    /// Inputs:
    /// - `template`: the chat template, in Jinja.
    /// - `tokenizer`: the model's tokenizer.
    /// - `bos`: the BOS string, if any.
    /// - `eos`: the EOS string, if any.
    ///
    /// Outputs: the `Chat`, or an error if the template does not compile.
    pub fn new(
        template: String,
        tokenizer: Tokenizer,
        bos: Option<String>,
        eos: Option<String>,
    ) -> Result<Chat>
    {
        let env = template::environment(template.clone(), None).map_err(ChatError::Template)?;
        Ok(Chat { template, env, tokenizer, bos, eos })
    }

    /// Purpose: This function fixes the time that `strftime_now` reports to the
    /// template.
    ///
    /// Inputs:
    /// - `now`: the time to report.
    ///
    /// Outputs: the same `Chat`, with its clock fixed.
    #[must_use]
    pub fn with_time(
        mut self,
        now: SystemTime,
    ) -> Chat
    {
        template::set_clock(&mut self.env, Some(now));
        self
    }

    /// Purpose: This function turns text into token ids.
    ///
    /// Inputs:
    /// - `text`: the text, usually a rendered prompt.
    /// - `add_special`: whether the tokenizer adds its own special tokens, such as BOS.
    ///
    /// Outputs: the token ids, or an error if encoding fails.
    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<u32>>
    {
        let encoding = self
            .tokenizer
            .encode_fast(text, add_special)
            .map_err(|e| ChatError::Tokenizer(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Purpose: This function turns token ids back into text.
    ///
    /// Inputs:
    /// - `ids`: the token ids.
    /// - `skip_special`: whether to drop special tokens such as turn markers.
    ///
    /// Outputs: the text, or an error if decoding fails.
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String>
    {
        self.tokenizer.decode(ids, skip_special).map_err(|e| ChatError::Tokenizer(format!("decode: {e}")))
    }

    /// Purpose: This function renders a conversation into the model's prompt string.
    ///
    /// Inputs:
    /// - `messages`: the conversation, in order.
    /// - `add_generation_prompt`: whether to end with the start of an assistant turn.
    ///
    /// Outputs: the prompt, or an error if rendering fails.
    pub fn render(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<String>
    {
        self.render_with(messages, add_generation_prompt, &())
    }

    /// Purpose: This function renders a conversation into the model's prompt string,
    /// with extra template values.
    ///
    /// Inputs:
    /// - `messages`: the conversation, in order.
    /// - `add_generation_prompt`: whether to end with the start of an assistant turn.
    /// - `extra`: a map of extra template values, such as `tools` or `enable_thinking`.
    ///
    /// Outputs: the prompt, or an error if the input is bad or rendering fails.
    pub fn render_with<E: Serialize + ?Sized>(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        extra: &E,
    ) -> Result<String>
    {
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

    /// Purpose: This function returns the BOS token's text.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the BOS text, or `None` if the model has no BOS token.
    pub fn bos(&self) -> Option<&str>
    {
        self.bos.as_deref()
    }

    /// Purpose: This function returns the EOS token's text.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the EOS text, or `None` if the model names no EOS token.
    pub fn eos(&self) -> Option<&str>
    {
        self.eos.as_deref()
    }

    /// Purpose: This function returns the chat template source.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the template text.
    pub fn template(&self) -> &str
    {
        &self.template
    }

    /// Purpose: This function returns the tokenizer.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer
    {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests;
