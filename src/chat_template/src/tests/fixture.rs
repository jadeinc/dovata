//! Author: Joe Ricks
//! File: src/chat_template/src/tests/fixture.rs
//! Description: Builds the test fixtures: a small GGUF header, a matching
//! `tokenizer.json`, and a scratch folder.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::json;

use crate::{Chat, Gguf, Tokenizer};

/// The vocabulary, in id order. Ids 0 to 3 are special.
pub(crate) const WORDS: &[&str] = &[
    "<unk>",
    "<|im_start|>",
    "<|im_end|>",
    "<s>",
    "hello",
    "world",
    "system",
    "user",
    "assistant",
    "Be",
    "brief.",
    "Hi",
];
pub(crate) const BOS_ID: u32 = 3;
pub(crate) const EOS_ID: u32 = 2;
const SPECIAL: usize = 4;

/// ChatML, as Qwen2 and many fine-tunes ship it, in its usual one-line form.
pub(crate) const CHATML: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// Purpose: This function builds the test `tokenizer.json`.
///
/// Inputs: none.
///
/// Outputs: the JSON text.
pub(crate) fn tokenizer_json() -> String
{
    let added: Vec<_> = WORDS[..SPECIAL]
        .iter()
        .enumerate()
        .map(|(id, w)| {
            json!({
                "id": id, "content": w, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    let vocab: serde_json::Map<String, serde_json::Value> =
        WORDS.iter().enumerate().map(|(id, w)| ((*w).to_owned(), json!(id))).collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                { "SpecialToken": { "id": "<s>", "type_id": 0 } },
                { "Sequence": { "id": "A", "type_id": 0 } }
            ],
            "pair": [
                { "SpecialToken": { "id": "<s>", "type_id": 0 } },
                { "Sequence": { "id": "A", "type_id": 0 } },
                { "Sequence": { "id": "B", "type_id": 1 } }
            ],
            "special_tokens": { "<s>": { "id": "<s>", "ids": [BOS_ID], "tokens": ["<s>"] } }
        },
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
    })
    .to_string()
}

/// Purpose: This function loads the test tokenizer.
///
/// Inputs: none.
///
/// Outputs: the tokenizer.
pub(crate) fn tokenizer() -> Tokenizer
{
    tokenizer_json().parse().expect("fixture tokenizer.json parses")
}

/// Purpose: This enum holds one metadata value of the test GGUF header.
enum Kv {
    Str(String),
    U32(u32),
    Strs(Vec<String>),
    I32s(Vec<i32>),
}

/// Purpose: This struct holds a test GGUF header, built key by key.
pub(crate) struct Header(Vec<(String, Kv)>);

impl Header {
    /// Purpose: This function starts a test GGUF header with only its architecture set.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the header.
    pub(crate) fn empty() -> Header
    {
        Header(vec![("general.architecture".into(), Kv::Str("llama".into()))])
    }

    /// Purpose: This function builds the test GGUF header with its tokens, BOS, EOS and
    /// ChatML template.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the header.
    pub(crate) fn chatml() -> Header
    {
        Header::empty()
            .tokens(WORDS)
            .u32("tokenizer.ggml.bos_token_id", BOS_ID)
            .u32("tokenizer.ggml.eos_token_id", EOS_ID)
            .str("tokenizer.chat_template", CHATML)
    }

    /// Purpose: This function sets a string key in the test header.
    ///
    /// Inputs:
    /// - `key`: the key.
    /// - `value`: the string.
    ///
    /// Outputs: the header.
    pub(crate) fn str(
        self,
        key: &str,
        value: &str,
    ) -> Header
    {
        self.set(key, Kv::Str(value.into()))
    }

    /// Purpose: This function sets a number key in the test header.
    ///
    /// Inputs:
    /// - `key`: the key.
    /// - `value`: the number.
    ///
    /// Outputs: the header.
    pub(crate) fn u32(
        self,
        key: &str,
        value: u32,
    ) -> Header
    {
        self.set(key, Kv::U32(value))
    }

    /// Purpose: This function sets the token list in the test header.
    ///
    /// Inputs:
    /// - `tokens`: the tokens, in id order.
    ///
    /// Outputs: the header.
    pub(crate) fn tokens(
        self,
        tokens: &[&str],
    ) -> Header
    {
        self.set("tokenizer.ggml.tokens", Kv::Strs(tokens.iter().map(|s| (*s).to_owned()).collect()))
    }

    /// Purpose: This function sets the token types in the test header.
    ///
    /// Inputs:
    /// - `types`: one type per token, in id order.
    ///
    /// Outputs: the header.
    pub(crate) fn token_types(
        self,
        types: &[i32],
    ) -> Header
    {
        self.set("tokenizer.ggml.token_type", Kv::I32s(types.to_vec()))
    }

    /// Purpose: This function removes a key from the test header.
    ///
    /// Inputs:
    /// - `key`: the key.
    ///
    /// Outputs: the header.
    pub(crate) fn without(
        mut self,
        key: &str,
    ) -> Header
    {
        self.0.retain(|(k, _)| k != key);
        self
    }

    /// Purpose: This function sets a key in the test header, replacing any old value.
    ///
    /// Inputs:
    /// - `key`: the key.
    /// - `value`: the value.
    ///
    /// Outputs: the header.
    fn set(
        mut self,
        key: &str,
        value: Kv,
    ) -> Header
    {
        self.0.retain(|(k, _)| k != key);
        self.0.push((key.into(), value));
        self
    }

    /// Purpose: This function writes the test header as GGUF bytes.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the bytes.
    pub(crate) fn bytes(&self) -> Vec<u8>
    {
        /// Purpose: This function writes one string in GGUF form.
        ///
        /// Inputs:
        /// - `out`: the bytes to write to.
        /// - `s`: the string.
        ///
        /// Outputs: none.
        fn string(
            out: &mut Vec<u8>,
            s: &str,
        )
        {
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes()); // tensors
        out.extend_from_slice(&(self.0.len() as u64).to_le_bytes());
        for (key, value) in &self.0 {
            string(&mut out, key);
            match value {
                Kv::Str(s) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut out, s);
                }
                Kv::U32(v) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Strs(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items {
                        string(&mut out, s);
                    }
                }
                Kv::I32s(items) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&5u32.to_le_bytes());
                    out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for v in items {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
        }
        out
    }

    /// Purpose: This function parses the test header with dovata's GGUF reader.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the parsed header.
    pub(crate) fn parse(&self) -> Gguf
    {
        Gguf::parse(&self.bytes()).expect("fixture header parses")
    }
}

/// Purpose: This function builds the test `Chat`.
///
/// Inputs: none.
///
/// Outputs: the `Chat`.
pub(crate) fn chat() -> Chat
{
    Chat::from_gguf(&Header::chatml().parse(), tokenizer()).expect("fixture chat builds")
}

/// Purpose: This struct holds a scratch folder that is deleted when the test ends.
pub(crate) struct Scratch(PathBuf);

impl Scratch {
    /// Purpose: This function creates an empty scratch folder for one test.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the scratch folder.
    pub(crate) fn new() -> Scratch
    {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dovata-chat-template-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        Scratch(dir)
    }

    /// Purpose: This function writes a file into the scratch folder.
    ///
    /// Inputs:
    /// - `name`: the file name.
    /// - `bytes`: the file's contents.
    ///
    /// Outputs: the file's path.
    pub(crate) fn write(
        &self,
        name: &str,
        bytes: impl AsRef<[u8]>,
    ) -> PathBuf
    {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }

    /// Purpose: This function returns the scratch folder's path.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the path.
    pub(crate) fn path(&self) -> &Path
    {
        &self.0
    }
}

impl Drop for Scratch {
    /// Purpose: This function deletes the scratch folder when a test ends.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
