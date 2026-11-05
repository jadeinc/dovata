//! Author: Joe Ricks
//! File: src/tokenize/find.rs
//! Description: Finds a model's `tokenizer.json`: beside the model file, in the model's
//! Hub repo, or in the repo of the model it was converted from.

use std::path::{Path, PathBuf};

use crate::gguf::Gguf;
use crate::hf_hub::{self, FetchOptions, Origin};

use super::{TokenizeError, KEY_BASE_URL};

/// The file name a Hugging Face tokenizer is saved under.
pub const TOKENIZER_JSON: &str = "tokenizer.json";

/// The Hub a base model is fetched from when the model itself came from disk.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Purpose: This function finds a model's `tokenizer.json`.
///
/// Inputs:
/// - `first_part`: the model's first part file.
/// - `gguf`: its parsed GGUF header.
/// - `origin`: the Hub repo the model came from, or `None` for a model on disk.
/// - `opts`: the Hugging Face cache directory and access token.
///
/// Outputs: the file's path, or a [`TokenizeError`] naming every place looked. The
/// places, in order:
/// 1. `tokenizer.json` in the model file's own directory;
/// 2. the model's Hub repo;
/// 3. the base model's repo, which the GGUF header names in
///    `general.base_model.0.repo_url`, or else the model repo's card in `base_model`.
///    GGUF repos rarely carry a `tokenizer.json`, so this is the usual source.
///
/// What is found is not checked here; [`super::describe`] checks it against the GGUF's
/// vocabulary, which is what makes the base model's file safe to use.
pub fn find(
    first_part: &Path,
    gguf: &Gguf,
    origin: Option<&Origin>,
    opts: &FetchOptions,
) -> Result<PathBuf, TokenizeError>
{
    let mut tried = Vec::new();
    if let Some(beside) = first_part.parent().map(|d| d.join(TOKENIZER_JSON)) {
        if beside.is_file() {
            return Ok(beside);
        }
        tried.push(beside.display().to_string());
    }
    let fetch = |o: &Origin, file: &str| hf_hub::fetch_file(o, file, opts).map_err(|e| TokenizeError::Fetch(format!("cannot fetch {file} from {}: {e}", o.repo)));
    if let Some(o) = origin {
        if let Some(p) = fetch(o, TOKENIZER_JSON)? {
            return Ok(p);
        }
        tried.push(format!("the repo {}", o.repo));
    }
    let mut base = gguf.string(KEY_BASE_URL).and_then(hf_hub::repo_from_url);
    if base.is_none() {
        if let Some(o) = origin {
            if let Some(card) = fetch(o, "README.md")? {
                base = std::fs::read_to_string(card).ok().and_then(|t| hf_hub::card_base_model(&t));
            }
        }
    }
    if let Some(repo) = base {
        let endpoint = origin.map_or_else(|| DEFAULT_ENDPOINT.to_owned(), |o| o.endpoint.clone());
        let b = Origin { endpoint, repo, revision: None };
        if let Some(p) = fetch(&b, TOKENIZER_JSON)? {
            return Ok(p);
        }
        tried.push(format!("the base model's repo {}", b.repo));
    } else {
        tried.push("a base model, which neither the GGUF nor the repo's card names".into());
    }
    Err(TokenizeError::NotFound(format!("found no tokenizer.json for the model; looked in {}. Give one with --tokenizer FILE", tried.join(", "))))
}
