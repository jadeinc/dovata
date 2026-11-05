//! Author: Joe Ricks
//! File: src/tokenize/engine.rs
//! Description: Builds a plan's tokenizer from a model and its `tokenizer.json`, and runs
//! it: text to token ids, token ids to text, and generated ids to text one at a time.

use tokenizers::decoders::DecoderWrapper;
use tokenizers::models::ModelWrapper;
use tokenizers::normalizers::NormalizerWrapper;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::processors::PostProcessorWrapper;
use tokenizers::DecodeStream;

use crate::load::Model;

use super::{gguf_token_types, gguf_tokens, sha256_hex, specials, vocab, Specials, TokenizeError, TokenizerPlan};

/// Purpose: This function builds the tokenizer section of a model's plan.
///
/// Inputs:
/// - `model`: the model, whose GGUF header gives the vocabulary, the tokenizer's kind
///   and the special tokens.
/// - `definition`: the model's `tokenizer.json`, as text.
///
/// Outputs: the [`TokenizerPlan`], or a [`TokenizeError`] when the `tokenizer.json` does
/// not parse or its vocabulary differs from the GGUF's, as [`vocab::check_vocab`]
/// decides.
pub fn describe(
    model: &Model,
    definition: String,
) -> Result<TokenizerPlan, TokenizeError>
{
    let meta = model.meta();
    let tokens = gguf_tokens(meta)?;
    let types = gguf_token_types(meta);
    let tokenizer = parse(&definition)?;
    vocab::check_vocab(&tokens, types.as_deref(), &tokenizer).map_err(TokenizeError::Vocab)?;
    let Specials { kind, bos, eos, stop } = specials(meta, tokens.len())?;
    let defined = tokenizer.get_vocab(true).values().max().map_or(0, |&m| m.saturating_add(1));
    // The embedding table's rows, from the tensor itself: `token_embd.weight` is
    // [n_embd, n_vocab], fastest-varying first. A model without one is sized by its
    // vocabulary, which the converter makes the same length.
    let rows = model
        .tensor("token_embd.weight")
        .and_then(|t| t.info.dims.get(1).copied())
        .and_then(|r| u32::try_from(r).ok())
        .unwrap_or(u32::try_from(tokens.len()).unwrap_or(u32::MAX));
    Ok(TokenizerPlan { kind, tokens: defined, embedding_rows: rows, bos, eos, stop, sha256: sha256_hex(&definition), definition })
}

/// Purpose: This function parses a `tokenizer.json`.
///
/// Inputs:
/// - `definition`: the file's text.
///
/// Outputs: the Hugging Face tokenizer, or [`TokenizeError::Unreadable`].
fn parse(
    definition: &str,
) -> Result<tokenizers::Tokenizer, TokenizeError>
{
    definition.parse().map_err(|e| TokenizeError::Unreadable(format!("tokenizer.json does not parse: {e}")))
}

/// Purpose: This struct holds a model's tokenizer, ready to run, built from its plan.
#[derive(Clone)]
pub struct Tokenizer {
    /// The Hugging Face tokenizer that runs `tokenizer.json`.
    inner: tokenizers::Tokenizer,
    /// The BOS token's id, if any.
    bos: Option<u32>,
    /// The EOS token's id, if any.
    eos: Option<u32>,
    /// The ids that end the model's turn.
    stop: Vec<u32>,
    /// The rows of the model's embedding table; every id must be below this.
    rows: u32,
}

impl std::fmt::Debug for Tokenizer {
    /// Purpose: This function prints a short summary of a tokenizer.
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
        f.debug_struct("Tokenizer").field("bos", &self.bos).field("eos", &self.eos).field("stop", &self.stop).field("rows", &self.rows).finish()
    }
}

/// Purpose: This type names the streaming decoder of the Hugging Face tokenizer that
/// [`Tokenizer`] wraps.
type HfStream<'a> = DecodeStream<'a, ModelWrapper, NormalizerWrapper, PreTokenizerWrapper, PostProcessorWrapper, DecoderWrapper>;

/// Purpose: This struct holds a decoder that turns generated ids into text one at a
/// time, holding back a token that ends partway through a character.
pub struct Stream<'a> {
    /// The Hugging Face streaming decoder.
    inner: HfStream<'a>,
}

impl Tokenizer {
    /// Purpose: This function builds a tokenizer from a plan's tokenizer section.
    ///
    /// Inputs:
    /// - `plan`: the section, from [`crate::memplan::Plan::tokenizer`].
    ///
    /// Outputs: the [`Tokenizer`], or a [`TokenizeError`] when the definition does not
    /// match its hash or does not parse.
    pub fn from_plan(
        plan: &TokenizerPlan,
    ) -> Result<Tokenizer, TokenizeError>
    {
        if sha256_hex(&plan.definition) != plan.sha256 {
            return Err(TokenizeError::Unreadable("the plan's tokenizer.json does not match its sha256; the plan was changed after it was made".into()));
        }
        Ok(Tokenizer { inner: parse(&plan.definition)?, bos: plan.bos, eos: plan.eos, stop: plan.stop.clone(), rows: plan.embedding_rows })
    }

    /// Purpose: This function turns text into token ids.
    ///
    /// Inputs:
    /// - `text`: the text, such as a prompt a chat template rendered.
    /// - `add_special`: add the special tokens `tokenizer.json` puts around a sequence,
    ///   such as a BOS. Pass `false` for a prompt whose template already wrote them, so
    ///   the model does not get two.
    ///
    /// Outputs: the ids, or a [`TokenizeError`]. Special tokens written in the text,
    /// such as `<|im_start|>`, become their single ids either way.
    pub fn encode(
        &self,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<u32>, TokenizeError>
    {
        let ids = self.inner.encode(text, add_special).map_err(|e| TokenizeError::Run(format!("encoding failed: {e}")))?.get_ids().to_vec();
        if let Some(bad) = ids.iter().find(|&&id| id >= self.rows) {
            return Err(TokenizeError::Run(format!("the tokenizer gave id {bad}, and the model has rows for ids below {}", self.rows)));
        }
        Ok(ids)
    }

    /// Purpose: This function turns token ids into text.
    ///
    /// Inputs:
    /// - `ids`: the ids.
    /// - `skip_special`: leave out special tokens such as turn markers, which a person
    ///   reading the reply does not want.
    ///
    /// Outputs: the text, or a [`TokenizeError`].
    pub fn decode(
        &self,
        ids: &[u32],
        skip_special: bool,
    ) -> Result<String, TokenizeError>
    {
        self.inner.decode(ids, skip_special).map_err(|e| TokenizeError::Run(format!("decoding failed: {e}")))
    }

    /// Purpose: This function starts decoding generated ids one at a time.
    ///
    /// Inputs:
    /// - `skip_special`: leave out special tokens, as for [`Tokenizer::decode`].
    ///
    /// Outputs: the [`Stream`].
    pub fn stream(
        &self,
        skip_special: bool,
    ) -> Stream<'_>
    {
        Stream { inner: self.inner.decode_stream(skip_special) }
    }

    /// Purpose: This function says whether an id ends the model's turn.
    ///
    /// Inputs:
    /// - `id`: a generated id.
    ///
    /// Outputs: `true` for EOS and the other stop ids the plan names.
    pub fn is_stop(
        &self,
        id: u32,
    ) -> bool
    {
        self.stop.contains(&id)
    }

    /// Purpose: This function gives the BOS token's id.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the id, or `None` for a model with no BOS token.
    pub fn bos(&self) -> Option<u32>
    {
        self.bos
    }

    /// Purpose: This function gives the EOS token's id.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the id, or `None`.
    pub fn eos(&self) -> Option<u32>
    {
        self.eos
    }

    /// Purpose: This function gives every id that ends the model's turn.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the ids.
    pub fn stop(&self) -> &[u32]
    {
        &self.stop
    }
}

impl Stream<'_> {
    /// Purpose: This function adds one generated id and gives the text it completes.
    ///
    /// Inputs:
    /// - `id`: the id.
    ///
    /// Outputs: the new text, which is empty while a character is still incomplete, or a
    /// [`TokenizeError`].
    pub fn push(
        &mut self,
        id: u32,
    ) -> Result<String, TokenizeError>
    {
        self.inner.step(id).map(Option::unwrap_or_default).map_err(|e| TokenizeError::Run(format!("decoding failed: {e}")))
    }
}
