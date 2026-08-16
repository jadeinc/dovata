//! Text to tokens and back, over llama's tokenizer.
//!
//! Two things live here, and the second is the reason the module exists.
//!
//! The wrappers are thin — `str_to_token` and `token_to_piece` with this engine's error
//! strings instead of llama's debug formatting.
//!
//! The [`PieceDecoder`] is not thin, conceptually. A token is a span of *bytes*, not of
//! characters, so a three-byte character can arrive as the tail of one token and the
//! head of the next. Decode each token's bytes on their own and the user sees two
//! replacement characters where one character belongs — most visibly in non-Latin text
//! and emoji, which is to say the cases nobody exercises locally. The decoder carries
//! the incomplete tail across calls.
//!
//! Every request path did this, and each set up its own decoder inline: one in the
//! standard path, one in compose, and a `Vec` of them in batch, one per sequence.

use llama_cpp_2::{model::{AddBos, LlamaModel}, token::LlamaToken};

/// Tokenize `text`, prepending BOS.
///
/// # Errors
/// If llama's tokenizer rejects the input.
pub fn encode(model: &LlamaModel, text: &str) -> Result<Vec<LlamaToken>, String> {
    model
        .str_to_token(text, AddBos::Always)
        .map_err(|e| format!("tokenize failed: {e}"))
}

/// A streaming token-to-text decoder.
///
/// One per sequence. Holding a shared decoder across interleaved sequences would splice
/// one sequence's incomplete byte tail onto another's next token — which is exactly the
/// bug the decoder exists to prevent, reintroduced by sharing it.
pub struct PieceDecoder {
    inner: encoding_rs::Decoder,
}

impl std::fmt::Debug for PieceDecoder {
    // encoding_rs::Decoder is not Debug, and the workspace lints want one.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PieceDecoder").finish_non_exhaustive()
    }
}

impl Default for PieceDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PieceDecoder {
    /// A decoder with no carried state.
    ///
    /// BOM handling is off: a model's token stream is not a document, and a leading
    /// U+FEFF would be part of the reply rather than an encoding marker.
    #[must_use]
    pub fn new() -> Self {
        Self { inner: encoding_rs::UTF_8.new_decoder_without_bom_handling() }
    }

    /// The text for one generated token, carrying any incomplete UTF-8 tail forward.
    ///
    /// # Errors
    /// If llama cannot render the token.
    pub fn piece(&mut self, model: &LlamaModel, token: LlamaToken) -> Result<String, String> {
        model
            .token_to_piece(token, &mut self.inner, true, None)
            .map_err(|e| format!("token to str failed: {e}"))
    }
}
