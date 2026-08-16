//! Choosing the next token, and controlling what it is allowed to be.
//!
//! `control.rs` is the pure half: a state machine over the *text* the tokens spell.
//! Anchor spans, stop anchors that arrive split across two tokens, JSON depth tracking
//! so the grammar can be retired once the structured part of a reply is closed. None of
//! it needs a model, which is why it is tested on a laptop.
//!
//! `chain.rs` builds the llama sampler chain. It exists because the chain
//! `temp -> min_p -> dist` had been copy-pasted, with hardcoded constants, into four
//! places across two request paths — so changing the temperature in one silently
//! disagreed with the others. It is the same class of drift the model profiles had
//! before `gen-profile-package.py`, and the same fix: one owner.
//!
//! A peer of [`crate::decode`] rather than part of it, because `compose` and `batch`
//! build samplers too.

pub mod control;

/// The llama sampler chain: one definition of what "sampling" means here.
#[cfg(target_os = "linux")]
pub mod chain;
