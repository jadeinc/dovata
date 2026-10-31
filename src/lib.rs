//! dovata — the JADE OS model memory and inference layout planner.
//!
//! As of 3.0, dovata is a Rust library the Jade compiler uses at build time. It reads a
//! model's GGUF file and a layout file describing the compute graph, decides where
//! every tensor lives on the target machine, and hands the compiler that decision as
//! data. The compiler generates the GPU code from it. dovata itself never runs a
//! model.
//!
//! Up to 2.x it was the opposite: a provider library a running Jade program loaded to
//! do inference through llama.cpp. That runtime is gone on this branch; `main` and the
//! `v2.*` tags keep it. The work is GPU-bound and the compiler should emit the machine
//! code, so a runtime engine in front of it was overhead.
//!
//! ## What is here today
//!
//! The 3.0 rewrite starts from the parts of 2.x that were already about memory: reading
//! the GGUF header, sizing the KV cache, and ranking MoE experts. The planner, the
//! layout format and the quantizer are still to come.
//!
//! ## How this directory is organized
//!
//! Every child of `src/` is one *capability*, named for what it does, behind a cargo
//! feature where it can be compiled out. They are flat rather than nested; the grouping
//! that helps a reader lives in the README's map, not in a directory tree. The doc
//! comments below say why each capability sits where it does, because "why here and
//! not inside that one" is the question a flat tree invites.

/// Reading a model file's header and refusing a model nothing can plan for, plus the
/// KV-cache sizing math. Separate from [`gguf`] because it is policy, and from
/// [`model`] because it decides once per model file, not per profile.
pub mod load;

/// Per-model vocabulary as data: tool-call delimiters, turn markers, chat templates.
/// A peer of `load` rather than part of it because the profile outlives admission, and
/// because swapping a `.gguf` must not mean editing the code.
pub mod model;

/// GGUF header reading: what a model file holds, answered without llama.cpp.
/// [`load::preflight`] is its caller, and `gguf-inspect`, a bin of this package, is its
/// CLI. It is dependency-free by design, since it has to run on a laptop against a
/// model no GPU has touched.
pub mod gguf;

/// MoE expert ranking: which experts each layer uses most, from an offline asset. The
/// planner will use it to decide which experts stay resident in VRAM.
#[cfg(feature = "moe")]
pub mod moe;

#[cfg(test)]
mod capabilities {
    //! That each capability's pure half is reachable under default features.
    //!
    //! A `#[cfg(feature = "…")]` with a mis-typed feature name compiles fine and
    //! silently drops the module. These name each feature once more, so a typo in
    //! Cargo.toml or in a gate fails here in under a second.
    //!
    //! Each test is itself gated: with the feature off the module is *supposed* to be
    //! absent, so the assertion is only meaningful when the feature is on.

    #[test]
    #[cfg(feature = "moe")]
    fn moe_pure_half_is_present() {
        // A malformed asset must be rejected, which also proves the loader linked.
        assert!(crate::moe::ExpertOrder::from_json(b"{}").is_err());
    }
}
