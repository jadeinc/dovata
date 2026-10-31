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

/// Refusing a model file that cannot be planned, and reading the model's shape and KV
/// cost from its header. Separate from [`gguf`] because it is policy: the reader reports
/// what any file says, and this decides whether that is enough to plan from.
pub mod load;

/// GGUF header reading: what a model file holds, answered without llama.cpp.
/// [`load::preflight`] is its caller, and `gguf-inspect`, a bin of this package, is its
/// CLI. It is dependency-free by design, since it has to run on a laptop against a
/// model no GPU has touched.
pub mod gguf;

/// Getting a model's files: a local path, or a GGUF downloaded from the Hugging Face
/// Hub. A peer of [`load`] rather than part of it because it is the one capability that
/// talks to the network; the pure half, which picks the file, always builds, and the
/// download is behind the `hub` feature.
pub mod hub;

/// Quantizing a GGUF with llama.cpp's own quantizer. Never called implicitly: fetching
/// and planning never quantize, so a caller who wants a smaller model calls this. The
/// quant-type table always builds; the call into llama.cpp is behind `quantize`.
pub mod quantize;

/// Layouts: an architecture's compute graph as data, written in ggml's op names. Each
/// built-in layout is one architecture dovata supports. Separate from the planner
/// because the layout is the contract with the Jade compiler's kernels, and the planner
/// only decides where its tensors live.
pub mod layout;

/// The planner: where every tensor of a bound layout lives on one target machine,
/// handed to the compiler as an address table of region and offset pairs. Separate from
/// [`layout`] because the same layout plans differently on every target.
pub mod plan;

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
