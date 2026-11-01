//! Author: Joe Ricks
//! File: src/lib.rs
//! Description: Declares the dovata library's capability modules.

/// Opens a GGUF model for planning and reads its shape.
pub mod load;

/// Reads GGUF model file headers without llama.cpp, and writes small GGUF test files.
pub mod gguf;

/// Gets a model's GGUF files from a local path or the Hugging Face Hub.
pub mod hub;

/// Quantizes a GGUF model by running llama.cpp's `llama-quantize` program.
pub mod quantize;

/// Describes each supported architecture's compute graph as data, in ggml's op names.
pub mod layout;

/// Places every tensor of a model in memory on one target machine.
pub mod plan;

/// Reads the current machine's memory capacity, its GPUs and their use.
pub mod hw;

/// Ranks each MoE layer's experts by how often they are used.
#[cfg(feature = "moe")]
pub mod moe;

#[cfg(test)]
mod capabilities {
    //! That each feature-gated capability is reachable under default features: the
    //! `moe` module, and the halves of `quantize`, `hub` and `hw` that reach outside dovata.
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

    #[test]
    #[cfg(feature = "quantize")]
    fn quantize_native_half_is_present() {
        // The pure half always builds; the feature adds running llama.cpp's
        // llama-quantize. With it, a missing input is a bad input, found before the
        // program is looked for. Without it, every call is NotBuilt.
        use crate::quantize::{quantize, Ftype, QuantError, QuantSpec};
        let path = std::path::Path::new("/nonexistent/model.gguf");
        let err = quantize(path, path, &QuantSpec::new(Ftype::Q8_0)).unwrap_err();
        assert!(matches!(err, QuantError::BadInput(_)), "{err:?}");
    }

    #[test]
    #[cfg(feature = "hub")]
    fn hub_network_half_is_present() {
        // With the feature, a Hub source reaches the download code, which refuses a
        // malformed repo id before touching the network. Without it, it is NotBuilt.
        use crate::hub::{fetch, FetchError, FetchOptions, ModelSource, Want};
        let src = ModelSource::Hub {
            endpoint: "https://huggingface.co".into(),
            repo: "no such/id here".into(),
            revision: None,
            file: None,
        };
        let err = fetch(&src, Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert!(matches!(err, FetchError::BadSource(_)), "{err:?}");
    }

    #[test]
    #[cfg(all(feature = "hw", any(target_os = "macos", target_os = "linux")))]
    fn hw_probe_half_is_present() {
        // With the feature, the machine reads; without it, it is NotBuilt.
        assert!(crate::hw::machine().is_ok_and(|m| m.memory_bytes > 0));
    }
}
