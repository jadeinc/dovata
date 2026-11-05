//! Author: Joe Ricks
//! File: src/lib.rs
//! Description: Declares the dovata library's capability modules.

/// Opens a GGUF model for planning and reads its shape.
pub mod load;

/// Reads GGUF model file headers without llama.cpp, and writes small GGUF test files.
pub mod gguf;

/// Gets a model's GGUF files from a local path or the Hugging Face Hub.
pub mod hf_hub;

/// Quantizes a GGUF model by running llama.cpp's `llama-quantize` program.
pub mod quantize;

/// Describes each supported architecture's compute graph as data, in ggml's op names.
pub mod layout;

/// Describes the machine a plan is for and its kinds of memory.
pub mod target;

/// Splits a mapped GGUF file into windows no larger than one GPU buffer.
pub mod window;

/// Packs one step's short-lived tensors into shared memory by when each is alive.
pub mod arena;

/// Places every tensor of a model in memory on one target machine.
pub mod memplan;

/// Describes a model's tokenizer for its plan, and turns text into its token ids and back.
pub mod tokenize;

/// Reads the current machine's memory capacity, its GPUs and their use.
pub mod machine;

/// Runs dovata as a process that holds one model in memory, and lets programs start it,
/// map its memory and stop it.
pub mod host;

/// Ranks each MoE layer's experts by how often they are used.
#[cfg(feature = "moe")]
pub mod moe;

#[cfg(test)]
mod capabilities {
    //! That each feature-gated capability is reachable under default features: the
    //! `moe` module, and the halves of `quantize`, `hf_hub`, `machine` and `host` that
    //! reach outside dovata.
    //!
    //! A `#[cfg(feature = "…")]` with a mis-typed feature name compiles fine and
    //! silently drops the module. These name each feature once more, so a typo in
    //! Cargo.toml or in a gate fails here in under a second.
    //!
    //! Each test is itself gated: with the feature off the module is *supposed* to be
    //! absent, so the assertion is only meaningful when the feature is on.

    #[test]
    #[cfg(feature = "moe")]
    fn moe_pure_half_is_present()
    {
        // A malformed asset must be rejected, which also proves the loader linked.
        assert!(crate::moe::ExpertOrder::from_json(b"{}").is_err());
    }

    #[test]
    #[cfg(feature = "quantize")]
    fn quantize_native_half_is_present()
    {
        // The pure half always builds; the feature adds running llama.cpp's
        // llama-quantize. With it, a missing input is a bad input, found before the
        // program is looked for. Without it, every call is NotBuilt.
        use crate::quantize::{quantize, Ftype, QuantError, QuantSpec};
        let path = std::path::Path::new("/nonexistent/model.gguf");
        let err = quantize(path, path, &QuantSpec::new(Ftype::Q8_0)).unwrap_err();
        assert!(matches!(err, QuantError::BadInput(_)), "{err:?}");
    }

    #[test]
    #[cfg(feature = "hf-hub")]
    fn hf_hub_network_half_is_present()
    {
        // With the feature, a Hub source reaches the download code, which refuses a
        // malformed repo id before touching the network. Without it, it is NotBuilt.
        use crate::hf_hub::{fetch, FetchError, FetchOptions, ModelSource, Want};
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
    #[cfg(all(feature = "machine", any(target_os = "macos", target_os = "linux")))]
    fn machine_probe_half_is_present()
    {
        // With the feature, the machine reads; without it, it is NotBuilt.
        assert!(crate::machine::current().is_ok_and(|m| m.memory_bytes > 0));
    }

    #[test]
    #[cfg(feature = "tokenize")]
    fn tokenize_runtime_half_is_present()
    {
        // With the feature, a tokenizer.json that does not parse is refused by the
        // tokenizer itself; without it, describing one is NotBuilt.
        use crate::tokenize::{Tokenizer, TokenizeError, TokenizerPlan};
        let t = TokenizerPlan {
            kind: "gpt2".into(),
            tokens: 0,
            embedding_rows: 0,
            bos: None,
            eos: None,
            stop: vec![],
            sha256: crate::tokenize::sha256_hex("{"),
            definition: "{".into(),
        };
        assert!(matches!(Tokenizer::from_plan(&t), Err(TokenizeError::Unreadable(_))));
    }

    #[test]
    #[cfg(all(feature = "host", unix))]
    fn host_process_half_is_present()
    {
        // With the feature, looking for the process reaches the socket, whether or not
        // one runs; without it, every call is NotBuilt.
        use crate::host::{connect, HostError};
        assert!(!matches!(connect(), Err(HostError::NotBuilt)));
    }
}
