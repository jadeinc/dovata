//! Author: Joe Ricks
//! File: src/load/preflight.rs
//! Description: Refuses a GGUF model file whose header shows it cannot be planned.

use std::fmt;
use std::path::Path;

use crate::gguf::{Gguf, Split};

/// How much of the file to read. A GGUF's metadata and tensor index both sit before the
/// tensor data; 64 MB covers both for every model we have planned, and it is a cap on a
/// file that may be hundreds of gigabytes, not a target.
pub const HEADER_BUDGET: u64 = 64 * 1024 * 1024;

/// Purpose: This struct holds a model file that passed preflight, with its main facts.
#[derive(Debug, Clone)]
pub struct Preflight {
    /// The parsed header. Complete: the tensor index is whole and `data_offset` is set.
    pub gguf: Gguf,
    /// `general.architecture`, the string llama.cpp and the built-in layouts key on.
    pub architecture: String,
    /// `general.name`, free text.
    pub name: Option<String>,
    /// The model's native context window, from `<arch>.context_length`.
    pub context_length: Option<u64>,
    /// Whether the file declares a mixture-of-experts model.
    pub is_moe: bool,
    /// `general.file_type`, the file's `llama_ftype`.
    pub file_type: Option<u32>,
    /// Where this file sits in a split model, if it is one part of several.
    pub split: Option<Split>,
}

impl Preflight {
    /// Purpose: This function reads a model file's header from disk and runs the
    /// preflight checks on it.
    ///
    /// Inputs:
    /// - `path`: the model file, or the first part of a split model.
    ///
    /// Outputs: the [`Preflight`], or an [`Error`].
    pub fn read(
        path: &Path,
    ) -> Result<Preflight, Error>
    {
        use std::io::Read;

        let unreadable = |e: std::io::Error| Error::Unreadable { path: path.display().to_string(), detail: e.to_string() };
        let file = std::fs::File::open(path).map_err(unreadable)?;
        let mut bytes = Vec::new();
        file.take(HEADER_BUDGET).read_to_end(&mut bytes).map_err(unreadable)?;

        Self::evaluate(&bytes)
    }

    /// Purpose: This function runs the preflight checks on a GGUF header already in
    /// memory.
    ///
    /// Inputs:
    /// - `bytes`: the start of the model file.
    ///
    /// Outputs: the [`Preflight`], or an [`Error`] naming why the file cannot be planned.
    pub fn evaluate(
        bytes: &[u8],
    ) -> Result<Preflight, Error>
    {
        let gguf = Gguf::parse(bytes).map_err(|e| Error::NotLoadable(e.to_string()))?;

        // Split parts after the first carry no model metadata, only `split.*` keys and
        // their share of the tensors. They are read with part 0, never on their own.
        if let Some(split) = gguf.split() {
            if split.no != 0 {
                return Err(Error::NotFirstPart(split));
            }
        }

        // Required-key check: a file missing these cannot load at all.
        let missing: Vec<String> = gguf
            .report()
            .blocking()
            .iter()
            .map(|c| format!("{} — {}", c.key, c.note))
            .collect();
        if !missing.is_empty() {
            return Err(Error::MissingKeys(missing));
        }

        // The planner places every tensor, so it needs every tensor. A range-fetched
        // header that stopped short is fine for gguf-inspect and not enough here.
        if gguf.truncated {
            return Err(Error::TruncatedIndex { have: gguf.tensors.len() as u64, declared: gguf.tensor_count });
        }

        let architecture = gguf.architecture().unwrap_or_default().to_owned();
        Ok(Preflight {
            name: gguf.name().map(str::to_owned),
            context_length: gguf.u64(&format!("{architecture}.context_length")),
            is_moe: gguf.is_moe(),
            file_type: gguf.file_type(),
            split: gguf.split(),
            architecture,
            gguf,
        })
    }

    /// Purpose: This function checks whether a context length is longer than the model
    /// was trained for.
    ///
    /// Inputs:
    /// - `requested`: the context length, in tokens.
    ///
    /// Outputs: `true` when it is longer than `<arch>.context_length`.
    pub fn ctx_exceeds_native(
        &self,
        requested: u32,
    ) -> bool
    {
        self.context_length.is_some_and(|native| u64::from(requested) > native)
    }
}

/// Purpose: This enum lists the reasons preflight refuses a model file.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The file could not be opened or read.
    Unreadable { path: String, detail: String },
    /// The bytes are not a GGUF header this reader understands.
    NotLoadable(String),
    /// Required metadata is absent, so llama.cpp would refuse the file too.
    MissingKeys(Vec<String>),
    /// The tensor index stopped before the count the header declares.
    TruncatedIndex { have: u64, declared: u64 },
    /// The file is a later part of a split model. Open part 1 instead; it lists the
    /// model's metadata, and the other parts are read with it.
    NotFirstPart(Split),
}

impl fmt::Display for Error {
    /// Purpose: This function formats preflight errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        match self {
            Error::Unreadable { path, detail } => write!(f, "cannot read the model at {path}: {detail}"),
            Error::NotLoadable(detail) => write!(
                f,
                "the model file is not a usable GGUF: {detail}\n  \
                 Inspect it with: cargo run --bin gguf-inspect -- <model>"
            ),
            Error::MissingKeys(keys) => {
                writeln!(f, "the model file is missing metadata llama.cpp requires:")?;
                for k in keys {
                    writeln!(f, "  {k}")?;
                }
                write!(f, "  This file will not load. It may be truncated or mis-converted.")
            }
            Error::TruncatedIndex { have, declared } => write!(
                f,
                "the tensor index is cut off: {have} of {declared} tensors were read in the first \
                 {HEADER_BUDGET} bytes. The file is incomplete, or its index is larger than the budget."
            ),
            Error::NotFirstPart(s) => write!(
                f,
                "this is part {} of {} of a split model; open part 1, which carries the metadata",
                s.no + 1,
                s.count
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Purpose: This function builds a test GGUF header with only the keys preflight
    /// requires.
    ///
    /// Inputs:
    /// - `arch`: the architecture to declare.
    /// - `name`: the `general.name` to declare, if any.
    ///
    /// Outputs: the header bytes, with no tensors.
    pub(crate) fn header(
        arch: &str,
        name: Option<&str>,
    ) -> Vec<u8>
    {
        header_with(arch, name, &[])
    }

    /// Purpose: This function builds the same test header as [`header`], plus extra
    /// metadata.
    ///
    /// Inputs:
    /// - `arch`: the architecture to declare.
    /// - `name`: the `general.name` to declare, if any.
    /// - `extra`: more entries, as `(key, type tag, encoded value)`.
    ///
    /// Outputs: the header bytes, with no tensors.
    pub(crate) fn header_with(
        arch: &str,
        name: Option<&str>,
        extra: &[(&str, u32, Vec<u8>)],
    ) -> Vec<u8>
    {
        let mut kvs: Vec<(String, u32, Vec<u8>)> = Vec::new();

        let push_str = |kvs: &mut Vec<(String, u32, Vec<u8>)>, k: &str, v: &str| {
            let mut p = (v.len() as u64).to_le_bytes().to_vec();
            p.extend_from_slice(v.as_bytes());
            kvs.push((k.to_owned(), 8, p));
        };
        let push_u32 = |kvs: &mut Vec<(String, u32, Vec<u8>)>, k: &str, v: u32| {
            kvs.push((k.to_owned(), 4, v.to_le_bytes().to_vec()));
        };

        push_str(&mut kvs, "general.architecture", arch);
        if let Some(n) = name {
            push_str(&mut kvs, "general.name", n);
        }
        push_u32(&mut kvs, &format!("{arch}.block_count"), 48);
        push_u32(&mut kvs, &format!("{arch}.context_length"), 262_144);
        push_u32(&mut kvs, &format!("{arch}.embedding_length"), 2048);
        push_u32(&mut kvs, &format!("{arch}.attention.head_count"), 32);
        push_str(&mut kvs, "tokenizer.ggml.model", "gpt2");
        // tokenizer.ggml.tokens: a one-element string array is enough to be "present".
        let mut toks = 8u32.to_le_bytes().to_vec();
        toks.extend_from_slice(&1u64.to_le_bytes());
        toks.extend_from_slice(&1u64.to_le_bytes());
        toks.extend_from_slice(b"!");
        kvs.push(("tokenizer.ggml.tokens".to_owned(), 9, toks));
        for (k, tag, payload) in extra {
            kvs.push(((*k).to_owned(), *tag, payload.clone()));
        }

        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, tag, payload) in &kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(payload);
        }
        b
    }

    #[test]
    fn a_complete_header_preflights()
    {
        let bytes = header_with("qwen3moe", Some("Qwen3-Coder-30B-A3B-Instruct"), &[("general.file_type", 4, 15u32.to_le_bytes().to_vec())]);
        let pf = Preflight::evaluate(&bytes).expect("preflights");
        assert_eq!(pf.architecture, "qwen3moe");
        assert_eq!(pf.context_length, Some(262_144));
        assert_eq!(pf.file_type, Some(15));
        assert_eq!(pf.split, None);
        assert!(pf.gguf.data_offset.is_some());
    }

    #[test]
    fn any_architecture_with_the_required_keys_passes()
    {
        // Whether dovata can plan it is the layout's question, not preflight's.
        let pf = Preflight::evaluate(&header("mamba2", Some("Some-New-Model"))).expect("preflights");
        assert_eq!(pf.architecture, "mamba2");
    }

    #[test]
    fn a_missing_required_key_is_reported()
    {
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&5u64.to_le_bytes());
        b.extend_from_slice(b"llama");

        match Preflight::evaluate(&b).unwrap_err() {
            Error::MissingKeys(keys) => {
                assert!(keys.iter().any(|k| k.contains("block_count")), "{keys:?}");
                assert!(keys.iter().any(|k| k.contains("tokenizer.ggml.tokens")), "{keys:?}");
            }
            other => panic!("expected MissingKeys, got {other:?}"),
        }
    }

    #[test]
    fn truncated_index_is_refused()
    {
        // Declare three tensors and supply none: the range-fetch case.
        let mut bytes = header("llama", None);
        bytes[8..16].copy_from_slice(&3u64.to_le_bytes());
        assert_eq!(Preflight::evaluate(&bytes).unwrap_err(), Error::TruncatedIndex { have: 0, declared: 3 });
    }

    #[test]
    fn a_later_split_part_is_refused_and_names_part_one()
    {
        let bytes = header_with(
            "llama",
            None,
            &[
                ("split.no", 2, 1u16.to_le_bytes().to_vec()),
                ("split.count", 2, 3u16.to_le_bytes().to_vec()),
                ("split.tensors.count", 5, 10i32.to_le_bytes().to_vec()),
            ],
        );
        let err = Preflight::evaluate(&bytes).unwrap_err();
        assert!(matches!(err, Error::NotFirstPart(_)), "{err:?}");
        assert!(err.to_string().contains("part 2 of 3"), "{err}");
    }

    #[test]
    fn a_file_that_is_not_a_gguf_is_rejected_with_a_pointer_to_the_inspector()
    {
        let err = Preflight::evaluate(b"\x7fELF____________").unwrap_err();
        assert!(matches!(err, Error::NotLoadable(_)), "{err:?}");
        assert!(err.to_string().contains("gguf-inspect"), "{err}");
    }

    #[test]
    fn an_oversized_context_is_detectable()
    {
        let pf = Preflight::evaluate(&header("qwen3moe", None)).unwrap();
        assert!(!pf.ctx_exceeds_native(32_768));
        assert!(pf.ctx_exceeds_native(1_000_000));
    }

    #[test]
    fn a_missing_file_is_unreadable()
    {
        let err = Preflight::read(Path::new("/nonexistent/model.gguf")).unwrap_err();
        assert!(matches!(err, Error::Unreadable { .. }), "{err:?}");
    }
}
