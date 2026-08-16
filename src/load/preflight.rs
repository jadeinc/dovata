//! Startup preflight: decide everything the model file can tell us *before* loading it.
//!
//! ## Why this runs before the engine
//!
//! The daemon used to learn what model it was serving at the far end of startup — after
//! `LlamaModel::load_from_file` had pulled the better part of 20 GB into VRAM, the name
//! was read off the loaded model and a profile was globbed from it. Every way that can
//! go wrong therefore cost a full load first: a mistyped `JADE_MODEL_PATH`, a file that
//! is not a GGUF, a model whose profile nobody has written yet.
//!
//! A GGUF stores all of its metadata ahead of the tensor data, so none of those
//! questions need the weights. This module reads the header with the [`gguf`] crate,
//! picks the profile, and fails in under a second with a message that names the actual
//! problem. What llama.cpp then does with the file is unchanged — this is a gate in
//! front of the load, never a replacement for it. llama.cpp remains the only authority
//! on whether a model loads; this module is the authority on which profile applies.
//!
//! ## Where the boundary is
//!
//! Everything here is pure except [`Preflight::read`], which reads bytes off disk. The
//! decision logic takes a byte slice, so it is unit-tested on any platform against
//! synthetic headers — the same split the rest of the crate uses. Only the call site in
//! `host.rs` is Linux-gated, because that is where the local engine exists.
//!
//! ## The unprofiled-model policy
//!
//! A model with no matching profile boots into bare ChatML with no tool-call
//! delimiters. It generates fluent text and every tool call it makes is parsed wrong.
//! That is a silent failure that surfaces much later as "the model is bad at tools", so
//! the default is to refuse the boot and say exactly what is missing. Setting
//! `JADE_ALLOW_UNPROFILED=1` downgrades it to a warning for the case where you are
//! deliberately serving a model for plain completion.

use std::fmt;

use crate::model::{ModelProfile, Registry, Selector};

/// How much of the file to read. A GGUF's metadata and tensor index both sit before the
/// tensor data; 64 MB covers both for every model we have shipped, and it is a cap on a
/// file that may be 20 GB, not a target.
const HEADER_BUDGET: u64 = 64 * 1024 * 1024;

/// What the header said, and which profile it selected.
#[derive(Debug, Clone)]
pub struct Preflight {
    /// `general.architecture` — the string llama.cpp dispatches on.
    pub architecture: Option<String>,
    /// `general.name` — free text, and what the profile globs match against.
    pub name: Option<String>,
    /// The model's native context window, from `<arch>.context_length`.
    pub context_length: Option<u64>,
    /// Whether the file declares a mixture-of-experts model.
    pub is_moe: bool,
    /// The chosen profile, and which identifier chose it. `None` only when the boot
    /// was allowed to continue unprofiled.
    pub profile: Option<(ModelProfile, Selector)>,
}

impl Preflight {
    /// Read the header off disk and evaluate it.
    ///
    /// The only function here that touches the filesystem. `allow_unprofiled` comes
    /// from `JADE_ALLOW_UNPROFILED`; see the module docs for why it defaults off.
    pub fn read(path: &str, registry: &Registry, allow_unprofiled: bool) -> Result<Preflight, Error> {
        use std::io::Read;

        let file = std::fs::File::open(path).map_err(|e| Error::Unreadable {
            path: path.to_owned(),
            detail: e.to_string(),
        })?;
        let mut bytes = Vec::new();
        file.take(HEADER_BUDGET)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::Unreadable { path: path.to_owned(), detail: e.to_string() })?;

        Self::evaluate(&bytes, registry, allow_unprofiled)
    }

    /// Evaluate an already-read header. Pure, so this is what the tests drive.
    pub fn evaluate(bytes: &[u8], registry: &Registry, allow_unprofiled: bool) -> Result<Preflight, Error> {
        let model = crate::gguf::Gguf::parse(bytes).map_err(|e| Error::NotLoadable(e.to_string()))?;

        // Required-key check first: a file missing these cannot load at all, so
        // reporting a profile miss on top of it would just be noise.
        let report = model.report();
        let missing: Vec<String> = report
            .blocking()
            .iter()
            .map(|c| format!("{} — {}", c.key, c.note))
            .collect();
        if !missing.is_empty() {
            return Err(Error::MissingKeys(missing));
        }

        let architecture = model.architecture().map(str::to_owned);
        let name = model.name().map(str::to_owned);
        let context_length = architecture.as_ref().and_then(|a| model.u64(&format!("{a}.context_length")));

        let selected = registry
            .select_for(name.as_deref(), architecture.as_deref())
            .map(|(p, how)| (p.clone(), how));

        if selected.is_none() && !allow_unprofiled {
            return Err(Error::NoProfile {
                architecture: architecture.clone(),
                name: name.clone(),
            });
        }

        Ok(Preflight {
            architecture,
            name,
            context_length,
            is_moe: model.is_moe(),
            profile: selected,
        })
    }

    /// The startup lines to print. Returned rather than printed so the caller owns the
    /// output stream and the tests can assert on the text.
    pub fn summary(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "dovata: model architecture {}",
            self.architecture.as_deref().unwrap_or("(absent)")
        ));
        out.push(format!("dovata: model name {}", self.name.as_deref().unwrap_or("(absent)")));
        if let Some(n) = self.context_length {
            out.push(format!("dovata: native context window {n}"));
        }
        if self.is_moe {
            out.push("dovata: mixture-of-experts model".to_owned());
        }
        match &self.profile {
            Some((p, how)) => out.push(format!("dovata: profile '{}' ({})", p.model, how.describe())),
            None => {
                out.push("dovata: WARNING no profile matches this model".to_owned());
                out.push("dovata: WARNING running bare ChatML — tool calls will parse wrong".to_owned());
                out.push("dovata: WARNING booting anyway because JADE_ALLOW_UNPROFILED is set".to_owned());
            }
        }
        out
    }

    /// Whether the native context window is smaller than the configured one. A context
    /// larger than the weights were trained for does not fail at load; it degrades
    /// quietly once prompts get long, which is exactly the kind of thing to say at boot.
    pub fn ctx_exceeds_native(&self, configured: u32) -> bool {
        self.context_length.is_some_and(|native| u64::from(configured) > native)
    }
}

/// The profile chosen at boot, published for the engine to read back.
///
/// The engine could re-derive this from the loaded model — it has the same two metadata
/// keys available through llama.cpp — but then the file would have two independent
/// selectors that could disagree, and the one that decided whether to boot would not be
/// the one actually serving requests. Publishing the decision keeps it single. Absent
/// when the preflight never ran (unit tests, the `JADE_*` self-test hijacks), in which
/// case the engine falls back to selecting by name as it always did.
static SELECTED: std::sync::OnceLock<Option<ModelProfile>> = std::sync::OnceLock::new();

/// Publish the preflight's choice. Called once, from the host, before the engine loads.
/// Later calls are ignored — the first boot decision stands.
pub fn publish(pf: &Preflight) {
    let _ = SELECTED.set(pf.profile.as_ref().map(|(p, _)| p.clone()));
}

/// The profile the preflight chose, if it ran and found one.
pub fn selected() -> Option<&'static ModelProfile> {
    SELECTED.get().and_then(Option::as_ref)
}

/// Why the preflight refused the model.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The file could not be opened or read.
    Unreadable { path: String, detail: String },
    /// The bytes are not a GGUF header this reader understands.
    NotLoadable(String),
    /// Required metadata is absent, so llama.cpp would fail too — just later.
    MissingKeys(Vec<String>),
    /// No profile matched, and `JADE_ALLOW_UNPROFILED` was not set.
    NoProfile { architecture: Option<String>, name: Option<String> },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unreadable { path, detail } => {
                write!(
                    f,
                    "cannot read the model at {path}: {detail}\n  \
                     JADE_MODEL_PATH must point at a readable .gguf file."
                )
            }
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
            Error::NoProfile { architecture, name } => write!(
                f,
                "no profile matches this model\n  \
                 general.architecture = {}\n  \
                 general.name         = {}\n  \
                 Without a profile the daemon falls back to bare ChatML: it will generate\n  \
                 text fine and parse every tool call wrong, silently.\n  \
                 fix:  cargo run --bin gguf-inspect -- <model> --profile-stub\n  \
                 or:   JADE_ALLOW_UNPROFILED=1 to boot without tool-call support",
                architecture.as_deref().unwrap_or("(absent)"),
                name.as_deref().unwrap_or("(absent)"),
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A GGUF header carrying exactly the keys the required-key check demands, so the
    /// tests can vary the two identifiers without tripping the earlier gate.
    ///
    /// Shared with `crate::load::vram`'s tests: a second hand-rolled GGUF byte builder is
    /// exactly the kind of duplicate that drifts from the real format silently.
    pub(crate) fn header(arch: &str, name: Option<&str>) -> Vec<u8> {
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
    fn the_production_model_selects_its_profile_by_name() {
        let bytes = header("qwen3moe", Some("Qwen3-Coder-30B-A3B-Instruct"));
        let pf = Preflight::evaluate(&bytes, &Registry::builtin(), false).expect("preflights");

        assert_eq!(pf.architecture.as_deref(), Some("qwen3moe"));
        assert_eq!(pf.context_length, Some(262_144));
        let (profile, how) = pf.profile.expect("a profile was selected");
        assert_eq!(profile.model, "Qwen3-Coder-30B");
        assert_eq!(how, Selector::Name, "the name glob is specific, so it must win");
    }

    #[test]
    fn an_unknown_name_on_a_known_architecture_falls_back_to_the_family() {
        // The case this whole module exists for: a later Qwen release whose name matches
        // nothing we ship, but which still declares the architecture we have a profile
        // for. It must boot with that family's tool-call conventions, not bare ChatML.
        let bytes = header("qwen3moe", Some("Qwen3.5-Coder-30B-Instruct"));
        let pf = Preflight::evaluate(&bytes, &Registry::builtin(), false).expect("preflights");

        let (profile, how) = pf.profile.expect("family fallback selected a profile");
        assert_eq!(profile.model, "Qwen3-Coder-30B");
        assert_eq!(how, Selector::Architecture);
    }

    #[test]
    fn an_unknown_architecture_refuses_the_boot() {
        let bytes = header("mamba2", Some("Some-New-Model"));
        let err = Preflight::evaluate(&bytes, &Registry::builtin(), false).unwrap_err();

        match &err {
            Error::NoProfile { architecture, name } => {
                assert_eq!(architecture.as_deref(), Some("mamba2"));
                assert_eq!(name.as_deref(), Some("Some-New-Model"));
            }
            other => panic!("expected NoProfile, got {other:?}"),
        }
        // The message has to carry the fix, not just the complaint.
        let text = err.to_string();
        assert!(text.contains("--profile-stub"), "{text}");
        assert!(text.contains("JADE_ALLOW_UNPROFILED=1"), "{text}");
    }

    #[test]
    fn allow_unprofiled_downgrades_the_refusal_to_a_warning() {
        let bytes = header("mamba2", Some("Some-New-Model"));
        let pf = Preflight::evaluate(&bytes, &Registry::builtin(), true).expect("boots anyway");

        assert!(pf.profile.is_none());
        let summary = pf.summary().join("\n");
        assert!(summary.contains("no profile matches"), "{summary}");
        assert!(summary.contains("tool calls will parse wrong"), "{summary}");
    }

    #[test]
    fn a_missing_required_key_is_reported_before_the_profile_is_considered() {
        // No tokenizer, no hparams — and also no profile. The missing keys are the
        // useful complaint, so they must come first.
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        let k = "general.architecture";
        b.extend_from_slice(&(k.len() as u64).to_le_bytes());
        b.extend_from_slice(k.as_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&7u64.to_le_bytes());
        b.extend_from_slice(b"qwen3moe"[..7].as_ref());

        let err = Preflight::evaluate(&b, &Registry::builtin(), false).unwrap_err();
        match err {
            Error::MissingKeys(keys) => {
                assert!(keys.iter().any(|k| k.contains("block_count")), "{keys:?}");
                assert!(keys.iter().any(|k| k.contains("tokenizer.ggml.tokens")), "{keys:?}");
            }
            other => panic!("expected MissingKeys, got {other:?}"),
        }
    }

    #[test]
    fn a_file_that_is_not_a_gguf_is_rejected_with_a_pointer_to_the_inspector() {
        let err = Preflight::evaluate(b"\x7fELF____________", &Registry::builtin(), false).unwrap_err();
        assert!(matches!(err, Error::NotLoadable(_)), "{err:?}");
        assert!(err.to_string().contains("gguf-inspect"), "{err}");
    }

    #[test]
    fn an_oversized_configured_context_is_detectable() {
        let bytes = header("qwen3moe", Some("Qwen3-Coder-30B-A3B-Instruct"));
        let pf = Preflight::evaluate(&bytes, &Registry::builtin(), false).unwrap();

        assert!(!pf.ctx_exceeds_native(32_768), "the default sits well inside 262144");
        assert!(pf.ctx_exceeds_native(1_000_000));
    }

    #[test]
    fn qwen35moe_selects_the_qwen3_6_profile_by_architecture() {
        // Qwen3.5 and Qwen3.6 both declare qwen35moe but carry names no glob of
        // ours anticipates, so the arch net is what has to catch them.
        for name in ["Qwen3.6-35B-A3B", "Qwen3.5-35B-A3B", "Qwen3.6-35B-A3B-Some-Requant"] {
            let bytes = header("qwen35moe", Some(name));
            let pf = Preflight::evaluate(&bytes, &Registry::builtin(), false)
                .unwrap_or_else(|e| panic!("{name} should preflight: {e}"));
            let (profile, _) = pf.profile.expect("a profile was selected");
            assert_eq!(profile.model, "Qwen3.6-35B-A3B", "for {name}");
            assert_eq!(profile.tool_result.role, "user", "qwen3.6 returns results on the user turn");
        }
    }

    #[test]
    fn the_two_qwen_families_do_not_collide() {
        // qwen3moe and qwen35moe are different architectures with different tool
        // conventions; each must keep selecting its own profile.
        let old = Preflight::evaluate(&header("qwen3moe", Some("Qwen3-Coder-30B-A3B-Instruct")),
                                      &Registry::builtin(), false).unwrap();
        assert_eq!(old.profile.unwrap().0.model, "Qwen3-Coder-30B");

        let new = Preflight::evaluate(&header("qwen35moe", Some("Qwen3.6-35B-A3B")),
                                      &Registry::builtin(), false).unwrap();
        assert_eq!(new.profile.unwrap().0.model, "Qwen3.6-35B-A3B");
    }

    #[test]
    fn a_missing_file_names_the_variable_to_fix() {
        let err = Preflight::read("/nonexistent/model.gguf", &Registry::builtin(), false).unwrap_err();
        assert!(matches!(err, Error::Unreadable { .. }), "{err:?}");
        assert!(err.to_string().contains("JADE_MODEL_PATH"), "{err}");
    }
}
