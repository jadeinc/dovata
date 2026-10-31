//! A pure-Rust reader for the GGUF header: the metadata, the tensor index, and where
//! the tensor data starts.
//!
//! # Why this exists
//!
//! Everything dovata decides about a model comes from this header. The planner needs
//! every tensor's name, shape, `ggml_type`, byte size and file offset to place it in
//! memory, and the layout binder needs the architecture and hyperparameters to pick and
//! check a layout. None of that needs the tensor data, so none of it needs llama.cpp.
//!
//! It has no dependencies and never reads the tensor data. It runs on a laptop, and,
//! because a GGUF puts all its metadata and its tensor index up front, it works on the
//! first few megabytes of a file fetched with an HTTP range request. You can plan a
//! model before downloading 18 GB of it.
//!
//! # What it deliberately does not do
//!
//! It does not decide whether llama.cpp *supports* an architecture. That answer lives
//! in llama.cpp's `LLM_ARCH_NAMES` table at whatever commit we have pinned, it changes
//! release to release, and a copy of it here would rot into a confident lie. This module
//! reports what the file claims; [`Report`] pairs that with the keys llama.cpp is known
//! to read. Which architectures dovata can *plan* is a separate question, answered by
//! the built-in layouts.
//!
//! # Format
//!
//! GGUF is a header of typed key/value metadata followed by a tensor index and then the
//! tensor data. Little-endian throughout. Versions 2 and 3 use 64-bit lengths and counts;
//! version 1 used 32-bit ones and predates every model we ship, so it is rejected with a
//! clear message rather than half-supported.

use std::collections::BTreeMap;
use std::fmt;

pub mod write;

/// GGUF's four-byte file magic, little-endian: `b"GGUF"`.
const MAGIC: [u8; 4] = *b"GGUF";

/// The metadata key holding the architecture. This is *the* field llama.cpp dispatches
/// on — see [`Gguf::architecture`].
pub const KEY_ARCHITECTURE: &str = "general.architecture";

/// The metadata key holding the human-readable model name. Free text, and used by
/// nothing inside llama.cpp — see [`Gguf::name`].
pub const KEY_NAME: &str = "general.name";

/// The metadata key holding the data-section alignment, a `u32` power of two. Absent
/// in most files, which means [`DEFAULT_ALIGNMENT`].
pub const KEY_ALIGNMENT: &str = "general.alignment";

/// The metadata key holding the `llama_ftype` the file was quantized to: the label for
/// the whole file, such as `Q4_K_M`, as opposed to each tensor's own `ggml_type`.
pub const KEY_FILE_TYPE: &str = "general.file_type";

/// Which part this file is, counting from 0, when a model is split across files.
pub const KEY_SPLIT_NO: &str = "split.no";

/// How many files the model is split across.
pub const KEY_SPLIT_COUNT: &str = "split.count";

/// How many tensors the whole split model has, across every part.
pub const KEY_SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// The alignment GGUF uses when `general.alignment` is absent: `GGUF_DEFAULT_ALIGNMENT`
/// in the vendored `ggml/include/gguf.h`.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// A parsed GGUF metadata value. Mirrors the format's type tags one for one.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// A homogeneous array. Token vocabularies land here and can hold 150k+ entries,
    /// so [`Value`]'s `Display` elides long ones rather than printing a screenful.
    Array(Vec<Value>),
}

impl Value {
    /// The value as a string slice, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value widened to `u64`, for any unsigned or non-negative signed integer.
    /// Hyperparameters are written with whatever width the converter felt like, so
    /// reading them by concrete type is a portability bug waiting to happen.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(u64::from(v)),
            Value::U16(v) => Some(u64::from(v)),
            Value::U32(v) => Some(u64::from(v)),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Number of elements, for an array.
    pub fn len(&self) -> Option<usize> {
        match self {
            Value::Array(v) => Some(v.len()),
            _ => None,
        }
    }

    /// The short type name, as GGUF spells it.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => "u8",
            Value::I8(_) => "i8",
            Value::U16(_) => "u16",
            Value::I16(_) => "i16",
            Value::U32(_) => "u32",
            Value::I32(_) => "i32",
            Value::U64(_) => "u64",
            Value::I64(_) => "i64",
            Value::F32(_) => "f32",
            Value::F64(_) => "f64",
            Value::Bool(_) => "bool",
            Value::String(_) => "str",
            Value::Array(_) => "array",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            // Chat templates are multi-line Jinja; collapse so one KV stays one line.
            Value::String(s) => write!(f, "{}", elide(&s.replace('\n', "\\n"), 120)),
            Value::Array(items) => {
                write!(f, "[{} items]", items.len())?;
                if let Some(first) = items.first() {
                    write!(f, " e.g. {}", elide(&first.to_string(), 40))?;
                }
                Ok(())
            }
        }
    }
}

/// One entry of the tensor index: enough to know what a tensor is called, how big it is,
/// and what type it was quantized to. The tensor *data* is never read.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Name, e.g. `blk.0.attn_q.weight`. llama.cpp matches these against the per-arch
    /// tensor mapping, so a right-arch file with wrong names still fails to load.
    pub name: String,
    /// Shape, fastest-varying dimension first.
    pub dims: Vec<u64>,
    /// The raw `ggml_type` tag. See [`ggml_type_name`].
    pub kind: u32,
    /// Byte offset into the tensor-data section, not into the file. Add
    /// [`Gguf::data_offset`] for the file offset, or use [`Gguf::file_offset`].
    pub offset: u64,
}

impl TensorInfo {
    /// Element count, the product of the dimensions.
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Size of the tensor's data in bytes, the way ggml's `ggml_nbytes` computes it for
    /// a contiguous tensor.
    ///
    /// Quantized types store whole blocks, so the first dimension has to be a multiple
    /// of the block size; ggml refuses a row that is not. `None` when the type is one
    /// [`ggml_type_layout`] does not know, when the first dimension does not divide into
    /// blocks, or when the product overflows. A guessed size here would place the next
    /// tensor on top of this one.
    pub fn bytes(&self) -> Option<u64> {
        let (block_elems, block_bytes) = ggml_type_layout(self.kind)?;
        let row = *self.dims.first().unwrap_or(&1);
        if row % block_elems != 0 {
            return None;
        }
        let rows = self.dims.iter().skip(1).try_fold(1u64, |acc, d| acc.checked_mul(*d))?;
        (row / block_elems).checked_mul(block_bytes)?.checked_mul(rows)
    }
}

/// Where a file sits in a split model. Large uploads come as several files named
/// `<name>-00001-of-00003.gguf`, written by llama.cpp's `gguf-split`. Only part 0 carries
/// the model's metadata; every part carries its own tensor index and data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// This file's index, from 0.
    pub no: u16,
    /// How many files the model has.
    pub count: u16,
    /// Tensors across every part.
    pub tensors_count: u64,
}

/// A parsed GGUF header.
#[derive(Debug, Clone)]
pub struct Gguf {
    /// Format version. 2 and 3 are supported.
    pub version: u32,
    /// Tensor count as declared in the header — not necessarily `tensors.len()`,
    /// see [`Gguf::truncated`].
    pub tensor_count: u64,
    /// Metadata in file order. A `Vec` rather than a map because order is meaningful
    /// when you are eyeballing a dump against another file's.
    pub metadata: Vec<(String, Value)>,
    /// The tensor index. Short of `tensor_count` when the input was truncated.
    pub tensors: Vec<TensorInfo>,
    /// True when the input ran out before the tensor index was complete — the normal
    /// case for a range-fetched header. Metadata is still complete and trustworthy;
    /// only the tensor list is partial.
    pub truncated: bool,
    /// The data-section alignment: `general.alignment`, or [`DEFAULT_ALIGNMENT`].
    /// Every tensor's [`TensorInfo::offset`] is a multiple of it.
    pub alignment: u64,
    /// File offset where the tensor data starts: the end of the tensor index, rounded
    /// up to [`Gguf::alignment`]. `None` when the index was truncated, since then the
    /// end of it is not known.
    pub data_offset: Option<u64>,
}

impl Gguf {
    /// Parse a GGUF header from bytes.
    ///
    /// The input may be a whole model file or just its leading bytes. Metadata must be
    /// complete or this errors; a tensor index cut short only sets [`Gguf::truncated`],
    /// because reading a header over an HTTP range request is a first-class use.
    pub fn parse(bytes: &[u8]) -> Result<Gguf, Error> {
        let mut r = Reader { bytes, pos: 0 };

        let magic = r.take(4)?;
        if magic != MAGIC {
            return Err(Error::NotGguf {
                found: [magic[0], magic[1], magic[2], magic[3]],
            });
        }
        let version = r.u32()?;
        if version < 2 {
            return Err(Error::UnsupportedVersion(version));
        }

        let tensor_count = r.u64()?;
        let kv_count = r.u64()?;

        let mut metadata = Vec::with_capacity(kv_count.min(4096) as usize);
        for i in 0..kv_count {
            let key = r.string().map_err(|e| e.at_kv(i))?;
            let value = r.value().map_err(|e| e.at_kv(i))?;
            metadata.push((key, value));
        }

        // Past this point a short read means "you handed us a truncated file", which is
        // expected and fine — we already have every byte that matters.
        let mut tensors = Vec::new();
        let mut truncated = false;
        for _ in 0..tensor_count {
            match r.tensor_info() {
                Ok(t) => tensors.push(t),
                Err(_) => {
                    truncated = true;
                    break;
                }
            }
        }

        // gguf.cpp reads `general.alignment` as a u32 and refuses anything that is not a
        // power of two. A file that gets this wrong cannot be loaded, so it cannot be
        // planned either.
        let alignment = match metadata.iter().find(|(k, _)| k == KEY_ALIGNMENT) {
            None => DEFAULT_ALIGNMENT,
            Some((_, Value::U32(a))) if a.is_power_of_two() => u64::from(*a),
            Some(_) => return Err(Error::Implausible("general.alignment is not a u32 power of two")),
        };

        // The data section starts at the next multiple of the alignment after the index.
        // gguf.cpp pads only when there are tensors, so a tensorless file is not padded.
        let data_offset = if truncated {
            None
        } else if tensor_count == 0 {
            Some(r.pos as u64)
        } else {
            Some((r.pos as u64).div_ceil(alignment) * alignment)
        };

        Ok(Gguf { version, tensor_count, metadata, tensors, truncated, alignment, data_offset })
    }

    /// Look up one tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Where a tensor's data starts in the file: [`Gguf::data_offset`] plus the
    /// tensor's own offset. `None` for a truncated index.
    pub fn file_offset(&self, t: &TensorInfo) -> Option<u64> {
        self.data_offset?.checked_add(t.offset)
    }

    /// The `llama_ftype` the file declares in `general.file_type`, such as 15 for
    /// `Q4_K_M`. A label for the file as a whole; individual tensors keep their own
    /// types, and a "Q4_K_M" file still holds F32 norms and Q6_K output weights.
    pub fn file_type(&self) -> Option<u32> {
        self.u64(KEY_FILE_TYPE).and_then(|v| u32::try_from(v).ok())
    }

    /// Where this file sits in a split model, or `None` when the model is one file.
    /// A file that declares `split.count` of 1 is not split either.
    pub fn split(&self) -> Option<Split> {
        let count = u16::try_from(self.u64(KEY_SPLIT_COUNT)?).ok()?;
        if count <= 1 {
            return None;
        }
        Some(Split {
            no: u16::try_from(self.u64(KEY_SPLIT_NO)?).ok()?,
            count,
            tensors_count: self.u64(KEY_SPLIT_TENSORS_COUNT)?,
        })
    }

    /// Look up one metadata key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// A metadata key as a string.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// A metadata key as an unsigned integer, whatever width it was written at.
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    /// The architecture string — `general.architecture`.
    ///
    /// This is the field that decides whether the file loads at all. llama.cpp looks it
    /// up in `LLM_ARCH_NAMES`; a miss is the `unknown model architecture` error, and a
    /// hit selects the hyperparameter keys and tensor-name mapping everything after it
    /// depends on. It is a controlled vocabulary — `qwen3moe`, `llama`, `gemma3` — not
    /// a marketing name, so a "Qwen3.5" release may well still declare `qwen3moe`.
    pub fn architecture(&self) -> Option<&str> {
        self.string(KEY_ARCHITECTURE)
    }

    /// The human-readable name — `general.name`.
    ///
    /// Free text set by whoever ran the conversion. llama.cpp does nothing with it, and
    /// neither does dovata's planning: two GGUFs of the same model from different
    /// publishers often disagree here. Select on [`Gguf::architecture`] instead.
    pub fn name(&self) -> Option<&str> {
        self.string(KEY_NAME)
    }

    /// Every metadata key under the architecture's own namespace, in file order.
    /// These are the hyperparameters llama.cpp reads once it knows the arch.
    pub fn arch_params(&self) -> Vec<(&str, &Value)> {
        let Some(arch) = self.architecture() else {
            return Vec::new();
        };
        let prefix = format!("{arch}.");
        self.metadata
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| (k.as_str(), v))
            .collect()
    }

    /// True when the file declares a mixture-of-experts model, i.e. it carries a
    /// non-zero expert count. Decided from metadata, not from the arch name, because
    /// several architectures cover both dense and MoE variants.
    pub fn is_moe(&self) -> bool {
        self.architecture()
            .and_then(|a| self.u64(&format!("{a}.expert_count")))
            .is_some_and(|n| n > 0)
    }

    /// Quantization mix: how many tensors of each `ggml_type`, most common first.
    /// Empty when the tensor index was truncated away.
    pub fn quant_mix(&self) -> Vec<(&'static str, usize)> {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for t in &self.tensors {
            *counts.entry(ggml_type_name(t.kind)).or_insert(0) += 1;
        }
        let mut v: Vec<_> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }

    /// Check the file against the keys llama.cpp reads for its declared architecture.
    pub fn report(&self) -> Report {
        Report::of(self)
    }
}

/// Whether a key llama.cpp reads is present, and what happens when it is not.
#[derive(Debug, Clone, PartialEq)]
pub struct Check {
    /// The fully-qualified metadata key, with the architecture already substituted in.
    pub key: String,
    /// Missing this key stops the load.
    pub required: bool,
    /// The value found, rendered — `None` when absent.
    pub found: Option<String>,
    /// What the key is for, or what llama.cpp falls back to without it.
    pub note: &'static str,
}

impl Check {
    /// A required key that is not there.
    pub fn is_blocking(&self) -> bool {
        self.required && self.found.is_none()
    }
}

/// The result of checking a file against what llama.cpp reads.
///
/// This is a *guide*, not a transcription of the loader. The set of keys any one
/// architecture demands lives in llama.cpp's `llm_load_hparams` and moves; what is
/// listed here is the core every architecture in the llama family shares, plus the MoE
/// and tokenizer keys. A clean report means "nothing obviously missing", never
/// "guaranteed to load".
#[derive(Debug, Clone)]
pub struct Report {
    /// The declared architecture, if any.
    pub architecture: Option<String>,
    /// The declared name, if any.
    pub name: Option<String>,
    /// Per-key findings, in the order a human wants to read them.
    pub checks: Vec<Check>,
}

impl Report {
    fn of(g: &Gguf) -> Report {
        let arch = g.architecture().map(str::to_owned);
        let mut checks = Vec::new();

        // Architecture first: everything below is chosen by it.
        checks.push(Check {
            key: KEY_ARCHITECTURE.to_owned(),
            required: true,
            found: g.architecture().map(str::to_owned),
            note: "selects the hparam keys and tensor mapping; must exist in llama.cpp's LLM_ARCH_NAMES",
        });

        let Some(arch) = arch.clone() else {
            return Report { architecture: None, name: g.name().map(str::to_owned), checks };
        };

        // The hyperparameters shared across the llama-family architectures. An arch can
        // demand more than this; none of them demand less.
        let core: &[(&str, bool, &'static str)] = &[
            ("block_count", true, "number of transformer layers"),
            ("context_length", true, "native context window the weights were trained for"),
            ("embedding_length", true, "model dimension"),
            ("attention.head_count", true, "attention heads"),
            ("attention.head_count_kv", false, "GQA groups; defaults to head_count when absent"),
            ("feed_forward_length", false, "dense FFN width; MoE models carry expert_feed_forward_length instead"),
            ("attention.layer_norm_rms_epsilon", false, "RMS-norm epsilon; the loader has a default but it is model-specific"),
            ("rope.freq_base", false, "RoPE base frequency; defaults to 10000"),
            ("rope.dimension_count", false, "rotary width; derived from head dim when absent"),
        ];
        for (suffix, required, note) in core {
            let key = format!("{arch}.{suffix}");
            checks.push(Check {
                found: g.get(&key).map(ToString::to_string),
                key,
                required: *required,
                note,
            });
        }

        // MoE keys, only meaningful — and only required — when the file says it is MoE.
        if g.is_moe() {
            for (suffix, note) in [
                ("expert_count", "total experts per MoE layer"),
                ("expert_used_count", "experts routed to per token"),
                ("expert_feed_forward_length", "per-expert FFN width"),
            ] {
                let key = format!("{arch}.{suffix}");
                checks.push(Check {
                    found: g.get(&key).map(ToString::to_string),
                    key,
                    required: true,
                    note,
                });
            }
        }

        // The tokenizer is loaded from the same header and fails just as hard.
        for (key, required, note) in [
            ("tokenizer.ggml.model", true, "tokenizer family, e.g. gpt2 for BPE"),
            ("tokenizer.ggml.tokens", true, "the vocabulary itself"),
            ("tokenizer.ggml.token_type", false, "per-token type flags"),
            ("tokenizer.ggml.merges", false, "BPE merge table; required for BPE tokenizers"),
            ("tokenizer.ggml.bos_token_id", false, "beginning-of-sequence id"),
            ("tokenizer.ggml.eos_token_id", false, "end-of-sequence id; wrong here means generation never stops"),
            ("tokenizer.chat_template", false, "the publisher's Jinja chat template; dovata-chat renders it"),
        ] {
            checks.push(Check {
                key: key.to_owned(),
                required,
                found: g.get(key).map(ToString::to_string),
                note,
            });
        }

        Report { architecture: Some(arch), name: g.name().map(str::to_owned), checks }
    }

    /// Required keys that are missing. Empty is the good case.
    pub fn blocking(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| c.is_blocking()).collect()
    }
}

/// What went wrong reading a header.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// The first four bytes were not `GGUF`.
    NotGguf { found: [u8; 4] },
    /// Version 1, or something newer than this reader understands.
    UnsupportedVersion(u32),
    /// The input ended inside the metadata section.
    Truncated { need: usize, have: usize },
    /// A value carried a type tag that is not in the spec.
    BadValueType(u32),
    /// A string was not valid UTF-8.
    BadUtf8,
    /// An array declared more elements than could possibly be present.
    Implausible(&'static str),
    /// Any of the above, located to a metadata index.
    AtKv { index: u64, inner: Box<Error> },
}

impl Error {
    fn at_kv(self, index: u64) -> Error {
        Error::AtKv { index, inner: Box::new(self) }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotGguf { found } => {
                write!(f, "not a GGUF file: magic is {found:02x?}, expected \"GGUF\"")
            }
            Error::UnsupportedVersion(v) => write!(
                f,
                "GGUF version {v} is not supported (this reader handles 2 and 3; \
                 version 1 predates every model we ship)"
            ),
            Error::Truncated { need, have } => write!(
                f,
                "input ended mid-header: needed {need} more bytes, had {have}. \
                 If this is a range-fetched header, fetch more of the file."
            ),
            Error::BadValueType(t) => write!(f, "unknown metadata value type {t}"),
            Error::BadUtf8 => write!(f, "metadata string is not valid UTF-8"),
            Error::Implausible(what) => write!(f, "implausible header: {what}"),
            Error::AtKv { index, inner } => write!(f, "metadata entry {index}: {inner}"),
        }
    }
}

impl std::error::Error for Error {}

/// A cursor over the header bytes. Every read is bounds-checked; nothing panics on a
/// malformed or truncated file, because "point this at a half-downloaded model" is a
/// thing people will do.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Implausible("length overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::Truncated { need: end - self.bytes.len(), have: self.bytes.len() - self.pos });
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()?;
        // A length longer than the whole input is either corruption or a wrong-endian
        // file; either way, fail before trying to allocate it.
        let len = usize::try_from(len).map_err(|_| Error::Implausible("string length exceeds usize"))?;
        let b = self.take(len)?;
        std::str::from_utf8(b).map(str::to_owned).map_err(|_| Error::BadUtf8)
    }

    fn value(&mut self) -> Result<Value, Error> {
        let tag = self.u32()?;
        self.value_of_type(tag)
    }

    fn value_of_type(&mut self, tag: u32) -> Result<Value, Error> {
        Ok(match tag {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => {
                let b = self.take(2)?;
                Value::U16(u16::from_le_bytes([b[0], b[1]]))
            }
            3 => {
                let b = self.take(2)?;
                Value::I16(i16::from_le_bytes([b[0], b[1]]))
            }
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let elem = self.u32()?;
                if elem == 9 {
                    // The spec has no nesting rule and llama.cpp never writes one; a
                    // nested array here means we have lost the plot, so stop.
                    return Err(Error::Implausible("nested arrays are not supported"));
                }
                let count = self.u64()?;
                let count =
                    usize::try_from(count).map_err(|_| Error::Implausible("array length exceeds usize"))?;
                // Guard the allocation: even a 1-byte element type cannot have more
                // elements than there are bytes left.
                if count > self.bytes.len().saturating_sub(self.pos) {
                    return Err(Error::Implausible("array declares more elements than the file has bytes"));
                }
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.value_of_type(elem)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            other => return Err(Error::BadValueType(other)),
        })
    }

    fn tensor_info(&mut self) -> Result<TensorInfo, Error> {
        let name = self.string()?;
        let n_dims = self.u32()?;
        if n_dims > 4 {
            return Err(Error::Implausible("tensor has more than 4 dimensions"));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(self.u64()?);
        }
        let kind = self.u32()?;
        let offset = self.u64()?;
        Ok(TensorInfo { name, dims, kind, offset })
    }
}

/// Name for a `ggml_type` tag.
///
/// Only the types we might actually meet in a published GGUF are named. The enum grows
/// upstream every few releases, so an unknown tag renders as `type N` rather than being
/// guessed at — a wrong quant name in a report is worse than an honest number.
pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        _ => "type ?",
    }
}

/// How a `ggml_type` stores its elements: `(elements per block, bytes per block)`.
///
/// Copied from `type_traits` in the vendored `ggml/src/ggml.c`, read through
/// `ggml_get_type_traits` rather than worked out from the struct definitions. Plain
/// types are blocks of one element. Retired tags, such as 4, 5 and 31 to 33, return
/// `None`, and so does any tag newer than this table: a size that is wrong by a few
/// bytes per block makes every offset after it wrong.
pub fn ggml_type_layout(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        _ => return None,
    })
}

/// Truncate for display, with an ellipsis when something was cut.
fn elide(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal but well-formed GGUF header in memory, so the parser is tested
    /// without a model file. Values are `(key, type tag, encoded bytes)`.
    fn build(kvs: &[(&str, u32, Vec<u8>)], tensor_count: u64) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&tensor_count.to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, tag, payload) in kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(payload);
        }
        b
    }

    fn str_val(s: &str) -> (u32, Vec<u8>) {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        (8, v)
    }

    fn u32_val(n: u32) -> (u32, Vec<u8>) {
        (4, n.to_le_bytes().to_vec())
    }

    #[test]
    fn parses_a_minimal_header() {
        let (at, av) = str_val("qwen3moe");
        let (nt, nv) = str_val("Qwen3-Coder-30B-A3B-Instruct");
        let (bt, bv) = u32_val(48);
        let bytes = build(
            &[
                ("general.architecture", at, av),
                ("general.name", nt, nv),
                ("qwen3moe.block_count", bt, bv),
            ],
            0,
        );

        let g = Gguf::parse(&bytes).expect("parses");
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("qwen3moe"));
        assert_eq!(g.name(), Some("Qwen3-Coder-30B-A3B-Instruct"));
        assert_eq!(g.u64("qwen3moe.block_count"), Some(48));
        assert!(!g.truncated);
    }

    #[test]
    fn arch_params_are_namespaced_by_architecture() {
        let (at, av) = str_val("qwen3moe");
        let (bt, bv) = u32_val(48);
        let (ht, hv) = u32_val(32);
        let bytes = build(
            &[
                ("general.architecture", at, av),
                ("qwen3moe.block_count", bt, bv),
                ("general.file_type", ht, hv),
            ],
            0,
        );
        let g = Gguf::parse(&bytes).unwrap();
        let params = g.arch_params();
        assert_eq!(params.len(), 1, "only the arch-prefixed key counts as an hparam");
        assert_eq!(params[0].0, "qwen3moe.block_count");
    }

    #[test]
    fn rejects_a_non_gguf_file() {
        let err = Gguf::parse(b"\x7fELF____________").unwrap_err();
        assert!(matches!(err, Error::NotGguf { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_version_one() {
        let mut b = MAGIC.to_vec();
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(Gguf::parse(&b).unwrap_err(), Error::UnsupportedVersion(1));
    }

    #[test]
    fn truncated_metadata_is_an_error_but_a_truncated_tensor_index_is_not() {
        let (at, av) = str_val("llama");
        let full = build(&[("general.architecture", at, av)], 3);

        // Cut inside the metadata: not recoverable, the caller fetched too little.
        let cut = &full[..full.len() - 4];
        assert!(matches!(Gguf::parse(cut).unwrap_err(), Error::AtKv { .. }));

        // Complete metadata, no tensor index: the range-fetch case. Metadata is good.
        let g = Gguf::parse(&full).expect("header parses without the tensor index");
        assert!(g.truncated);
        assert_eq!(g.tensor_count, 3);
        assert!(g.tensors.is_empty());
        assert_eq!(g.architecture(), Some("llama"));
    }

    #[test]
    fn an_absurd_array_length_does_not_allocate() {
        let mut payload = 8u32.to_le_bytes().to_vec(); // element type: string
        payload.extend_from_slice(&u64::MAX.to_le_bytes()); // count
        let bytes = build(&[("tokenizer.ggml.tokens", 9, payload)], 0);
        let err = Gguf::parse(&bytes).unwrap_err();
        assert!(matches!(err, Error::AtKv { .. }), "got {err:?}");
    }

    #[test]
    fn report_flags_a_missing_required_key() {
        let (at, av) = str_val("llama");
        let bytes = build(&[("general.architecture", at, av)], 0);
        let g = Gguf::parse(&bytes).unwrap();
        let report = g.report();

        let blocking: Vec<&str> = report.blocking().iter().map(|c| c.key.as_str()).collect();
        assert!(blocking.contains(&"llama.block_count"));
        assert!(blocking.contains(&"tokenizer.ggml.tokens"));
        // The architecture itself is present, so it must not be flagged.
        assert!(!blocking.contains(&"general.architecture"));
    }

    #[test]
    fn moe_keys_are_only_required_when_the_file_says_moe() {
        let (at, av) = str_val("qwen3moe");
        let (et, ev) = u32_val(128);
        let dense = Gguf::parse(&build(&[("general.architecture", at, av.clone())], 0)).unwrap();
        assert!(!dense.is_moe());
        assert!(!dense.report().checks.iter().any(|c| c.key.ends_with("expert_used_count")));

        let moe = Gguf::parse(&build(
            &[("general.architecture", 8, av), ("qwen3moe.expert_count", et, ev)],
            0,
        ))
        .unwrap();
        assert!(moe.is_moe());
        assert!(moe.report().checks.iter().any(|c| c.key == "qwen3moe.expert_used_count"));
    }

    /// One tensor-index entry, encoded the way gguf.cpp writes it.
    fn tensor_entry(name: &str, dims: &[u64], kind: u32, offset: u64) -> Vec<u8> {
        let mut b = (name.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(name.as_bytes());
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        b
    }

    #[test]
    fn data_offset_is_aligned_after_the_index() {
        let (at, av) = str_val("llama");
        let mut bytes = build(&[("general.architecture", at, av)], 2);
        bytes.extend(tensor_entry("token_embd.weight", &[64, 8], 0, 0));
        bytes.extend(tensor_entry("output_norm.weight", &[64], 0, 2048));
        let index_end = bytes.len() as u64;

        let g = Gguf::parse(&bytes).unwrap();
        assert!(!g.truncated);
        assert_eq!(g.alignment, DEFAULT_ALIGNMENT);
        let data = g.data_offset.expect("a complete index has a data offset");
        assert!(data >= index_end && data < index_end + DEFAULT_ALIGNMENT, "{data} vs {index_end}");
        assert_eq!(data % DEFAULT_ALIGNMENT, 0);
        let norm = g.tensor("output_norm.weight").unwrap();
        assert_eq!(g.file_offset(norm), Some(data + 2048));
    }

    #[test]
    fn a_truncated_index_has_no_data_offset() {
        let (at, av) = str_val("llama");
        let mut bytes = build(&[("general.architecture", at, av)], 2);
        bytes.extend(tensor_entry("token_embd.weight", &[64, 8], 0, 0));
        let g = Gguf::parse(&bytes).unwrap();
        assert!(g.truncated);
        assert_eq!(g.data_offset, None);
        assert_eq!(g.file_offset(&g.tensors[0]), None);
    }

    #[test]
    fn a_tensorless_file_is_not_padded() {
        // gguf.cpp pads to the alignment only when there is tensor data to align.
        let (at, av) = str_val("llama");
        let bytes = build(&[("general.architecture", at, av)], 0);
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.data_offset, Some(bytes.len() as u64));
    }

    #[test]
    fn alignment_key_overrides_default() {
        let (at, av) = str_val("llama");
        let (lt, lv) = u32_val(64);
        let mut bytes = build(&[("general.architecture", at, av), ("general.alignment", lt, lv)], 1);
        bytes.extend(tensor_entry("output_norm.weight", &[64], 0, 0));
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.alignment, 64);
        assert_eq!(g.data_offset.unwrap() % 64, 0);
    }

    #[test]
    fn an_alignment_that_is_not_a_power_of_two_is_refused() {
        let (at, av) = str_val("llama");
        let (lt, lv) = u32_val(48);
        let bytes = build(&[("general.architecture", at, av), ("general.alignment", lt, lv)], 0);
        assert!(matches!(Gguf::parse(&bytes).unwrap_err(), Error::Implausible(_)));
    }

    #[test]
    fn tensor_bytes_for_block_types() {
        let t = |dims: &[u64], kind| TensorInfo { name: "t".into(), dims: dims.to_vec(), kind, offset: 0 };
        assert_eq!(t(&[4096, 2], 0).bytes(), Some(4096 * 2 * 4), "F32");
        assert_eq!(t(&[4096, 2], 30).bytes(), Some(4096 * 2 * 2), "BF16");
        assert_eq!(t(&[4096, 2], 8).bytes(), Some(4096 / 32 * 34 * 2), "Q8_0");
        assert_eq!(t(&[4096, 2], 2).bytes(), Some(4096 / 32 * 18 * 2), "Q4_0");
        assert_eq!(t(&[4096, 2], 12).bytes(), Some(4096 / 256 * 144 * 2), "Q4_K");
        // A MoE expert stack: experts are the third dimension.
        assert_eq!(t(&[2048, 768, 128], 12).bytes(), Some(2048 / 256 * 144 * 768 * 128));
        // A row that does not divide into whole blocks has no valid size.
        assert_eq!(t(&[100, 2], 12).bytes(), None);
    }

    #[test]
    fn unknown_type_has_no_size() {
        assert_eq!(ggml_type_layout(4), None, "Q4_2 was removed from ggml");
        assert_eq!(ggml_type_layout(9999), None);
        let t = TensorInfo { name: "t".into(), dims: vec![32], kind: 31, offset: 0 };
        assert_eq!(t.bytes(), None);
    }

    #[test]
    fn every_named_type_has_a_layout() {
        // The two tables are kept side by side; a type with a name and no size would
        // print fine and then fail to plan.
        for t in 0..64 {
            assert_eq!(ggml_type_name(t) != "type ?", ggml_type_layout(t).is_some(), "type {t}");
        }
    }

    #[test]
    fn file_type_and_split_are_read() {
        let (at, av) = str_val("llama");
        let (ft, fv) = u32_val(15);
        let one = Gguf::parse(&build(&[("general.architecture", at, av.clone()), ("general.file_type", ft, fv)], 0)).unwrap();
        assert_eq!(one.file_type(), Some(15));
        assert_eq!(one.split(), None);

        let part = Gguf::parse(&build(
            &[
                ("general.architecture", at, av),
                ("split.no", 2, 1u16.to_le_bytes().to_vec()),
                ("split.count", 2, 3u16.to_le_bytes().to_vec()),
                ("split.tensors.count", 5, 579i32.to_le_bytes().to_vec()),
            ],
            0,
        ))
        .unwrap();
        assert_eq!(part.split(), Some(Split { no: 1, count: 3, tensors_count: 579 }));
    }

    #[test]
    fn integers_read_back_regardless_of_declared_width() {
        let (at, av) = str_val("llama");
        let bytes = build(
            &[
                ("general.architecture", at, av),
                ("llama.block_count", 10, 32u64.to_le_bytes().to_vec()),
                ("llama.embedding_length", 5, 4096i32.to_le_bytes().to_vec()),
            ],
            0,
        );
        let g = Gguf::parse(&bytes).unwrap();
        assert_eq!(g.u64("llama.block_count"), Some(32));
        assert_eq!(g.u64("llama.embedding_length"), Some(4096));
    }
}
