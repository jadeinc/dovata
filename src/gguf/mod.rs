//! Author: Joe Ricks
//! File: src/gguf/mod.rs
//! Description: Reads a GGUF model file's header without llama.cpp.

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
/// in ggml's `ggml/include/gguf.h`.
pub const DEFAULT_ALIGNMENT: u64 = 32;

/// Purpose: This enum holds one parsed GGUF metadata value, one variant per GGUF type.
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
    /// Purpose: This function returns a metadata value as a string, if it is one.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the string, or `None` for any other type.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Purpose: This function returns a metadata value as a `u64`, whatever integer width
    /// it was written at.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the number, or `None` for a negative integer or a value that is not an
    /// integer.
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

    /// Purpose: This function returns how many elements a metadata array holds.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the count, or `None` when the value is not an array.
    pub fn len(&self) -> Option<usize> {
        match self {
            Value::Array(v) => Some(v.len()),
            _ => None,
        }
    }

    /// Purpose: This function returns the GGUF name of a metadata value's type.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a name such as `u32`, `str` or `array`.
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
    /// Purpose: This function formats a GGUF metadata value as one short line of text.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
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

/// Purpose: This struct holds one entry of a GGUF tensor index: a tensor's name, shape,
/// type and data offset.
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
    /// Purpose: This function returns how many elements a tensor holds.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the product of its dimensions.
    pub fn elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Purpose: This function returns the size of a tensor's data in bytes, as ggml's
    /// `ggml_nbytes` computes it.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the byte count, or `None` when the type is unknown, the first dimension
    /// does not divide into whole blocks, or the size overflows.
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

/// Purpose: This struct holds where a file sits in a model split across several GGUF
/// files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Split {
    /// This file's index, from 0.
    pub no: u16,
    /// How many files the model has.
    pub count: u16,
    /// Tensors across every part.
    pub tensors_count: u64,
}

/// Purpose: This struct holds a parsed GGUF header: its metadata and tensor index.
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
    /// Purpose: This function parses a GGUF header from the bytes at the start of a model
    /// file.
    ///
    /// Inputs:
    /// - `bytes`: a whole GGUF file, or only its first bytes.
    ///
    /// Outputs: the parsed [`Gguf`], or an [`Error`] when the bytes are not a usable GGUF
    /// header.
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

    /// Purpose: This function finds one tensor in the tensor index by name.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    ///
    /// Outputs: the tensor's entry, or `None`.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Purpose: This function returns where a tensor's data starts, counted from the
    /// start of the file.
    ///
    /// Inputs:
    /// - `t`: a tensor from this header's index.
    ///
    /// Outputs: the offset, or `None` when the tensor index is truncated.
    pub fn file_offset(&self, t: &TensorInfo) -> Option<u64> {
        self.data_offset?.checked_add(t.offset)
    }

    /// Purpose: This function returns the file's quantization type from
    /// `general.file_type`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the `llama_ftype` number, or `None` when the key is absent.
    pub fn file_type(&self) -> Option<u32> {
        self.u64(KEY_FILE_TYPE).and_then(|v| u32::try_from(v).ok())
    }

    /// Purpose: This function returns where this file sits in a model split across
    /// several files.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the part number and counts, or `None` when the model is one file.
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

    /// Purpose: This function finds one metadata value by key.
    ///
    /// Inputs:
    /// - `key`: the full metadata key, such as `general.architecture`.
    ///
    /// Outputs: the value, or `None` when the key is absent.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Purpose: This function finds one metadata value by key and returns it as a string.
    ///
    /// Inputs:
    /// - `key`: the full metadata key.
    ///
    /// Outputs: the string, or `None`.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Purpose: This function finds one metadata value by key and returns it as a `u64`.
    ///
    /// Inputs:
    /// - `key`: the full metadata key.
    ///
    /// Outputs: the number, or `None`.
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(Value::as_u64)
    }

    /// Purpose: This function returns the model architecture from `general.architecture`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a name such as `qwen3moe`, or `None` when the key is absent.
    pub fn architecture(&self) -> Option<&str> {
        self.string(KEY_ARCHITECTURE)
    }

    /// Purpose: This function returns the model name from `general.name`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the name, or `None` when the key is absent.
    pub fn name(&self) -> Option<&str> {
        self.string(KEY_NAME)
    }

    /// Purpose: This function lists the metadata keys under the architecture's own
    /// prefix.
    ///
    /// Inputs: none.
    ///
    /// Outputs: each key and value, in file order.
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

    /// Purpose: This function checks whether the file declares a mixture-of-experts
    /// model.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` when `<arch>.expert_count` is above zero.
    pub fn is_moe(&self) -> bool {
        self.architecture()
            .and_then(|a| self.u64(&format!("{a}.expert_count")))
            .is_some_and(|n| n > 0)
    }

    /// Purpose: This function counts how many tensors have each tensor type.
    ///
    /// Inputs: none.
    ///
    /// Outputs: each type name and count, most common first.
    pub fn quant_mix(&self) -> Vec<(&'static str, usize)> {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for t in &self.tensors {
            *counts.entry(ggml_type_name(t.kind)).or_insert(0) += 1;
        }
        let mut v: Vec<_> = counts.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        v
    }

    /// Purpose: This function checks the file against the metadata keys llama.cpp reads.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a [`Report`] of each key, whether it is required, and what was found.
    pub fn report(&self) -> Report {
        Report::of(self)
    }
}

/// Purpose: This struct holds whether one metadata key llama.cpp reads is present in a
/// GGUF file.
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
    /// Purpose: This function checks whether a required metadata key is missing.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` when the key is required and absent.
    pub fn is_blocking(&self) -> bool {
        self.required && self.found.is_none()
    }
}

/// Purpose: This struct holds the result of checking a GGUF file against the metadata
/// keys llama.cpp reads.
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
    /// Purpose: This function builds the report of which llama.cpp metadata keys a header
    /// has.
    ///
    /// Inputs:
    /// - `g`: the parsed GGUF header.
    ///
    /// Outputs: the [`Report`].
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

        // MoE keys, only meaningful when the file says it is MoE. The expert FFN width
        // is required by some architectures, such as deepseek2, and optional in others,
        // such as qwen3moe and Mixtral-style llama, which fall back to the dense width.
        // That is per-architecture, so it is the layout's to require, not this check's.
        if g.is_moe() {
            for (suffix, required, note) in [
                ("expert_count", true, "total experts per MoE layer"),
                ("expert_used_count", true, "experts routed to per token"),
                ("expert_feed_forward_length", false, "per-expert FFN width; some architectures require it"),
            ] {
                let key = format!("{arch}.{suffix}");
                checks.push(Check {
                    found: g.get(&key).map(ToString::to_string),
                    key,
                    required,
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

    /// Purpose: This function lists the required metadata keys that are missing.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the failed required checks; empty when nothing is missing.
    pub fn blocking(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| c.is_blocking()).collect()
    }
}

/// Purpose: This enum lists the errors reading a GGUF header can return.
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
    /// Purpose: This function wraps a header error with the index of the metadata entry
    /// it happened in.
    ///
    /// Inputs:
    /// - `index`: the metadata entry's position, from 0.
    ///
    /// Outputs: an [`Error::AtKv`].
    fn at_kv(self, index: u64) -> Error {
        Error::AtKv { index, inner: Box::new(self) }
    }
}

impl fmt::Display for Error {
    /// Purpose: This function formats GGUF header reading errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
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

/// Purpose: This struct reads values from GGUF header bytes, checking every read against
/// the end of the input.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Purpose: This function reads the next `n` bytes of the header.
    ///
    /// Inputs:
    /// - `n`: how many bytes to read.
    ///
    /// Outputs: the bytes, or an error when the input ends first.
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.pos.checked_add(n).ok_or(Error::Implausible("length overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::Truncated { need: end - self.bytes.len(), have: self.bytes.len() - self.pos });
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    /// Purpose: This function reads a little-endian `u32` from the header.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the number, or an error when the input ends first.
    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Purpose: This function reads a little-endian `u64` from the header.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the number, or an error when the input ends first.
    fn u64(&mut self) -> Result<u64, Error> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// Purpose: This function reads a GGUF string, a length followed by UTF-8 bytes, from
    /// the header.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the string, or an error when it is cut off or not UTF-8.
    fn string(&mut self) -> Result<String, Error> {
        let len = self.u64()?;
        // A length longer than the whole input is either corruption or a wrong-endian
        // file; either way, fail before trying to allocate it.
        let len = usize::try_from(len).map_err(|_| Error::Implausible("string length exceeds usize"))?;
        let b = self.take(len)?;
        std::str::from_utf8(b).map(str::to_owned).map_err(|_| Error::BadUtf8)
    }

    /// Purpose: This function reads one metadata value and its type tag from the header.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the value, or an error.
    fn value(&mut self) -> Result<Value, Error> {
        let tag = self.u32()?;
        self.value_of_type(tag)
    }

    /// Purpose: This function reads one metadata value of a known type from the header.
    ///
    /// Inputs:
    /// - `tag`: the GGUF type tag, 0 to 12.
    ///
    /// Outputs: the value, or an error.
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

    /// Purpose: This function reads one tensor index entry from the header.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the tensor's name, shape, type and offset, or an error.
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

/// Purpose: This function returns the name of a ggml tensor type.
///
/// Inputs:
/// - `t`: the `ggml_type` number.
///
/// Outputs: a name such as `Q4_K`, or `type ?` when unknown.
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

/// Purpose: This function returns how a ggml tensor type packs its elements into blocks.
///
/// Inputs:
/// - `t`: the `ggml_type` number.
///
/// Outputs: `(elements per block, bytes per block)`, or `None` when unknown.
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

/// Purpose: This function shortens a string for display, adding an ellipsis when it cuts.
///
/// Inputs:
/// - `s`: the text.
/// - `max`: the most characters to keep.
///
/// Outputs: the shortened text.
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

    /// Purpose: This function builds a small test GGUF header in memory.
    ///
    /// Inputs:
    /// - `kvs`: metadata entries, as `(key, type tag, encoded value)`.
    /// - `tensor_count`: the tensor count the header declares.
    ///
    /// Outputs: the header bytes, with no tensor index entries.
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

    /// Purpose: This function encodes a string as a test metadata value.
    ///
    /// Inputs:
    /// - `s`: the string.
    ///
    /// Outputs: the type tag and the encoded bytes.
    fn str_val(s: &str) -> (u32, Vec<u8>) {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(s.as_bytes());
        (8, v)
    }

    /// Purpose: This function encodes a `u32` as a test metadata value.
    ///
    /// Inputs:
    /// - `n`: the number.
    ///
    /// Outputs: the type tag and the encoded bytes.
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

    /// Purpose: This function encodes one test tensor index entry the way gguf.cpp writes
    /// it.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    /// - `dims`: its shape.
    /// - `kind`: its `ggml_type` number.
    /// - `offset`: its offset in the data section.
    ///
    /// Outputs: the encoded entry.
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
