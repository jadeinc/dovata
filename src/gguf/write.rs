//! Author: Joe Ricks
//! File: src/gguf/write.rs
//! Description: Writes small GGUF files for test fixtures.

use std::collections::HashSet;
use std::fmt;
use std::path::Path;

use super::{ggml_type_layout, TensorInfo, Value, DEFAULT_ALIGNMENT, KEY_ALIGNMENT, MAGIC};

/// The longest tensor name gguf.cpp accepts, in bytes: `GGML_MAX_NAME` less its NUL.
pub const MAX_NAME_LEN: usize = 63;

/// The most dimensions a tensor may have: `GGML_MAX_DIMS`.
pub const MAX_DIMS: usize = 4;

/// The GGUF format version this writes.
pub const VERSION: u32 = 3;

/// Purpose: This struct holds a GGUF file being built in memory for a test fixture.
#[derive(Debug, Clone, Default)]
pub struct Writer {
    metadata: Vec<(String, Value)>,
    tensors: Vec<Tensor>,
}

/// Purpose: This struct holds one tensor added to a [`Writer`], with its data.
#[derive(Debug, Clone)]
struct Tensor {
    name: String,
    dims: Vec<u64>,
    kind: u32,
    data: Vec<u8>,
}

impl Writer {
    /// Purpose: This function creates an empty GGUF file in memory.
    ///
    /// Inputs: none.
    ///
    /// Outputs: an empty [`Writer`].
    pub fn new() -> Writer
    {
        Writer::default()
    }

    /// Purpose: This function sets one metadata key, replacing its value if it is already
    /// set.
    ///
    /// Inputs:
    /// - `key`: the full metadata key.
    /// - `value`: the value.
    ///
    /// Outputs: the writer, for chaining.
    pub fn kv(
        &mut self,
        key: impl Into<String>,
        value: Value,
    ) -> &mut Writer
    {
        let key = key.into();
        match self.metadata.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.metadata.push((key, value)),
        }
        self
    }

    /// Purpose: This function adds a tensor of any ggml type from its raw bytes.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    /// - `dims`: its shape, fastest-varying first.
    /// - `ggml_type`: its `ggml_type` number.
    /// - `data`: its encoded bytes.
    ///
    /// Outputs: the writer, for chaining.
    pub fn tensor(
        &mut self,
        name: impl Into<String>,
        dims: &[u64],
        ggml_type: u32,
        data: Vec<u8>,
    ) -> &mut Writer
    {
        self.tensors.push(Tensor { name: name.into(), dims: dims.to_vec(), kind: ggml_type, data });
        self
    }

    /// Purpose: This function adds an F32 tensor from its values.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    /// - `dims`: its shape, fastest-varying first.
    /// - `values`: its elements.
    ///
    /// Outputs: the writer, for chaining.
    pub fn f32_tensor(
        &mut self,
        name: impl Into<String>,
        dims: &[u64],
        values: &[f32],
    ) -> &mut Writer
    {
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(name, dims, 0, data)
    }

    /// Purpose: This function lays out the whole GGUF file in memory.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the file's bytes, or a [`WriteError`] for anything a GGUF reader would
    /// refuse.
    pub fn to_bytes(&self) -> Result<Vec<u8>, WriteError>
    {
        let alignment = self.alignment()?;

        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        b.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            put_string(&mut b, key);
            b.extend_from_slice(&tag(value).to_le_bytes());
            put_payload(&mut b, key, value)?;
        }

        let mut seen = HashSet::new();
        let mut offset = 0u64;
        for t in &self.tensors {
            if !seen.insert(t.name.as_str()) {
                return Err(WriteError::DuplicateTensor(t.name.clone()));
            }
            let size = check_tensor(t)?;
            put_string(&mut b, &t.name);
            b.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&t.kind.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
            offset += size.div_ceil(alignment) * alignment;
        }

        // gguf.cpp pads after the index even when there are no tensors.
        pad(&mut b, alignment);
        for t in &self.tensors {
            b.extend_from_slice(&t.data);
            pad(&mut b, alignment);
        }
        Ok(b)
    }

    /// Purpose: This function lays out the GGUF file and writes it to disk.
    ///
    /// Inputs:
    /// - `path`: the file to write.
    ///
    /// Outputs: `Ok(())`, or a [`WriteError`].
    pub fn write(
        &self,
        path: &Path,
    ) -> Result<(), WriteError>
    {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|e| WriteError::Io { path: path.display().to_string(), detail: e.to_string() })
    }

    /// Purpose: This function returns the alignment the file's tensor data will use.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `general.alignment` if set, else 32, or an error when the key is not a
    /// `u32` power of two.
    fn alignment(&self) -> Result<u64, WriteError>
    {
        match self.metadata.iter().find(|(k, _)| k == KEY_ALIGNMENT) {
            None => Ok(DEFAULT_ALIGNMENT),
            Some((_, Value::U32(a))) if a.is_power_of_two() => Ok(u64::from(*a)),
            Some(_) => Err(WriteError::BadAlignment),
        }
    }
}

/// Purpose: This function checks one tensor against what a GGUF reader accepts.
///
/// Inputs:
/// - `t`: the tensor.
///
/// Outputs: its data size, or a [`WriteError`].
fn check_tensor(
    t: &Tensor,
) -> Result<u64, WriteError>
{
    if t.name.len() > MAX_NAME_LEN {
        return Err(WriteError::NameTooLong(t.name.clone()));
    }
    if t.dims.len() > MAX_DIMS {
        return Err(WriteError::TooManyDims { name: t.name.clone(), dims: t.dims.len() });
    }
    if ggml_type_layout(t.kind).is_none() {
        return Err(WriteError::UnknownType { name: t.name.clone(), ggml_type: t.kind });
    }
    let info = TensorInfo { name: t.name.clone(), dims: t.dims.clone(), kind: t.kind, offset: 0 };
    let expected = info.bytes().ok_or_else(|| WriteError::BadShape { name: t.name.clone(), dims: t.dims.clone() })?;
    if t.data.len() as u64 != expected {
        return Err(WriteError::DataSize { name: t.name.clone(), expected, got: t.data.len() as u64 });
    }
    Ok(expected)
}

/// Purpose: This function returns the GGUF type tag of a metadata value.
///
/// Inputs:
/// - `v`: the value.
///
/// Outputs: the tag, 0 to 12.
fn tag(
    v: &Value,
) -> u32
{
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

/// Purpose: This function appends a GGUF string to the file bytes.
///
/// Inputs:
/// - `b`: the file bytes.
/// - `s`: the string.
///
/// Outputs: none.
fn put_string(
    b: &mut Vec<u8>,
    s: &str,
)
{
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

/// Purpose: This function appends a metadata value's bytes to the file.
///
/// Inputs:
/// - `b`: the file bytes.
/// - `key`: the value's key, for error messages.
/// - `v`: the value.
///
/// Outputs: `Ok(())`, or an error for an empty, nested or mixed array.
fn put_payload(
    b: &mut Vec<u8>,
    key: &str,
    v: &Value,
) -> Result<(), WriteError>
{
    match v {
        Value::U8(x) => b.push(*x),
        Value::I8(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::U64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => b.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => b.push(u8::from(*x)),
        Value::String(s) => put_string(b, s),
        Value::Array(items) => {
            let first = items.first().ok_or_else(|| WriteError::EmptyArray(key.to_owned()))?;
            let elem = tag(first);
            if elem == 9 {
                return Err(WriteError::NestedArray(key.to_owned()));
            }
            if items.iter().any(|i| tag(i) != elem) {
                return Err(WriteError::MixedArray(key.to_owned()));
            }
            b.extend_from_slice(&elem.to_le_bytes());
            b.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_payload(b, key, item)?;
            }
        }
    }
    Ok(())
}

/// Purpose: This function appends zero bytes up to the next multiple of the alignment.
///
/// Inputs:
/// - `b`: the file bytes.
/// - `alignment`: the alignment.
///
/// Outputs: none.
fn pad(
    b: &mut Vec<u8>,
    alignment: u64,
)
{
    let len = b.len() as u64;
    b.resize(len.div_ceil(alignment).saturating_mul(alignment) as usize, 0);
}

/// Purpose: This enum lists the errors writing a GGUF file can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteError {
    /// `general.alignment` is not a `u32` power of two.
    BadAlignment,
    /// An array with no elements, so no element type to write.
    EmptyArray(String),
    /// An array whose elements are not all the same type.
    MixedArray(String),
    /// An array of arrays, which neither gguf.cpp nor the reader accepts.
    NestedArray(String),
    /// Two tensors with one name.
    DuplicateTensor(String),
    /// A tensor name longer than [`MAX_NAME_LEN`] bytes.
    NameTooLong(String),
    /// A tensor with more than [`MAX_DIMS`] dimensions.
    TooManyDims { name: String, dims: usize },
    /// A `ggml_type` with no known block layout, so no size.
    UnknownType { name: String, ggml_type: u32 },
    /// A shape whose first dimension does not divide into whole blocks of the type, or
    /// whose size overflows.
    BadShape { name: String, dims: Vec<u64> },
    /// The data is not the size the shape and type call for.
    DataSize { name: String, expected: u64, got: u64 },
    /// The file could not be written.
    Io { path: String, detail: String },
}

impl fmt::Display for WriteError {
    /// Purpose: This function formats GGUF writer errors.
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
            WriteError::BadAlignment => write!(f, "{KEY_ALIGNMENT} must be a u32 power of two"),
            WriteError::EmptyArray(k) => write!(f, "metadata {k} is an empty array, which has no element type to write"),
            WriteError::MixedArray(k) => write!(f, "metadata {k} is an array whose elements are not all one type"),
            WriteError::NestedArray(k) => write!(f, "metadata {k} is an array of arrays, which GGUF readers refuse"),
            WriteError::DuplicateTensor(n) => write!(f, "tensor {n} is added twice"),
            WriteError::NameTooLong(n) => write!(f, "tensor name {n} is longer than {MAX_NAME_LEN} bytes"),
            WriteError::TooManyDims { name, dims } => write!(f, "tensor {name} has {dims} dimensions; the most is {MAX_DIMS}"),
            WriteError::UnknownType { name, ggml_type } => {
                write!(f, "tensor {name} has ggml type {ggml_type}, whose block size is not known")
            }
            WriteError::BadShape { name, dims } => {
                write!(f, "tensor {name} has shape {dims:?}, which does not divide into whole blocks of its type")
            }
            WriteError::DataSize { name, expected, got } => {
                write!(f, "tensor {name} needs {expected} bytes of data but was given {got}")
            }
            WriteError::Io { path, detail } => write!(f, "cannot write {path}: {detail}"),
        }
    }
}

impl std::error::Error for WriteError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;

    /// Purpose: This function lists one test metadata value of every GGUF type.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the keys and values.
    fn every_value_type() -> Vec<(&'static str, Value)>
    {
        vec![
            ("t.u8", Value::U8(200)),
            ("t.i8", Value::I8(-5)),
            ("t.u16", Value::U16(65_000)),
            ("t.i16", Value::I16(-30_000)),
            ("t.u32", Value::U32(4_000_000_000)),
            ("t.i32", Value::I32(-2_000_000_000)),
            ("t.u64", Value::U64(u64::MAX)),
            ("t.i64", Value::I64(i64::MIN)),
            ("t.f32", Value::F32(1.5e-5)),
            ("t.f64", Value::F64(-2.25)),
            ("t.bool", Value::Bool(true)),
            ("t.str", Value::String("qwen3moe\nwith a newline".into())),
            ("t.strs", Value::Array(vec![Value::String("<s>".into()), Value::String("</s>".into())])),
            ("t.f32s", Value::Array(vec![Value::F32(0.5), Value::F32(-0.25)])),
            ("t.i32s", Value::Array(vec![Value::I32(1), Value::I32(3), Value::I32(6)])),
        ]
    }

    #[test]
    fn round_trips_through_the_reader()
    {
        let mut w = Writer::new();
        w.kv("general.architecture", Value::String("llama".into()));
        for (k, v) in every_value_type() {
            w.kv(k, v);
        }
        w.kv(KEY_ALIGNMENT, Value::U32(64));
        let f32_values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let q8 = vec![7u8; 2 * 2 * 34]; // two rows of two Q8_0 blocks
        w.f32_tensor("a", &[4, 3], &f32_values)
            .tensor("b", &[32], 1, vec![1u8; 64])
            .tensor("c", &[64, 2], 8, q8.clone());

        let bytes = w.to_bytes().unwrap();
        let g = Gguf::parse(&bytes).unwrap();

        assert_eq!(g.version, VERSION);
        assert!(!g.truncated);
        assert_eq!(g.alignment, 64);
        assert_eq!(g.metadata.len(), 2 + every_value_type().len());
        for (k, v) in every_value_type() {
            assert_eq!(g.get(k), Some(&v), "{k}");
        }

        let data = g.data_offset.unwrap();
        assert_eq!(data % 64, 0);
        let names: Vec<&str> = g.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"], "tensors keep the order they were added in");
        // Each tensor's offset is the padded total of the ones before it.
        assert_eq!(g.tensors.iter().map(|t| t.offset).collect::<Vec<_>>(), [0, 64, 128]);
        assert_eq!(g.tensor("c").unwrap().dims, [64, 2]);
        assert_eq!(g.tensor("c").unwrap().kind, 8);

        let at = |name: &str| {
            let t = g.tensor(name).unwrap();
            let start = g.file_offset(t).unwrap() as usize;
            &bytes[start..start + t.bytes().unwrap() as usize]
        };
        assert_eq!(at("a"), f32_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
        assert_eq!(at("b"), vec![1u8; 64]);
        assert_eq!(at("c"), q8);
        assert_eq!(bytes.len() as u64, data + 64 + 64 + 192, "the last tensor is padded too");
    }

    #[test]
    fn setting_a_key_twice_replaces_it_in_place()
    {
        let mut w = Writer::new();
        w.kv("a", Value::U32(1)).kv("b", Value::U32(2)).kv("a", Value::U32(3));
        let g = Gguf::parse(&w.to_bytes().unwrap()).unwrap();
        assert_eq!(g.metadata, vec![("a".to_owned(), Value::U32(3)), ("b".to_owned(), Value::U32(2))]);
    }

    #[test]
    fn a_file_with_no_tensors_parses()
    {
        let mut w = Writer::new();
        w.kv("general.architecture", Value::String("llama".into()));
        let g = Gguf::parse(&w.to_bytes().unwrap()).unwrap();
        assert_eq!(g.architecture(), Some("llama"));
        assert!(g.tensors.is_empty());
    }

    #[test]
    fn refuses_what_a_reader_would()
    {
        let err = |w: &Writer| w.to_bytes().unwrap_err();

        let mut w = Writer::new();
        w.f32_tensor("a", &[4], &[0.0; 3]);
        assert_eq!(err(&w), WriteError::DataSize { name: "a".into(), expected: 16, got: 12 });

        let mut w = Writer::new();
        w.f32_tensor("a", &[1], &[0.0]).f32_tensor("a", &[1], &[0.0]);
        assert_eq!(err(&w), WriteError::DuplicateTensor("a".into()));

        let mut w = Writer::new();
        w.tensor("q", &[48], 8, vec![0; 51]);
        assert!(matches!(err(&w), WriteError::BadShape { .. }), "48 is not a whole number of Q8_0 blocks");

        let mut w = Writer::new();
        w.tensor("x", &[1], 4, vec![0; 1]);
        assert!(matches!(err(&w), WriteError::UnknownType { .. }), "type 4 was removed from ggml");

        let mut w = Writer::new();
        w.f32_tensor("n".repeat(MAX_NAME_LEN + 1), &[1], &[0.0]);
        assert!(matches!(err(&w), WriteError::NameTooLong(_)));

        let mut w = Writer::new();
        w.kv("e", Value::Array(vec![]));
        assert_eq!(err(&w), WriteError::EmptyArray("e".into()));

        let mut w = Writer::new();
        w.kv("m", Value::Array(vec![Value::U32(1), Value::I32(1)]));
        assert_eq!(err(&w), WriteError::MixedArray("m".into()));

        let mut w = Writer::new();
        w.kv(KEY_ALIGNMENT, Value::U32(48));
        assert_eq!(err(&w), WriteError::BadAlignment);
    }
}
