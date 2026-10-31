//! Author: Joe Ricks
//! File: src/load/model.rs
//! Description: Opens a model stored in one GGUF file or split across several.

use std::fmt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::gguf::{Gguf, TensorInfo};
use crate::load::preflight::{self, Preflight};

/// Purpose: This struct holds one tensor of a model, with the part file it is in and its
/// offset there.
#[derive(Debug, Clone, Copy)]
pub struct TensorRef<'a> {
    /// Which part file holds it, from 0. Always 0 for a one-file model.
    pub part: u16,
    /// Its entry in that part's tensor index.
    pub info: &'a TensorInfo,
    /// Where its data starts in that part's file.
    pub file_offset: u64,
}

/// Purpose: This struct holds a model that passed preflight, with every part file's
/// header.
#[derive(Debug, Clone)]
pub struct Model {
    /// Part 1's preflight, including its parsed header.
    pub preflight: Preflight,
    /// Every part's path, part 1 first.
    pub paths: Vec<PathBuf>,
    /// The headers of the parts after the first, in order. Empty for a one-file model.
    rest: Vec<Gguf>,
    /// Each part's identity, part 1 first.
    identity: Vec<PartIdentity>,
}

/// Purpose: This struct holds what identifies one GGUF part file: its header hash and
/// expected size.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct PartIdentity {
    /// SHA-256 of the file's first `data_offset` bytes, the metadata and the tensor
    /// index, in lowercase hex. Any change to a tensor's name, type, shape or offset
    /// changes it. Hashing the whole file would also catch changed weights, at the cost
    /// of reading every byte at startup.
    pub header_sha256: String,
    /// The size the file must have: the data offset plus the tensor data, padded to the
    /// alignment the way gguf.cpp writes it. A cheap check for a cut-off download.
    pub file_bytes: u64,
}

impl PartIdentity {
    /// Purpose: This function computes a part file's header hash and expected file size.
    ///
    /// Inputs:
    /// - `bytes`: the start of the file, holding its whole header.
    /// - `g`: the same bytes, parsed.
    ///
    /// Outputs: the [`PartIdentity`].
    fn of(bytes: &[u8], g: &Gguf) -> PartIdentity {
        let data = g.data_offset.unwrap_or(bytes.len() as u64);
        let header = &bytes[..usize::try_from(data).unwrap_or(bytes.len()).min(bytes.len())];
        let digest = Sha256::digest(header);
        let end = g.tensors.iter().filter_map(|t| t.bytes().map(|b| t.offset + b)).max().unwrap_or(0);
        PartIdentity {
            header_sha256: digest.iter().map(|b| format!("{b:02x}")).collect(),
            file_bytes: data + end.div_ceil(g.alignment) * g.alignment,
        }
    }
}

impl Model {
    /// Purpose: This function opens a model from its file, or from the first part of a
    /// split model.
    ///
    /// Inputs:
    /// - `path`: the model file, or its first part.
    ///
    /// Outputs: the [`Model`], or an [`Error`] when a file is unreadable or does not
    /// belong.
    pub fn open(path: &Path) -> Result<Model, Error> {
        let first = read_head(path).map_err(|detail| preflight::Error::Unreadable { path: path.display().to_string(), detail })?;
        let preflight = Preflight::evaluate(&first)?;
        let paths = match preflight.split {
            None => vec![path.to_path_buf()],
            Some(s) => split_paths(path, s.count).ok_or_else(|| Error::SplitName(path.display().to_string()))?,
        };
        Self::read_rest(preflight, &first, paths)
    }

    /// Purpose: This function opens a model from a list of its part files.
    ///
    /// Inputs:
    /// - `paths`: every part file, first part first.
    ///
    /// Outputs: the [`Model`], or an [`Error`].
    pub fn from_parts(paths: &[PathBuf]) -> Result<Model, Error> {
        let path = paths.first().ok_or(Error::NoParts)?;
        let first = read_head(path).map_err(|detail| preflight::Error::Unreadable { path: path.display().to_string(), detail })?;
        let preflight = Preflight::evaluate(&first)?;
        Self::read_rest(preflight, &first, paths.to_vec())
    }

    /// Purpose: This function reads the headers of every part after the first and checks
    /// the set.
    ///
    /// Inputs:
    /// - `preflight`: the first part's preflight result.
    /// - `first`: the first part's header bytes.
    /// - `paths`: every part's path, first part first.
    ///
    /// Outputs: the [`Model`], or an [`Error`].
    fn read_rest(preflight: Preflight, first: &[u8], paths: Vec<PathBuf>) -> Result<Model, Error> {
        let mut identity = vec![PartIdentity::of(first, &preflight.gguf)];
        let mut rest = Vec::new();
        for p in &paths[1..] {
            let fail = |detail: String| Error::Part { path: p.display().to_string(), detail };
            let bytes = read_head(p).map_err(fail)?;
            let g = Gguf::parse(&bytes).map_err(|e| fail(e.to_string()))?;
            identity.push(PartIdentity::of(&bytes, &g));
            rest.push(g);
        }
        Self::assemble(preflight, paths, rest, identity)
    }

    /// Purpose: This function builds a model from GGUF header bytes already in memory.
    ///
    /// Inputs:
    /// - `parts`: each part's path and header bytes, first part first.
    ///
    /// Outputs: the [`Model`], or an [`Error`].
    pub fn from_headers(parts: &[(PathBuf, Vec<u8>)]) -> Result<Model, Error> {
        let (first_path, first) = parts.first().ok_or(Error::NoParts)?;
        let preflight = Preflight::evaluate(first)?;
        let mut identity = vec![PartIdentity::of(first, &preflight.gguf)];
        let mut rest = Vec::new();
        for (path, bytes) in &parts[1..] {
            let g = Gguf::parse(bytes).map_err(|e| Error::Part { path: path.display().to_string(), detail: e.to_string() })?;
            identity.push(PartIdentity::of(bytes, &g));
            rest.push(g);
        }
        let mut paths = vec![first_path.clone()];
        paths.extend(parts[1..].iter().map(|(p, _)| p.clone()));
        Self::assemble(preflight, paths, rest, identity)
    }

    /// Purpose: This function checks that a set of part files forms one complete model
    /// and builds it.
    ///
    /// Inputs:
    /// - `preflight`: the first part's preflight result.
    /// - `paths`: every part's path.
    /// - `rest`: the parsed headers of the later parts.
    /// - `identity`: every part's identity.
    ///
    /// Outputs: the [`Model`], or an [`Error`] naming the first failed check.
    fn assemble(preflight: Preflight, paths: Vec<PathBuf>, rest: Vec<Gguf>, identity: Vec<PartIdentity>) -> Result<Model, Error> {
        let expected = preflight.split.map_or(1, |s| usize::from(s.count));
        if paths.len() != expected {
            return Err(Error::PartCount { expected, found: paths.len() });
        }
        for (i, g) in rest.iter().enumerate() {
            let path = paths[i + 1].display().to_string();
            let s = g.split().ok_or_else(|| Error::Part { path: path.clone(), detail: "it is not a split part".into() })?;
            if usize::from(s.no) != i + 1 || usize::from(s.count) != expected {
                return Err(Error::Part {
                    path,
                    detail: format!("it says it is part {} of {}, expected part {} of {expected}", s.no + 1, s.count, i + 2),
                });
            }
            if g.truncated {
                return Err(Error::Part { path, detail: "its tensor index is cut off".into() });
            }
        }

        let model = Model { preflight, paths, rest, identity };
        if let Some(s) = model.preflight.split {
            let total = (0..model.part_count()).map(|i| model.part(i).tensors.len() as u64).sum::<u64>();
            if total != s.tensors_count {
                return Err(Error::TensorCount { declared: s.tensors_count, found: total });
            }
        }
        let mut names: Vec<&str> = model.tensors().map(|t| t.info.name.as_str()).collect();
        names.sort_unstable();
        if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
            return Err(Error::DuplicateTensor(w[0].to_owned()));
        }
        Ok(model)
    }

    /// Purpose: This function returns the model's architecture, such as `qwen3moe`.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the `general.architecture` string.
    pub fn architecture(&self) -> &str {
        &self.preflight.architecture
    }

    /// Purpose: This function returns the first part's GGUF header, which holds the
    /// model's metadata.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the header.
    pub fn meta(&self) -> &Gguf {
        &self.preflight.gguf
    }

    /// Purpose: This function returns how many files the model is stored in.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the part count.
    pub fn part_count(&self) -> usize {
        1 + self.rest.len()
    }

    /// Purpose: This function returns one part file's GGUF header.
    ///
    /// Inputs:
    /// - `i`: the part's index, from 0.
    ///
    /// Outputs: the header; panics when `i` is not below [`Model::part_count`].
    pub fn part(&self, i: usize) -> &Gguf {
        if i == 0 { &self.preflight.gguf } else { &self.rest[i - 1] }
    }

    /// Purpose: This function lists every tensor of the model across all its part files.
    ///
    /// Inputs: none.
    ///
    /// Outputs: each tensor with its part and file offset.
    pub fn tensors(&self) -> impl Iterator<Item = TensorRef<'_>> {
        (0..self.part_count()).flat_map(move |i| {
            let g = self.part(i);
            // A complete header always has a data offset; preflight and `assemble` both
            // refuse a truncated one.
            let data = g.data_offset.unwrap_or(0);
            g.tensors.iter().map(move |t| TensorRef {
                part: u16::try_from(i).unwrap_or(u16::MAX),
                info: t,
                file_offset: data + t.offset,
            })
        })
    }

    /// Purpose: This function finds one tensor of the model by name.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    ///
    /// Outputs: the tensor, or `None`.
    pub fn tensor(&self, name: &str) -> Option<TensorRef<'_>> {
        self.tensors().find(|t| t.info.name == name)
    }

    /// Purpose: This function returns each part file's header hash and expected size.
    ///
    /// Inputs: none.
    ///
    /// Outputs: one [`PartIdentity`] per part, first part first.
    pub fn identity(&self) -> &[PartIdentity] {
        &self.identity
    }
}

/// Purpose: This function reads at most [`preflight::HEADER_BUDGET`] bytes from the start
/// of a file.
///
/// Inputs:
/// - `path`: the file.
///
/// Outputs: the bytes, or the I/O error's text.
fn read_head(path: &Path) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(preflight::HEADER_BUDGET).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    Ok(bytes)
}

/// Purpose: This function names every part file of a split model by llama.cpp's naming
/// rule.
///
/// Inputs:
/// - `path`: the path of any part.
/// - `count`: how many parts the model has.
///
/// Outputs: every part's path, or `None` when the name does not follow the rule.
pub fn split_paths(path: &Path, count: u16) -> Option<Vec<PathBuf>> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // "-00001-of-00003" is 15 bytes.
    let cut = stem.len().checked_sub(15)?;
    if !stem.is_char_boundary(cut) {
        return None;
    }
    let (prefix, tail) = stem.split_at(cut);
    let ok = tail.starts_with('-')
        && tail[1..6].bytes().all(|b| b.is_ascii_digit())
        && &tail[6..10] == "-of-"
        && tail[10..].parse::<u16>().ok() == Some(count);
    if !ok {
        return None;
    }
    let dir = path.parent().unwrap_or(Path::new(""));
    Some((1..=count).map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf"))).collect())
}

/// Purpose: This enum lists the errors opening a model can return.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Part 1 failed preflight.
    Preflight(preflight::Error),
    /// No files were given.
    NoParts,
    /// The first file says it is split, but its name does not follow llama.cpp's
    /// `-00001-of-00003.gguf` rule, so the other parts cannot be found.
    SplitName(String),
    /// The wrong number of part files.
    PartCount { expected: usize, found: usize },
    /// A part after the first is unreadable or does not belong.
    Part { path: String, detail: String },
    /// The parts' tensors do not add up to `split.tensors.count`.
    TensorCount { declared: u64, found: u64 },
    /// A tensor name appears twice.
    DuplicateTensor(String),
}

impl From<preflight::Error> for Error {
    /// Purpose: This function converts a preflight error into a model-opening error.
    ///
    /// Inputs:
    /// - `e`: the preflight error.
    ///
    /// Outputs: an [`Error::Preflight`].
    fn from(e: preflight::Error) -> Self {
        Error::Preflight(e)
    }
}

impl fmt::Display for Error {
    /// Purpose: This function formats model-opening errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Preflight(e) => write!(f, "{e}"),
            Error::NoParts => write!(f, "no model files were given"),
            Error::SplitName(p) => write!(
                f,
                "{p} is part of a split model, but its name does not follow the \
                 <name>-00001-of-00003.gguf rule, so the other parts cannot be found"
            ),
            Error::PartCount { expected, found } => {
                write!(f, "the model is split into {expected} files, but {found} were given")
            }
            Error::Part { path, detail } => write!(f, "split part {path}: {detail}"),
            Error::TensorCount { declared, found } => {
                write!(f, "the parts hold {found} tensors, but split.tensors.count says {declared}")
            }
            Error::DuplicateTensor(n) => write!(f, "tensor {n} appears in more than one part"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::preflight::tests::header_with;

    /// Purpose: This function appends a tensor index to a test GGUF header.
    ///
    /// Inputs:
    /// - `b`: the header bytes.
    /// - `tensors`: each tensor as `(name, shape, ggml_type, offset)`.
    ///
    /// Outputs: the header with its tensor index.
    fn with_tensors(mut b: Vec<u8>, tensors: &[(&str, &[u64], u32, u64)]) -> Vec<u8> {
        b[8..16].copy_from_slice(&(tensors.len() as u64).to_le_bytes());
        for (name, dims, kind, offset) in tensors {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                b.extend_from_slice(&d.to_le_bytes());
            }
            b.extend_from_slice(&kind.to_le_bytes());
            b.extend_from_slice(&offset.to_le_bytes());
        }
        b
    }

    /// Purpose: This function builds the `split.*` metadata entries for a test part file.
    ///
    /// Inputs:
    /// - `no`: the part's number, from 0.
    /// - `count`: how many parts there are.
    /// - `total`: the tensor count across all parts.
    ///
    /// Outputs: the entries, as `(key, type tag, encoded value)`.
    fn split_kv(no: u16, count: u16, total: i32) -> Vec<(&'static str, u32, Vec<u8>)> {
        vec![
            ("split.no", 2, no.to_le_bytes().to_vec()),
            ("split.count", 2, count.to_le_bytes().to_vec()),
            ("split.tensors.count", 5, total.to_le_bytes().to_vec()),
        ]
    }

    /// Purpose: This function builds the GGUF header of a later part of a test split
    /// model.
    ///
    /// Inputs:
    /// - `no`: the part's number, from 0.
    /// - `count`: how many parts there are.
    /// - `total`: the tensor count across all parts.
    /// - `tensors`: this part's tensors as `(name, shape, ggml_type, offset)`.
    ///
    /// Outputs: the header bytes.
    fn later_part(no: u16, count: u16, total: i32, tensors: &[(&str, &[u64], u32, u64)]) -> Vec<u8> {
        let kvs = split_kv(no, count, total);
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, tag, payload) in &kvs {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(payload);
        }
        with_tensors(b, tensors)
    }

    #[test]
    fn a_one_file_model_has_one_part() {
        let b = with_tensors(header_with("llama", None, &[]), &[("token_embd.weight", &[64, 8], 0, 0)]);
        let m = Model::from_headers(&[("m.gguf".into(), b)]).unwrap();
        assert_eq!(m.part_count(), 1);
        let t = m.tensor("token_embd.weight").unwrap();
        assert_eq!(t.part, 0);
        assert_eq!(t.file_offset, m.meta().data_offset.unwrap());
    }

    #[test]
    fn split_parts_are_one_model_with_per_part_offsets() {
        let p0 = with_tensors(header_with("llama", None, &split_kv(0, 2, 3)), &[("a", &[32], 0, 0), ("b", &[32], 0, 128)]);
        let p1 = later_part(1, 2, 3, &[("c", &[32], 0, 0)]);
        let m = Model::from_headers(&[("m-00001-of-00002.gguf".into(), p0), ("m-00002-of-00002.gguf".into(), p1)]).unwrap();
        assert_eq!(m.part_count(), 2);
        assert_eq!(m.tensors().count(), 3);
        let c = m.tensor("c").unwrap();
        assert_eq!(c.part, 1);
        assert_eq!(c.file_offset, m.part(1).data_offset.unwrap(), "offsets are into the part's own file");
    }

    #[test]
    fn a_missing_part_is_refused() {
        let p0 = with_tensors(header_with("llama", None, &split_kv(0, 2, 2)), &[("a", &[32], 0, 0)]);
        let err = Model::from_headers(&[("m-00001-of-00002.gguf".into(), p0)]).unwrap_err();
        assert_eq!(err, Error::PartCount { expected: 2, found: 1 });
    }

    #[test]
    fn parts_in_the_wrong_order_are_refused() {
        let p0 = with_tensors(header_with("llama", None, &split_kv(0, 3, 3)), &[("a", &[32], 0, 0)]);
        let p2 = later_part(2, 3, 3, &[("c", &[32], 0, 0)]);
        let p1 = later_part(1, 3, 3, &[("b", &[32], 0, 0)]);
        let err = Model::from_headers(&[("0".into(), p0), ("2".into(), p2), ("1".into(), p1)]).unwrap_err();
        assert!(matches!(err, Error::Part { .. }), "{err:?}");
    }

    #[test]
    fn tensor_counts_must_add_up() {
        let p0 = with_tensors(header_with("llama", None, &split_kv(0, 2, 5)), &[("a", &[32], 0, 0)]);
        let p1 = later_part(1, 2, 5, &[("b", &[32], 0, 0)]);
        let err = Model::from_headers(&[("0".into(), p0), ("1".into(), p1)]).unwrap_err();
        assert_eq!(err, Error::TensorCount { declared: 5, found: 2 });
    }

    #[test]
    fn a_tensor_in_two_parts_is_refused() {
        let p0 = with_tensors(header_with("llama", None, &split_kv(0, 2, 2)), &[("a", &[32], 0, 0)]);
        let p1 = later_part(1, 2, 2, &[("a", &[32], 0, 0)]);
        let err = Model::from_headers(&[("0".into(), p0), ("1".into(), p1)]).unwrap_err();
        assert_eq!(err, Error::DuplicateTensor("a".into()));
    }

    #[test]
    fn identity_changes_with_the_header_and_predicts_the_file_size() {
        let a = with_tensors(header_with("llama", None, &[]), &[("t", &[64], 0, 0)]);
        let b = with_tensors(header_with("llama", None, &[]), &[("t", &[64], 1, 0)]);
        let ma = Model::from_headers(&[("a".into(), a)]).unwrap();
        let mb = Model::from_headers(&[("b".into(), b)]).unwrap();
        assert_ne!(ma.identity()[0].header_sha256, mb.identity()[0].header_sha256);
        assert_eq!(ma.identity()[0].header_sha256.len(), 64);
        // 64 F32 elements is 256 bytes, already a multiple of the 32-byte alignment.
        assert_eq!(ma.identity()[0].file_bytes, ma.meta().data_offset.unwrap() + 256);
    }

    #[test]
    fn split_paths_follow_llama_cpps_rule() {
        let got = split_paths(Path::new("/m/Qwen3-235B-BF16-00002-of-00003.gguf"), 3).unwrap();
        assert_eq!(
            got,
            vec![
                PathBuf::from("/m/Qwen3-235B-BF16-00001-of-00003.gguf"),
                PathBuf::from("/m/Qwen3-235B-BF16-00002-of-00003.gguf"),
                PathBuf::from("/m/Qwen3-235B-BF16-00003-of-00003.gguf"),
            ]
        );
        assert_eq!(split_paths(Path::new("/m/model.gguf"), 3), None);
        assert_eq!(split_paths(Path::new("/m/x-00001-of-00004.gguf"), 3), None, "count must match the header");
    }
}
