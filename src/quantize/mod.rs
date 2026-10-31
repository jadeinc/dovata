//! Quantizing a GGUF with llama.cpp's own `llama_model_quantize`.
//!
//! # Why llama.cpp does the work
//!
//! The quant formats are ggml's, and the Jade compiler's kernels read ggml's block
//! layouts. A second quantizer would have to match llama.cpp bit for bit, including the
//! per-tensor type choices that make a "Q4_K_M" file a mix of Q4_K, Q6_K and F32
//! tensors. Calling the real one through the bindgen'd sys crate is the only way to get
//! files that are the same as what `llama-quantize` writes.
//!
//! # Never implicit
//!
//! Nothing in dovata quantizes on its own. `hub::fetch` downloads a file as it is, and
//! planning reads what it is given. A caller who wants a smaller model calls
//! [`quantize`] and names the input, the output and the type. Quantizing a 30B model
//! reads and writes tens of gigabytes, so it should never happen as a side effect.
//!
//! # Pure core, native edge
//!
//! This file is pure and always built: the [`Ftype`] table, the spec and report types,
//! the checks that run before llama.cpp is called, and the [`default_output`] naming
//! rule. `exec.rs`, behind the `quantize` cargo feature, holds the one call into
//! llama.cpp. With the feature off, [`quantize`] still exists and returns
//! [`QuantError::NotBuilt`], so a caller compiled without it fails with a message that
//! names the feature rather than failing to link.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::load::model::split_paths;
use crate::load::Model;

#[cfg(feature = "quantize")]
mod exec;
#[cfg(test)]
pub(crate) mod fixture;

#[cfg(feature = "quantize")]
pub use exec::quantize;

/// Quantize `input` into `output` as `spec` says. Without the `quantize` feature this
/// build has no quantizer, so every call returns [`QuantError::NotBuilt`].
#[cfg(not(feature = "quantize"))]
pub fn quantize(input: &Path, output: &Path, spec: &QuantSpec) -> Result<QuantReport, QuantError> {
    let _ = (input, output, spec);
    Err(QuantError::NotBuilt)
}

/// A file type: what `general.file_type` holds and what `llama-quantize` takes, such as
/// `Q4_K_M`.
///
/// Each variant is spelled the way `llama-quantize` spells it and its discriminant is
/// its `llama_ftype` value in the vendored `include/llama.h`, so `Ftype::Q4_K_M as u32`
/// is 15. The `ftype_table_matches_llama_h` test parses that header and checks every
/// entry, so the table cannot drift from the pinned llama.cpp.
///
/// An ftype names a whole file. Its tensors keep their own `ggml_type`s, and a Q4_K_M
/// file still holds F32 norms and some Q6_K weights; llama.cpp picks the mix.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Ftype {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 7,
    Q5_0 = 8,
    Q5_1 = 9,
    Q2_K = 10,
    Q3_K_S = 11,
    Q3_K_M = 12,
    Q3_K_L = 13,
    Q4_K_S = 14,
    Q4_K_M = 15,
    Q5_K_S = 16,
    Q5_K_M = 17,
    Q6_K = 18,
    IQ2_XXS = 19,
    IQ2_XS = 20,
    Q2_K_S = 21,
    IQ3_XS = 22,
    IQ3_XXS = 23,
    IQ1_S = 24,
    IQ4_NL = 25,
    IQ3_S = 26,
    IQ3_M = 27,
    IQ2_S = 28,
    IQ2_M = 29,
    IQ4_XS = 30,
    IQ1_M = 31,
    BF16 = 32,
    TQ1_0 = 36,
    TQ2_0 = 37,
    MXFP4_MOE = 38,
    NVFP4 = 39,
    Q1_0 = 40,
}

impl Ftype {
    /// Every ftype, from the least precise to the most. [`Ftype::rank`] is the position
    /// in this list.
    ///
    /// The order is bits per weight. Where two types are close, it follows the sizes the
    /// files of one model come out at: `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` has
    /// Q4_0 at 17.38 GB, Q4_K_S at 17.46, Q4_K_M at 18.56 and Q4_1 at 19.19. F16 and
    /// BF16 have the same bits; BF16 ranks higher because most weights are trained in
    /// BF16, so a BF16 file holds them exactly and an F16 one may not.
    pub const ALL: [Ftype; 35] = [
        Ftype::Q1_0,
        Ftype::IQ1_S,
        Ftype::TQ1_0,
        Ftype::IQ1_M,
        Ftype::IQ2_XXS,
        Ftype::TQ2_0,
        Ftype::IQ2_XS,
        Ftype::IQ2_S,
        Ftype::IQ2_M,
        Ftype::Q2_K_S,
        Ftype::Q2_K,
        Ftype::IQ3_XXS,
        Ftype::IQ3_XS,
        Ftype::Q3_K_S,
        Ftype::IQ3_S,
        Ftype::IQ3_M,
        Ftype::Q3_K_M,
        Ftype::Q3_K_L,
        Ftype::IQ4_XS,
        Ftype::MXFP4_MOE,
        Ftype::IQ4_NL,
        Ftype::NVFP4,
        Ftype::Q4_0,
        Ftype::Q4_K_S,
        Ftype::Q4_K_M,
        Ftype::Q4_1,
        Ftype::Q5_0,
        Ftype::Q5_K_S,
        Ftype::Q5_K_M,
        Ftype::Q5_1,
        Ftype::Q6_K,
        Ftype::Q8_0,
        Ftype::F16,
        Ftype::BF16,
        Ftype::F32,
    ];

    /// The name `llama-quantize` takes, such as `Q4_K_M`. NVFP4 is the one type
    /// `llama-quantize` does not offer; its name is the one llama.cpp prints for it.
    pub fn name(self) -> &'static str {
        match self {
            Ftype::F32 => "F32",
            Ftype::F16 => "F16",
            Ftype::Q4_0 => "Q4_0",
            Ftype::Q4_1 => "Q4_1",
            Ftype::Q8_0 => "Q8_0",
            Ftype::Q5_0 => "Q5_0",
            Ftype::Q5_1 => "Q5_1",
            Ftype::Q2_K => "Q2_K",
            Ftype::Q3_K_S => "Q3_K_S",
            Ftype::Q3_K_M => "Q3_K_M",
            Ftype::Q3_K_L => "Q3_K_L",
            Ftype::Q4_K_S => "Q4_K_S",
            Ftype::Q4_K_M => "Q4_K_M",
            Ftype::Q5_K_S => "Q5_K_S",
            Ftype::Q5_K_M => "Q5_K_M",
            Ftype::Q6_K => "Q6_K",
            Ftype::IQ2_XXS => "IQ2_XXS",
            Ftype::IQ2_XS => "IQ2_XS",
            Ftype::Q2_K_S => "Q2_K_S",
            Ftype::IQ3_XS => "IQ3_XS",
            Ftype::IQ3_XXS => "IQ3_XXS",
            Ftype::IQ1_S => "IQ1_S",
            Ftype::IQ4_NL => "IQ4_NL",
            Ftype::IQ3_S => "IQ3_S",
            Ftype::IQ3_M => "IQ3_M",
            Ftype::IQ2_S => "IQ2_S",
            Ftype::IQ2_M => "IQ2_M",
            Ftype::IQ4_XS => "IQ4_XS",
            Ftype::IQ1_M => "IQ1_M",
            Ftype::BF16 => "BF16",
            Ftype::TQ1_0 => "TQ1_0",
            Ftype::TQ2_0 => "TQ2_0",
            Ftype::MXFP4_MOE => "MXFP4_MOE",
            Ftype::NVFP4 => "NVFP4",
            Ftype::Q1_0 => "Q1_0",
        }
    }

    /// The `llama_ftype` value, as written to `general.file_type`.
    pub fn value(self) -> u32 {
        self as u32
    }

    /// Where the type sits in [`Ftype::ALL`]: higher is more precise.
    pub fn rank(self) -> usize {
        Ftype::ALL.iter().position(|f| *f == self).unwrap_or(0)
    }

    /// False for the three full-precision types, F32, F16 and BF16.
    pub fn is_quantized(self) -> bool {
        !matches!(self, Ftype::F32 | Ftype::F16 | Ftype::BF16)
    }

    /// Whether `llama_model_quantize` can write this type. Every type but NVFP4: the
    /// pinned llama.cpp loads NVFP4 files, but `llama_ftype_get_default_type` has no
    /// case for it, so asking for one fails.
    pub fn is_target(self) -> bool {
        self != Ftype::NVFP4
    }

    /// Whether quantizing to this type needs an importance matrix. llama.cpp refuses to
    /// make these without one, because the result is unusable. [`QuantSpec`] has no
    /// way to pass one yet, so [`quantize`] refuses them up front.
    ///
    /// The list is the types whose default tensor type is one that
    /// `tensor_requires_imatrix` in `src/llama-quant.cpp` names, plus Q2_K_S, which
    /// that function names by ftype. A test derives the same list from that file.
    pub fn needs_imatrix(self) -> bool {
        matches!(
            self,
            Ftype::IQ1_S
                | Ftype::IQ1_M
                | Ftype::IQ2_XXS
                | Ftype::IQ2_XS
                | Ftype::IQ2_S
                | Ftype::IQ2_M
                | Ftype::IQ3_XXS
                | Ftype::Q2_K_S
        )
    }

    /// Read a type name the way `llama-quantize` does, ignoring case. Its three
    /// aliases are taken too: Q3_K, Q4_K and Q5_K mean the `_M` mix of each.
    pub fn parse(s: &str) -> Result<Ftype, UnknownFtype> {
        let up = s.trim().to_ascii_uppercase();
        let alias = match up.as_str() {
            "Q3_K" => Some(Ftype::Q3_K_M),
            "Q4_K" => Some(Ftype::Q4_K_M),
            "Q5_K" => Some(Ftype::Q5_K_M),
            _ => None,
        };
        alias
            .or_else(|| Ftype::ALL.iter().copied().find(|f| f.name() == up))
            .ok_or_else(|| UnknownFtype(s.to_owned()))
    }

    /// The type a `general.file_type` value names. `None` for a value llama.cpp no
    /// longer writes, such as the removed Q4_0_4_4, and for one newer than this table.
    pub fn from_file_type(value: u32) -> Option<Ftype> {
        Ftype::ALL.iter().copied().find(|f| f.value() == value)
    }
}

impl fmt::Display for Ftype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Ftype {
    type Err = UnknownFtype;
    fn from_str(s: &str) -> Result<Ftype, UnknownFtype> {
        Ftype::parse(s)
    }
}

/// A type name [`Ftype::parse`] does not know. Its message lists the ones it does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFtype(pub String);

impl fmt::Display for UnknownFtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = Ftype::ALL.iter().rev().map(|t| t.name()).collect();
        write!(f, "unknown quantization type {:?}; the types are {}", self.0, names.join(", "))
    }
}

impl std::error::Error for UnknownFtype {}

/// What to quantize to, and how.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantSpec {
    /// The type to write.
    pub ftype: Ftype,
    /// Allow an input that is already quantized. Quantizing twice loses more than
    /// quantizing once from full precision, so it is off unless asked for, and the
    /// report says when it happened.
    pub allow_requantize: bool,
    /// Threads for llama.cpp to use. 0 means llama.cpp's default, one per hardware
    /// thread.
    pub nthread: u32,
    /// For a split input, write the output as the same number of parts. Ignored for a
    /// one-file input, which always gives one file.
    pub keep_split: bool,
}

impl QuantSpec {
    /// Quantize to `ftype` from full precision, with llama.cpp's thread default.
    ///
    /// `keep_split` starts on: the output keeps the input's shape, so a split model
    /// stays split. dovata never merges a split model, and a quantized copy is no
    /// reason to start.
    pub fn new(ftype: Ftype) -> QuantSpec {
        QuantSpec { ftype, allow_requantize: false, nthread: 0, keep_split: true }
    }
}

/// What a quantization did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantReport {
    /// The input's first file.
    pub input: PathBuf,
    /// The output's first file, the one to open. A split output's other parts are found
    /// next to it by the usual `-00001-of-00003.gguf` rule.
    pub output: PathBuf,
    /// The input's `general.file_type`, when it has one dovata knows.
    pub from: Option<Ftype>,
    /// The type written, confirmed from the output's header.
    pub to: Ftype,
    /// The input was already quantized, so this quantized it a second time.
    pub requantized: bool,
    /// The input's size on disk, every part.
    pub bytes_in: u64,
    /// The output's size on disk, every part.
    pub bytes_out: u64,
}

/// Why a quantization did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// This build of dovata has no quantizer: the `quantize` feature is off.
    NotBuilt,
    /// The input is already quantized and [`QuantSpec::allow_requantize`] is off.
    Requantize { input: PathBuf, from: Option<Ftype>, to: Ftype },
    /// The input already is the requested type.
    ///
    /// This is an error, not a quiet success. A no-op would either copy the whole file
    /// to `output`, which is the 60 GB accident this check exists to stop, or leave
    /// `output` unwritten while reporting success, and a caller that later deletes its
    /// output would delete the input instead. The error carries the input's path, so a
    /// caller that wants the file as it is can use it directly.
    AlreadyAt { input: PathBuf, ftype: Ftype },
    /// The arguments cannot work: the input does not open as a model, the output would
    /// overwrite it or land in the Hugging Face cache, or the type cannot be written.
    BadInput(String),
    /// llama.cpp failed, or wrote a file that does not say what was asked for.
    Llama(String),
}

impl fmt::Display for QuantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::NotBuilt => write!(
                f,
                "this build of dovata cannot quantize: it was compiled without the `quantize` cargo feature. \
                 Enable it to link llama.cpp's quantizer"
            ),
            QuantError::Requantize { input, from, to } => {
                let from = from.map_or_else(|| "quantized tensors".to_owned(), |t| t.name().to_owned());
                write!(
                    f,
                    "{} is already quantized ({from}); quantizing it again to {to} loses more quality than \
                     quantizing once from BF16 or F16. Start from the unquantized file, or set allow_requantize \
                     to do it anyway",
                    input.display()
                )
            }
            QuantError::AlreadyAt { input, ftype } => {
                write!(f, "{} is already {ftype}; there is nothing to quantize, use the file as it is", input.display())
            }
            QuantError::BadInput(detail) => f.write_str(detail),
            QuantError::Llama(detail) => write!(f, "llama.cpp could not quantize the model: {detail}"),
        }
    }
}

impl std::error::Error for QuantError {}

/// Where to write a quantized copy of `input_first_part` when the caller has no
/// preference.
///
/// The file is named `<stem>.<FTYPE>.gguf`, where the stem is the input's name less
/// `.gguf` and less any `-00001-of-00003` part suffix. The source type stays in the
/// stem, so `X-BF16.gguf` becomes `X-BF16.Q4_K_M.gguf` and two copies made from
/// different sources never collide. A split output, from
/// [`QuantSpec::keep_split`], adds its own part suffix to this name.
///
/// The directory is the input's own, with one exception: an input inside a Hugging
/// Face hub cache. That cache keeps each file once in `blobs/` and lists each revision
/// in `snapshots/<commit>/` as symlinks into it, and hf-hub owns both. A file written
/// there would look like part of the upstream repo and could be overwritten or deleted
/// by the next download. So for an input at
/// `<cache>/models--org--repo/snapshots/<commit>/<dirs>/<file>`, the output goes to
/// `<cache>/models--org--repo/dovata/<commit>/<dirs>/<stem>.<FTYPE>.gguf`: beside
/// `snapshots/` rather than inside it. It stays with the repo it came from, and
/// deleting the repo from the cache deletes it too. An input path into `blobs/` gets the
/// same treatment.
///
/// This works on the path as given and never touches the disk. [`quantize`] checks the
/// resolved path as well, so a symlinked directory cannot get around the rule.
pub fn default_output(input_first_part: &Path, ftype: Ftype) -> PathBuf {
    let name = input_first_part.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let file = format!("{}.{}.gguf", model_stem(&name), ftype.name());
    match hub_cache_place(input_first_part) {
        Some((repo_dir, rel)) => repo_dir.join("dovata").join(rel).join(file),
        None => input_first_part.with_file_name(file),
    }
}

/// A file name less `.gguf`, in any case, and less a `-00001-of-00003` part suffix.
pub(crate) fn model_stem(name: &str) -> &str {
    let stem = strip_suffix_ignore_case(name, ".gguf").unwrap_or(name);
    split_suffix(stem).map_or(stem, |(prefix, _, _)| prefix)
}

/// `(prefix, part, count)` for a stem ending in llama.cpp's `-%05d-of-%05d`.
pub(crate) fn split_suffix(stem: &str) -> Option<(&str, u16, u16)> {
    // "-00001-of-00003" is 15 bytes.
    let cut = stem.len().checked_sub(15)?;
    if !stem.is_char_boundary(cut) {
        return None;
    }
    let (prefix, tail) = stem.split_at(cut);
    let t = tail.as_bytes();
    let digits = |r: std::ops::Range<usize>| t[r].iter().all(u8::is_ascii_digit);
    if prefix.is_empty() || t[0] != b'-' || !digits(1..6) || &tail[6..10] != "-of-" || !digits(10..15) {
        return None;
    }
    Some((prefix, tail[1..6].parse().ok()?, tail[10..15].parse().ok()?))
}

pub(crate) fn strip_suffix_ignore_case<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let cut = s.len().checked_sub(suffix.len())?;
    (s.is_char_boundary(cut) && s[cut..].eq_ignore_ascii_case(suffix)).then(|| &s[..cut])
}

/// For a path inside a Hugging Face hub cache repo, the repo's directory and the
/// directories between its `snapshots/` (or `blobs/`) and the file. hf-hub names repo
/// directories `models--org--repo`, `datasets--…` or `spaces--…`.
fn hub_cache_place(path: &Path) -> Option<(PathBuf, PathBuf)> {
    let parts: Vec<_> = path.components().collect();
    let parent_len = parts.len().checked_sub(1)?;
    for i in 0..parent_len.saturating_sub(1) {
        let repo = parts[i].as_os_str().to_string_lossy();
        let area = parts[i + 1].as_os_str();
        let is_repo = ["models--", "datasets--", "spaces--"].iter().any(|p| repo.starts_with(p));
        if is_repo && (area == "snapshots" || area == "blobs") {
            let repo_dir: PathBuf = parts[..=i].iter().collect();
            let rel: PathBuf = parts[i + 2..parent_len].iter().collect();
            return Some((repo_dir, rel));
        }
    }
    None
}

/// Whether `path` lies inside a hub cache's `snapshots/` or `blobs/`, where dovata must
/// never write.
pub(crate) fn in_hub_cache(path: &Path) -> bool {
    hub_cache_place(path).is_some()
}

/// Whether a `ggml_type` is a quantized one, the same split as ggml's
/// `ggml_is_quantized`. The plain types are F32, F16, BF16, F64 and the integers;
/// every other type, known or not, counts as quantized, so an unknown one is never
/// silently requantized.
pub(crate) fn is_quantized_type(ggml_type: u32) -> bool {
    !matches!(ggml_type, 0 | 1 | 24..=28 | 30)
}

/// A quantization that passed every check that can be made without llama.cpp.
#[cfg_attr(not(feature = "quantize"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
struct Job {
    from: Option<Ftype>,
    requantized: bool,
    keep_split: bool,
    /// What llama.cpp is given as the output: the file, or for a split output the
    /// prefix it names the parts from.
    target: PathBuf,
    /// The files llama.cpp will write, first part first.
    outputs: Vec<PathBuf>,
}

/// Decide whether a quantization can run, before llama.cpp spends minutes loading the
/// model. Pure, so the refusals are tested on any machine.
#[cfg_attr(not(feature = "quantize"), allow(dead_code))]
fn prepare(model: &Model, output: &Path, spec: &QuantSpec) -> Result<Job, QuantError> {
    let input = model.paths[0].clone();
    let to = spec.ftype;
    if !to.is_target() {
        return Err(QuantError::BadInput(format!(
            "llama.cpp cannot write {to}: it reads {to} files, but its quantizer has no {to} output"
        )));
    }
    if to.needs_imatrix() {
        return Err(QuantError::BadInput(format!(
            "{to} needs an importance matrix, and dovata does not pass one to llama.cpp yet; \
             without one llama.cpp refuses, since the result would be unusable. Pick a type that \
             does not need one, such as Q3_K_S, Q4_K_M or IQ4_XS"
        )));
    }

    let from = model.meta().file_type().and_then(Ftype::from_file_type);
    if from == Some(to) {
        return Err(QuantError::AlreadyAt { input, ftype: to });
    }
    let quantized = model.tensors().any(|t| is_quantized_type(t.info.kind));
    if quantized && !spec.allow_requantize {
        return Err(QuantError::Requantize { input, from, to });
    }

    let parts = model.part_count();
    let keep_split = spec.keep_split && parts > 1;
    let (target, outputs) = if keep_split {
        let target = output_prefix(output);
        let count = u16::try_from(parts).map_err(|_| QuantError::BadInput(format!("{parts} parts is too many")))?;
        let first = PathBuf::from(format!("{}-00001-of-{count:05}.gguf", target.display()));
        let outputs = split_paths(&first, count)
            .ok_or_else(|| QuantError::BadInput(format!("cannot name the parts of {}", output.display())))?;
        (target, outputs)
    } else {
        (output.to_path_buf(), vec![output.to_path_buf()])
    };

    if let Some(clash) = outputs.iter().find(|o| model.paths.contains(o)) {
        return Err(QuantError::BadInput(format!(
            "the output {} is one of the input's own files; quantizing in place would destroy the input",
            clash.display()
        )));
    }
    if let Some(cached) = outputs.iter().find(|o| in_hub_cache(o)) {
        return Err(QuantError::BadInput(cache_refusal(cached)));
    }

    Ok(Job { from, requantized: quantized, keep_split, target, outputs })
}

/// The message for an output inside the hub cache.
fn cache_refusal(path: &Path) -> String {
    format!(
        "the output {} is inside a Hugging Face cache's snapshots or blobs, which hf-hub owns and may \
         overwrite. Write it elsewhere; quantize::default_output picks a place beside the snapshot",
        path.display()
    )
}

/// What llama.cpp takes as a split output's name: the output less `.gguf`, or less a
/// whole `-00001-of-00003.gguf`, the same as `llama-quantize --keep-split` does.
#[cfg_attr(not(feature = "quantize"), allow(dead_code))]
fn output_prefix(output: &Path) -> PathBuf {
    let name = output.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    output.with_file_name(model_stem(&name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::write::Writer;
    use crate::gguf::Value;

    /// The vendored llama.cpp tree. The drift tests read its sources, so they need the
    /// submodule checked out; without it they fail and say so, since a drift check that
    /// passes when it cannot see its reference proves nothing.
    pub(crate) fn llama_cpp_file(rel: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/llama/bindings-sys/llama.cpp").join(rel);
        std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("cannot read {}: {e}. Check out the llama.cpp submodule to run this test", path.display())
        })
    }

    /// The body of a C block from the line holding `start` to the next line that
    /// starts with `end`.
    fn block<'a>(src: &'a str, start: &str, end: &str) -> Vec<&'a str> {
        let mut lines = src.lines().skip_while(|l| !l.contains(start));
        let first = lines.next().unwrap_or_else(|| panic!("{start} not found"));
        std::iter::once(first).chain(lines.take_while(|l| !l.trim_start().starts_with(end))).collect()
    }

    /// `NAME = VALUE` entries of the `llama_ftype` enum, commented-out ones skipped.
    fn llama_h_ftypes() -> Vec<(String, u32)> {
        let src = llama_cpp_file("include/llama.h");
        block(&src, "enum llama_ftype {", "};")
            .into_iter()
            .map(|l| l.split("//").next().unwrap_or("").trim())
            .filter_map(|l| {
                let (name, value) = l.trim_end_matches(',').split_once('=')?;
                Some((name.trim().to_owned(), value.trim().parse().ok()?))
            })
            .filter(|(name, _)| name.starts_with("LLAMA_FTYPE_"))
            .collect()
    }

    fn c_name(f: Ftype) -> String {
        match f {
            Ftype::F32 => "LLAMA_FTYPE_ALL_F32".to_owned(),
            other => format!("LLAMA_FTYPE_MOSTLY_{}", other.name()),
        }
    }

    #[test]
    fn ftype_table_matches_llama_h() {
        let entries = llama_h_ftypes();
        assert!(entries.len() > 30, "parsed only {} entries; the parser is broken", entries.len());
        for f in Ftype::ALL {
            let found = entries.iter().find(|(n, _)| *n == c_name(f));
            assert_eq!(found.map(|e| e.1), Some(f.value()), "{} in llama.h", c_name(f));
        }
        // And the other way: a type upstream adds must be added here too, or a file of
        // that type would have no Ftype.
        for (name, value) in &entries {
            if name == "LLAMA_FTYPE_GUESSED" {
                continue;
            }
            assert!(Ftype::from_file_type(*value).is_some(), "{name} = {value} is missing from Ftype");
        }
        let mut values: Vec<u32> = Ftype::ALL.iter().map(|f| f.value()).collect();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), Ftype::ALL.len(), "Ftype::ALL lists a type twice");
    }

    #[test]
    fn ftype_names_match_llama_quantize() {
        // QUANT_OPTIONS in tools/quantize/quantize.cpp: { "Q4_K_M", LLAMA_FTYPE_MOSTLY_Q4_K_M, ... }.
        let src = llama_cpp_file("tools/quantize/quantize.cpp");
        let options: Vec<(String, String)> = block(&src, "QUANT_OPTIONS = {", "};")
            .into_iter()
            .filter_map(|l| {
                let l = l.trim().strip_prefix("{ \"")?;
                let (name, rest) = l.split_once('"')?;
                let c = rest.trim_start_matches([',', ' ']).split(',').next()?.trim();
                Some((name.to_owned(), c.to_owned()))
            })
            .collect();
        assert!(options.len() > 30, "parsed only {} options", options.len());
        for f in Ftype::ALL {
            let offered = options.iter().any(|(n, c)| n == f.name() && *c == c_name(f));
            assert_eq!(offered, f.is_target(), "{} in llama-quantize's QUANT_OPTIONS", f.name());
        }
        // Every name llama-quantize takes parses, aliases included; COPY is not a type.
        for (name, c) in options.iter().filter(|(n, _)| n != "COPY") {
            let f = Ftype::parse(name).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(c_name(f), *c, "{name}");
        }
    }

    #[test]
    fn targets_and_imatrix_needs_match_llama_quant() {
        let src = llama_cpp_file("src/llama-quant.cpp");
        // llama_ftype_get_default_type: runs of `case LLAMA_FTYPE_X:` ending in `return GGML_TYPE_Y;`.
        let mut default_type = Vec::new();
        let mut pending = Vec::new();
        for line in block(&src, "ggml_type llama_ftype_get_default_type", "default:") {
            for word in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
                if word.starts_with("LLAMA_FTYPE_") {
                    pending.push(word.to_owned());
                } else if let Some(t) = word.strip_prefix("GGML_TYPE_") {
                    default_type.extend(pending.drain(..).map(|f| (f, t.to_owned())));
                }
            }
        }
        // tensor_requires_imatrix: the GGML types listed before its first `return true;`.
        let needs: Vec<String> = block(&src, "static bool tensor_requires_imatrix", "return true;")
            .into_iter()
            .filter_map(|l| l.trim().strip_prefix("case GGML_TYPE_")?.strip_suffix(':').map(str::to_owned))
            .collect();
        assert!(needs.len() >= 5, "parsed {needs:?}");
        assert!(src.contains("return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;"), "the Q2_K_S rule moved");

        for f in Ftype::ALL {
            let t = default_type.iter().find(|(c, _)| *c == c_name(f)).map(|(_, t)| t.as_str());
            assert_eq!(t.is_some(), f.is_target(), "{f}: llama_ftype_get_default_type");
            let imatrix = f == Ftype::Q2_K_S || t.is_some_and(|t| needs.iter().any(|n| n == t));
            assert_eq!(imatrix, f.needs_imatrix(), "{f}: needs_imatrix");
        }
    }

    #[test]
    fn ftype_names_parse() {
        assert_eq!(Ftype::parse("Q4_K_M"), Ok(Ftype::Q4_K_M));
        assert_eq!(Ftype::parse("q4_k_m"), Ok(Ftype::Q4_K_M), "case is ignored");
        assert_eq!(Ftype::parse(" bf16 "), Ok(Ftype::BF16));
        assert_eq!(Ftype::parse("mxfp4_moe"), Ok(Ftype::MXFP4_MOE));
        assert_eq!(Ftype::parse("Q4_K"), Ok(Ftype::Q4_K_M), "llama-quantize's alias");
        assert_eq!("IQ4_XS".parse::<Ftype>(), Ok(Ftype::IQ4_XS));
        for f in Ftype::ALL {
            assert_eq!(Ftype::parse(f.name()), Ok(f), "every name reads back");
            assert_eq!(f.to_string(), f.name());
        }
        let err = Ftype::parse("Q4_K_XL").unwrap_err();
        assert!(err.to_string().contains("Q4_K_M"), "the message lists the real names: {err}");
        assert!(Ftype::parse("COPY").is_err(), "llama-quantize's COPY is a mode, not a type");
    }

    #[test]
    fn file_type_maps_to_ftype() {
        assert_eq!(Ftype::from_file_type(0), Some(Ftype::F32));
        assert_eq!(Ftype::from_file_type(7), Some(Ftype::Q8_0));
        assert_eq!(Ftype::from_file_type(15), Some(Ftype::Q4_K_M));
        assert_eq!(Ftype::from_file_type(32), Some(Ftype::BF16));
        assert_eq!(Ftype::from_file_type(38), Some(Ftype::MXFP4_MOE));
        assert_eq!(Ftype::from_file_type(4), None, "Q4_1_SOME_F16 is gone");
        assert_eq!(Ftype::from_file_type(33), None, "Q4_0_4_4 was removed from gguf files");
        assert_eq!(Ftype::from_file_type(1024), None, "LLAMA_FTYPE_GUESSED is never a file's type");
        for f in Ftype::ALL {
            assert_eq!(Ftype::from_file_type(f.value()), Some(f));
        }
    }

    #[test]
    fn rank_orders_by_precision() {
        let r = Ftype::rank;
        assert!(r(Ftype::F32) > r(Ftype::BF16) && r(Ftype::BF16) > r(Ftype::F16));
        assert!(r(Ftype::F16) > r(Ftype::Q8_0) && r(Ftype::Q8_0) > r(Ftype::Q6_K));
        assert!(r(Ftype::Q4_1) > r(Ftype::Q4_K_M) && r(Ftype::Q4_K_M) > r(Ftype::Q4_K_S) && r(Ftype::Q4_K_S) > r(Ftype::Q4_0));
        assert!(r(Ftype::Q2_K) > r(Ftype::IQ1_S));
        assert_eq!(r(Ftype::Q1_0), 0);
        let full: Vec<Ftype> = Ftype::ALL.iter().copied().filter(|f| !f.is_quantized()).collect();
        assert_eq!(full, [Ftype::F16, Ftype::BF16, Ftype::F32]);
        assert_eq!(&Ftype::ALL[Ftype::ALL.len() - 3..], full.as_slice(), "full precision sorts last");
    }

    #[test]
    fn default_output_sits_beside_the_input() {
        assert_eq!(
            default_output(Path::new("/m/Qwen3-Coder-30B-A3B-Instruct-BF16.gguf"), Ftype::Q4_K_M),
            Path::new("/m/Qwen3-Coder-30B-A3B-Instruct-BF16.Q4_K_M.gguf")
        );
        assert_eq!(
            default_output(Path::new("/m/X-BF16-00001-of-00002.gguf"), Ftype::Q8_0),
            Path::new("/m/X-BF16.Q8_0.gguf"),
            "the part suffix is dropped; a split output adds its own"
        );
        assert_eq!(default_output(Path::new("model.GGUF"), Ftype::Q8_0), Path::new("model.Q8_0.gguf"));
    }

    #[test]
    fn default_output_never_writes_into_a_hub_snapshot() {
        let snap = "/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/snapshots/b17cb02dd882d5b6ab62fc777ad2995f19668350";
        let out = default_output(Path::new(&format!("{snap}/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf")), Ftype::Q4_K_M);
        assert_eq!(
            out,
            Path::new("/home/u/.cache/huggingface/hub/models--unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF/dovata/b17cb02dd882d5b6ab62fc777ad2995f19668350/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16.Q4_K_M.gguf")
        );
        assert!(!in_hub_cache(&out));
        assert!(in_hub_cache(Path::new(&format!("{snap}/x.gguf"))));

        let blob = Path::new("/c/models--a--b/blobs/9f2c1e");
        assert_eq!(default_output(blob, Ftype::Q8_0), Path::new("/c/models--a--b/dovata/9f2c1e.Q8_0.gguf"));
        // A directory merely called `snapshots` elsewhere is not the cache.
        assert_eq!(default_output(Path::new("/w/snapshots/m.gguf"), Ftype::Q8_0), Path::new("/w/snapshots/m.Q8_0.gguf"));
    }

    #[test]
    fn split_suffixes_are_read() {
        assert_eq!(split_suffix("X-BF16-00002-of-00010"), Some(("X-BF16", 2, 10)));
        assert_eq!(split_suffix("X-BF16"), None);
        assert_eq!(split_suffix("-00001-of-00002"), None, "a part needs a name before it");
        assert_eq!(split_suffix("X-0001-of-00002"), None);
        assert_eq!(model_stem("a-00001-of-00002.gguf"), "a");
    }

    // ── prepare: the refusals, against headers only ─────────────────────────────

    /// A llama header, no tensor data, declaring `file_type` and holding one tensor of
    /// `ggml_type`. Enough for every check `prepare` makes.
    fn model_with(file_type: Option<Ftype>, ggml_type: u32) -> Model {
        let mut w = Writer::new();
        fixture::llama_metadata(&mut w);
        if let Some(f) = file_type {
            w.kv("general.file_type", Value::U32(f.value()));
        }
        let (blocks, bytes) = crate::gguf::ggml_type_layout(ggml_type).unwrap();
        w.tensor("token_embd.weight", &[256, 4], ggml_type, vec![0; (256 / blocks * bytes * 4) as usize]);
        Model::from_headers(&[("/m/in.gguf".into(), w.to_bytes().unwrap())]).unwrap()
    }

    #[test]
    fn already_at_the_type_is_an_error_not_a_copy() {
        let m = model_with(Some(Ftype::F16), 1);
        let err = prepare(&m, Path::new("/m/out.gguf"), &QuantSpec::new(Ftype::F16)).unwrap_err();
        assert_eq!(err, QuantError::AlreadyAt { input: "/m/in.gguf".into(), ftype: Ftype::F16 });
        assert!(err.to_string().contains("use the file as it is"), "{err}");
    }

    #[test]
    fn requantizing_needs_the_flag() {
        let m = model_with(Some(Ftype::Q8_0), 8);
        let spec = QuantSpec::new(Ftype::Q4_K_M);
        let err = prepare(&m, Path::new("/m/out.gguf"), &spec).unwrap_err();
        assert_eq!(err, QuantError::Requantize { input: "/m/in.gguf".into(), from: Some(Ftype::Q8_0), to: Ftype::Q4_K_M });
        assert!(err.to_string().contains("allow_requantize"), "{err}");

        let job = prepare(&m, Path::new("/m/out.gguf"), &QuantSpec { allow_requantize: true, ..spec }).unwrap();
        assert!(job.requantized, "the report must say it happened");

        // A file with no general.file_type is judged by its tensors.
        let unlabeled = model_with(None, 12);
        assert!(matches!(prepare(&unlabeled, Path::new("/m/o.gguf"), &spec), Err(QuantError::Requantize { from: None, .. })));
        let full = model_with(None, 30);
        assert!(!prepare(&full, Path::new("/m/o.gguf"), &spec).unwrap().requantized);
    }

    #[test]
    fn types_llama_cpp_cannot_make_are_refused_up_front() {
        let m = model_with(Some(Ftype::BF16), 30);
        for (to, says) in [(Ftype::NVFP4, "cannot write NVFP4"), (Ftype::IQ2_XXS, "importance matrix"), (Ftype::Q2_K_S, "importance matrix")] {
            let err = prepare(&m, Path::new("/m/o.gguf"), &QuantSpec::new(to)).unwrap_err();
            assert!(matches!(&err, QuantError::BadInput(d) if d.contains(says)), "{to}: {err}");
        }
    }

    #[test]
    fn the_output_may_not_be_the_input_or_in_the_cache() {
        let m = model_with(Some(Ftype::BF16), 30);
        let spec = QuantSpec::new(Ftype::Q8_0);
        assert!(matches!(prepare(&m, Path::new("/m/in.gguf"), &spec), Err(QuantError::BadInput(d)) if d.contains("in place")));
        let cached = Path::new("/c/models--a--b/snapshots/0123/out.gguf");
        assert!(matches!(prepare(&m, cached, &spec), Err(QuantError::BadInput(d)) if d.contains("default_output")));
        let ok = prepare(&m, Path::new("/m/out.gguf"), &spec).unwrap();
        assert_eq!(ok.outputs, [PathBuf::from("/m/out.gguf")]);
        assert!(!ok.keep_split, "a one-file input gives one file");
    }

    #[test]
    fn a_split_output_is_named_by_llama_cpps_rule() {
        let (p0, p1) = fixture::split_headers();
        let m = Model::from_headers(&[("/m/in-00001-of-00002.gguf".into(), p0), ("/m/in-00002-of-00002.gguf".into(), p1)]).unwrap();
        let job = prepare(&m, Path::new("/o/in.Q8_0.gguf"), &QuantSpec::new(Ftype::Q8_0)).unwrap();
        assert!(job.keep_split);
        assert_eq!(job.target, Path::new("/o/in.Q8_0"), "llama.cpp gets the prefix, as with --keep-split");
        assert_eq!(job.outputs, [PathBuf::from("/o/in.Q8_0-00001-of-00002.gguf"), PathBuf::from("/o/in.Q8_0-00002-of-00002.gguf")]);

        let merged = prepare(&m, Path::new("/o/in.Q8_0.gguf"), &QuantSpec { keep_split: false, ..QuantSpec::new(Ftype::Q8_0) }).unwrap();
        assert_eq!(merged.outputs, [PathBuf::from("/o/in.Q8_0.gguf")]);
    }

    #[test]
    #[cfg(not(feature = "quantize"))]
    fn quant_request_is_refused_without_feature() {
        let err = quantize(Path::new("in.gguf"), Path::new("out.gguf"), &QuantSpec::new(Ftype::Q4_K_M)).unwrap_err();
        assert_eq!(err, QuantError::NotBuilt);
        assert!(err.to_string().contains("`quantize` cargo feature"), "{err}");
    }
}
