//! Author: Joe Ricks
//! File: src/quantize/mod.rs
//! Description: Lists the GGUF file types and checks a quantization before llama.cpp runs.

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

/// Purpose: This function returns `NotBuilt` in a build without the `quantize` feature.
///
/// Inputs:
/// - `input`: the model file to quantize; unused.
/// - `output`: the file to write; unused.
/// - `spec`: what to make; unused.
///
/// Outputs: always [`QuantError::NotBuilt`].
#[cfg(not(feature = "quantize"))]
pub fn quantize(input: &Path, output: &Path, spec: &QuantSpec) -> Result<QuantReport, QuantError> {
    let _ = (input, output, spec);
    Err(QuantError::NotBuilt)
}

/// Purpose: This enum lists the GGUF file types llama.cpp's quantizer knows, such as
/// `Q4_K_M`.
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
    Q2_0 = 41,
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
    pub const ALL: [Ftype; 36] = [
        Ftype::Q1_0,
        Ftype::IQ1_S,
        Ftype::TQ1_0,
        Ftype::IQ1_M,
        Ftype::IQ2_XXS,
        Ftype::TQ2_0,
        Ftype::Q2_0,
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

    /// Purpose: This function returns the file type's name, as `llama-quantize` spells
    /// it.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a name such as `Q4_K_M`.
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
            Ftype::Q2_0 => "Q2_0",
        }
    }

    /// Purpose: This function returns the file type's `llama_ftype` number.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the number, such as 15 for `Q4_K_M`.
    pub fn value(self) -> u32 {
        self as u32
    }

    /// Purpose: This function returns the file type's position in [`Ftype::ALL`], from
    /// least to most precise.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the position; higher is more precise.
    pub fn rank(self) -> usize {
        Ftype::ALL.iter().position(|f| *f == self).unwrap_or(0)
    }

    /// Purpose: This function checks whether the file type is quantized.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `false` for F32, F16 and BF16, and `true` for the rest.
    pub fn is_quantized(self) -> bool {
        !matches!(self, Ftype::F32 | Ftype::F16 | Ftype::BF16)
    }

    /// Purpose: This function checks whether llama.cpp's quantizer can write this file
    /// type.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for every type but NVFP4.
    pub fn is_target(self) -> bool {
        self != Ftype::NVFP4
    }

    /// Purpose: This function checks whether quantizing to this file type needs an
    /// importance matrix.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS and
    /// Q2_K_S.
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

    /// Purpose: This function reads a file type from its name, ignoring case.
    ///
    /// Inputs:
    /// - `s`: the name, such as `q4_k_m` or the alias `Q4_K`.
    ///
    /// Outputs: the type, or an [`UnknownFtype`] error.
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

    /// Purpose: This function finds the file type a `general.file_type` number names.
    ///
    /// Inputs:
    /// - `value`: the `llama_ftype` number from a GGUF file.
    ///
    /// Outputs: the type, or `None` for an unknown number.
    pub fn from_file_type(value: u32) -> Option<Ftype> {
        Ftype::ALL.iter().copied().find(|f| f.value() == value)
    }
}

impl fmt::Display for Ftype {
    /// Purpose: This function formats a file type as its `llama-quantize` name.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Ftype {
    type Err = UnknownFtype;
    /// Purpose: This function reads a file type from its name, as [`Ftype::parse`] does.
    ///
    /// Inputs:
    /// - `s`: the name.
    ///
    /// Outputs: the type, or an [`UnknownFtype`] error.
    fn from_str(s: &str) -> Result<Ftype, UnknownFtype> {
        Ftype::parse(s)
    }
}

/// Purpose: This struct holds a file type name that [`Ftype::parse`] does not know.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownFtype(pub String);

impl fmt::Display for UnknownFtype {
    /// Purpose: This function formats the unknown file type error, listing the valid
    /// names.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = Ftype::ALL.iter().rev().map(|t| t.name()).collect();
        write!(f, "unknown quantization type {:?}; the types are {}", self.0, names.join(", "))
    }
}

impl std::error::Error for UnknownFtype {}

/// Purpose: This struct holds what file type to quantize to, and the options.
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
    /// Purpose: This function builds a quantization request for one file type with the
    /// default options.
    ///
    /// Inputs:
    /// - `ftype`: the file type to make.
    ///
    /// Outputs: the [`QuantSpec`].
    pub fn new(ftype: Ftype) -> QuantSpec {
        QuantSpec { ftype, allow_requantize: false, nthread: 0, keep_split: true }
    }
}

/// Purpose: This struct holds what a quantization wrote.
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

/// Purpose: This enum lists the errors quantizing can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantError {
    /// This build of dovata has no quantizer: the `quantize` feature is off.
    NotBuilt,
    /// llama.cpp's `llama-quantize` program is not on `PATH`.
    NotInstalled,
    /// The input is already quantized and [`QuantSpec::allow_requantize`] is off.
    Requantize { input: PathBuf, from: Option<Ftype>, to: Ftype },
    /// The input already is the requested type.
    ///
    /// This is an error, not a quiet success. A no-op would either copy the whole file
    /// to `output`, which is the 60 GB accident this check exists to stop, or leave
    /// `output` unwritten while reporting success, and a program that later deletes its
    /// output would delete the input instead. The error carries the input's path, so a
    /// program that wants the file as it is can use it directly.
    AlreadyAt { input: PathBuf, ftype: Ftype },
    /// The arguments cannot work: the input does not open as a model, the output would
    /// overwrite it or land in the Hugging Face cache, or the type cannot be written.
    BadInput(String),
    /// llama.cpp failed, or wrote a file that does not say what was asked for.
    Llama(String),
}

impl fmt::Display for QuantError {
    /// Purpose: This function formats quantization errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantError::NotBuilt => write!(
                f,
                "this build of dovata cannot quantize: it was compiled without the `quantize` cargo feature. \
                 Enable it to run llama.cpp's llama-quantize"
            ),
            QuantError::NotInstalled => write!(
                f,
                "dovata quantizes by running llama.cpp's llama-quantize program, and it is not on PATH. \
                 Install llama.cpp: `brew install llama.cpp` on macOS, or a release build from \
                 https://github.com/ggml-org/llama.cpp/releases on Linux"
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

/// Purpose: This function picks the path for a quantized copy of a model.
///
/// Inputs:
/// - `input_first_part`: the input model's file, or its first part.
/// - `ftype`: the file type the copy will be.
///
/// Outputs: the output path, `<stem>.<FTYPE>.gguf`.
pub fn default_output(input_first_part: &Path, ftype: Ftype) -> PathBuf {
    let name = input_first_part.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let file = format!("{}.{}.gguf", model_stem(&name), ftype.name());
    match hub_cache_place(input_first_part) {
        Some((repo_dir, rel)) => repo_dir.join("dovata").join(rel).join(file),
        None => input_first_part.with_file_name(file),
    }
}

/// Purpose: This function strips `.gguf` and any split-part suffix from a model file
/// name.
///
/// Inputs:
/// - `name`: the file name.
///
/// Outputs: the base name.
pub(crate) fn model_stem(name: &str) -> &str {
    let stem = strip_suffix_ignore_case(name, ".gguf").unwrap_or(name);
    split_suffix(stem).map_or(stem, |(prefix, _, _)| prefix)
}

/// Purpose: This function reads llama.cpp's `-%05d-of-%05d` part suffix off the end of a
/// name.
///
/// Inputs:
/// - `stem`: a file name without `.gguf`.
///
/// Outputs: the name before the suffix, the part number and the part count, or `None`.
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

/// Purpose: This function removes a suffix from a string, ignoring ASCII case.
///
/// Inputs:
/// - `s`: the string.
/// - `suffix`: the suffix.
///
/// Outputs: the string without the suffix, or `None` when it does not end with it.
pub(crate) fn strip_suffix_ignore_case<'a>(s: &'a str, suffix: &str) -> Option<&'a str> {
    let cut = s.len().checked_sub(suffix.len())?;
    (s.is_char_boundary(cut) && s[cut..].eq_ignore_ascii_case(suffix)).then(|| &s[..cut])
}

/// Purpose: This function finds where a path sits inside a Hugging Face hub cache repo.
///
/// Inputs:
/// - `path`: the file's path.
///
/// Outputs: the repo directory and the directories between its `snapshots/` or `blobs/`
/// and the file, or `None`.
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

/// Purpose: This function checks whether a path is inside a Hugging Face hub cache's
/// `snapshots/` or `blobs/`.
///
/// Inputs:
/// - `path`: the file's path.
///
/// Outputs: `true` when it is.
pub(crate) fn in_hub_cache(path: &Path) -> bool {
    hub_cache_place(path).is_some()
}

/// Purpose: This function checks whether a ggml tensor type is quantized.
///
/// Inputs:
/// - `ggml_type`: the tensor's `ggml_type` number.
///
/// Outputs: `false` for F32, F16, BF16, F64 and the integer types, and `true` for every
/// other number.
pub(crate) fn is_quantized_type(ggml_type: u32) -> bool {
    !matches!(ggml_type, 0 | 1 | 24..=28 | 30)
}

/// Purpose: This struct holds a quantization that passed every check made before
/// llama.cpp runs.
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

/// Purpose: This function checks a quantization request before llama.cpp runs and names
/// the output files.
///
/// Inputs:
/// - `model`: the opened input model.
/// - `output`: the output path.
/// - `spec`: what to make.
///
/// Outputs: the [`Job`] to run, or a [`QuantError`] saying why it cannot run.
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

/// Purpose: This function writes the error message for an output path inside the Hugging
/// Face hub cache.
///
/// Inputs:
/// - `path`: the refused output path.
///
/// Outputs: the message.
fn cache_refusal(path: &Path) -> String {
    format!(
        "the output {} is inside a Hugging Face cache's snapshots or blobs, which hf-hub owns and may \
         overwrite. Write it elsewhere; quantize::default_output picks a place beside the snapshot",
        path.display()
    )
}

/// Purpose: This function works out the name llama.cpp takes for a split output.
///
/// Inputs:
/// - `output`: the output path.
///
/// Outputs: the path without `.gguf` or its part suffix.
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
        let mut values: Vec<u32> = Ftype::ALL.iter().map(|f| f.value()).collect();
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), Ftype::ALL.len(), "Ftype::ALL lists a type twice");
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

    /// Purpose: This function builds a small test llama model with one tensor of a chosen
    /// type.
    ///
    /// Inputs:
    /// - `file_type`: the `general.file_type` to declare, if any.
    /// - `ggml_type`: the type of the model's one tensor.
    ///
    /// Outputs: the [`Model`].
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
