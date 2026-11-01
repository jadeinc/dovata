//! Author: Joe Ricks
//! File: src/hub/mod.rs
//! Description: Finds one model's GGUF files, on disk or on the Hugging Face Hub.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::load::model::{self, split_paths};
use crate::load::Model;
use crate::quantize::{model_stem, split_suffix, strip_suffix_ignore_case, Ftype};

#[cfg(feature = "hub")]
mod fetch;

/// Purpose: This enum holds where a model comes from: a path or a Hub repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSource {
    /// A GGUF file on disk, or the first or any part of a split one.
    Path(PathBuf),
    /// A model repo on the Hugging Face Hub.
    Hub {
        /// The Hub to download from, such as `https://huggingface.co`, taken from the
        /// Hub link given to [`ModelSource::parse`]. There is no default.
        endpoint: String,
        /// `org/repo`.
        repo: String,
        /// A branch, tag or commit. `None` is the repo's `main`.
        revision: Option<String>,
        /// A file in the repo, or a directory holding one model's parts. `None` lets
        /// [`fetch`] choose by [`Want`].
        file: Option<String>,
    },
}

impl ModelSource {
    /// Purpose: This function reads a model source from a file path or a Hugging Face
    /// link.
    ///
    /// Inputs:
    /// - `s`: the text, such as `/models/x.gguf` or a huggingface.co link.
    ///
    /// Outputs: the source, or an error when the text names no model.
    pub fn parse(s: &str) -> Result<ModelSource, SourceError> {
        let s = s.trim();
        let fail = |reason: &str| SourceError { input: s.to_owned(), reason: reason.to_owned() };
        if s.is_empty() {
            return Err(fail("it is empty"));
        }
        if let Some(rest) = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")) {
            return parse_url(rest).map_err(|r| fail(&r));
        }
        if s.starts_with('/') || s.starts_with('.') {
            return Ok(ModelSource::Path(PathBuf::from(s)));
        }
        if let Some(rest) = s.strip_prefix('~') {
            if rest.is_empty() || rest.starts_with('/') {
                let home = std::env::var_os("HOME").ok_or_else(|| fail("~ needs HOME to be set"))?;
                return Ok(ModelSource::Path(PathBuf::from(home).join(rest.trim_start_matches('/'))));
            }
        }
        if Path::new(s).exists() {
            return Ok(ModelSource::Path(PathBuf::from(s)));
        }
        if is_repo_id(s) && strip_suffix_ignore_case(s, ".gguf").is_none() {
            return Err(fail(&format!(
                "it is not a file on disk, and a Hub model needs its full link, as in https://huggingface.co/{s}"
            )));
        }
        Ok(ModelSource::Path(PathBuf::from(s)))
    }
}

impl std::str::FromStr for ModelSource {
    type Err = SourceError;
    /// Purpose: This function reads a model source from text, the same as
    /// `ModelSource::parse`.
    ///
    /// Inputs:
    /// - `s`: the text.
    ///
    /// Outputs: the source, or an error.
    fn from_str(s: &str) -> Result<ModelSource, SourceError> {
        ModelSource::parse(s)
    }
}

/// Shown in messages: a path as it is, and a Hub source as the link to it, which
/// [`ModelSource::parse`] reads back.
impl fmt::Display for ModelSource {
    /// Purpose: This function formats a model source as a file path or a Hugging Face
    /// link.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelSource::Path(p) => write!(f, "{}", p.display()),
            ModelSource::Hub { endpoint, repo, revision: None, file: None } => write!(f, "{endpoint}/{repo}"),
            ModelSource::Hub { endpoint, repo, revision, file } => {
                let rev = revision.as_deref().unwrap_or("main").replace('/', "%2F");
                match file {
                    None => write!(f, "{endpoint}/{repo}/tree/{rev}"),
                    Some(file) if strip_suffix_ignore_case(file, ".gguf").is_some() => {
                        write!(f, "{endpoint}/{repo}/blob/{rev}/{file}")
                    }
                    Some(dir) => write!(f, "{endpoint}/{repo}/tree/{rev}/{dir}"),
                }
            }
        }
    }
}

/// Purpose: This function reads a Hugging Face link, given without its `https://`.
///
/// Inputs:
/// - `rest`: the link after its scheme.
///
/// Outputs: the Hub source, or the reason the link names no model.
fn parse_url(rest: &str) -> Result<ModelSource, String> {
    let rest = rest.split(['?', '#']).next().unwrap_or(rest);
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    // `hf.co` and `www.huggingface.co` redirect to `huggingface.co`, and hf-hub keeps the
    // token only on a redirect to the same host (`RedirectAuthHeaders::SameHost` in
    // hf-hub's `api/sync.rs`), so a gated repo would fail through the redirect. The
    // endpoint is the host they redirect to. It is always `https`, even for an `http`
    // link, so a token never goes out in plain text.
    let endpoint = match host {
        "huggingface.co" | "www.huggingface.co" | "hf.co" => "https://huggingface.co".to_owned(),
        _ => return Err(format!("{host} is not the Hugging Face Hub")),
    };
    let segs: Vec<String> = path.split('/').filter(|s| !s.is_empty()).map(percent_decode).collect();
    if matches!(segs.first().map(String::as_str), Some("datasets" | "spaces")) {
        return Err("only model repos hold models".to_owned());
    }
    if segs.len() < 2 {
        return Err("a Hub link needs an org and a repo, as in https://huggingface.co/org/repo".to_owned());
    }
    let repo = format!("{}/{}", segs[0], segs[1]);
    if !is_repo_id(&repo) {
        return Err(format!("{repo} is not a repo id"));
    }
    let hub = |revision: Option<&String>, file: Option<String>| ModelSource::Hub {
        endpoint: endpoint.clone(),
        repo: repo.clone(),
        revision: revision.cloned(),
        file,
    };
    match segs.get(2).map(String::as_str) {
        None => Ok(hub(None, None)),
        Some("tree") => match &segs[3..] {
            [] => Err("a tree link needs a revision".to_owned()),
            [rev] => Ok(hub(Some(rev), None)),
            [rev, dir @ ..] => Ok(hub(Some(rev), Some(dir.join("/")))),
        },
        Some("blob" | "resolve") => match &segs[3..] {
            [rev, path @ ..] if !path.is_empty() => {
                let file = path.join("/");
                if strip_suffix_ignore_case(&file, ".gguf").is_none() {
                    return Err(format!("{file} is not a GGUF file"));
                }
                Ok(hub(Some(rev), Some(file)))
            }
            _ => Err("a file link needs a revision and a file".to_owned()),
        },
        Some(other) => Err(format!("a /{other}/ link does not name a model; use the repo, a file or a tree link")),
    }
}

/// Purpose: This function tells whether text is a Hub repo id such as `org/repo`.
///
/// Inputs:
/// - `s`: the text.
///
/// Outputs: `true` when it is.
fn is_repo_id(s: &str) -> bool {
    let ok = |n: &str| !n.is_empty() && n != "." && !n.contains("..") && n.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b));
    matches!(s.split_once('/'), Some((org, name)) if ok(org) && ok(name))
}

/// Purpose: This function decodes the `%XX` escapes in one part of a link.
///
/// Inputs:
/// - `s`: the text.
///
/// Outputs: the decoded text.
fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    let hex = |c: u8| (c as char).to_digit(16);
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2])) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Purpose: This struct holds the error for text that does not name a model source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceError {
    pub input: String,
    pub reason: String,
}

impl fmt::Display for SourceError {
    /// Purpose: This function formats model source errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot read {:?} as a model: {}", self.input, self.reason)
    }
}

impl std::error::Error for SourceError {}

/// Purpose: This enum holds which model of a Hub repo to take when no file is named.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Want {
    /// The unquantized model: BF16, then F16, then F32, else the most precise one.
    #[default]
    Unquantized,
    /// Exactly this type.
    Ftype(Ftype),
}

impl fmt::Display for Want {
    /// Purpose: This function formats which model of a repo is wanted.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Want::Unquantized => f.write_str("the unquantized model"),
            Want::Ftype(t) => write!(f, "{t}"),
        }
    }
}

/// Purpose: This struct holds one model in a Hub repo: its label, its type and its files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    /// What the repo calls it, from the file name: `Q4_K_M`, `BF16`, `UD-Q4_K_XL`.
    pub label: String,
    /// The llama ftype the name gives, or `None` for a label that is not one.
    pub ftype: Option<Ftype>,
    /// Its files' paths in the repo, part 1 first.
    pub files: Vec<String>,
}

impl fmt::Display for Level {
    /// Purpose: This function formats a model's label and part count.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.label)?;
        if self.files.len() > 1 {
            write!(f, " ({} parts)", self.files.len())?;
        }
        Ok(())
    }
}

/// Purpose: This type alias names the map that groups a repo's files into models.
type Groups<'a> = BTreeMap<(&'a str, &'a str, u16), Vec<(u16, String)>>;

/// Purpose: This function groups a Hub repo's file list into models.
///
/// Inputs:
/// - `files`: the file paths in the Hub repo.
///
/// Outputs: one entry per model, most precise first.
pub fn levels(files: &[String]) -> Vec<Level> {
    let mut groups: Groups<'_> = BTreeMap::new();
    for path in files.iter().filter(|p| is_model_gguf(p)) {
        let (dir, name) = path.rsplit_once('/').unwrap_or(("", path.as_str()));
        let stem = strip_suffix_ignore_case(name, ".gguf").unwrap_or(name);
        let (key, part) = match split_suffix(stem) {
            Some((prefix, part, count)) => ((dir, prefix, count), part),
            None => ((dir, stem, 0), 0),
        };
        groups.entry(key).or_default().push((part, path.clone()));
    }
    let mut out: Vec<Level> = groups
        .into_values()
        .map(|mut parts| {
            parts.sort();
            let files: Vec<String> = parts.into_iter().map(|(_, p)| p).collect();
            let (label, ftype) = classify(&files[0]);
            Level { label, ftype, files }
        })
        .collect();
    out.sort_by(|a, b| {
        let rank = |l: &Level| l.ftype.map(Ftype::rank);
        rank(b).cmp(&rank(a)).then_with(|| a.label.cmp(&b.label)).then_with(|| a.files.cmp(&b.files))
    });
    out
}

/// Purpose: This function tells whether a file is a GGUF that holds a model.
///
/// Inputs:
/// - `path`: the file's path in the repo.
///
/// Outputs: `true` for a model, and `false` for a vision projector or an importance
/// matrix.
fn is_model_gguf(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path).to_ascii_lowercase();
    name.ends_with(".gguf") && !name.starts_with("mmproj") && !name.contains("imatrix")
}

/// Names a file may use for a type, beyond the [`Ftype`] names themselves: the three
/// aliases `llama-quantize` takes, and three spellings of llama.cpp's own types.
const NAME_ALIASES: [(&str, Ftype); 6] = [
    ("Q3_K", Ftype::Q3_K_M),
    ("Q4_K", Ftype::Q4_K_M),
    ("Q5_K", Ftype::Q5_K_M),
    ("FP16", Ftype::F16),
    ("FP32", Ftype::F32),
    ("MXFP4", Ftype::MXFP4_MOE),
];

/// Purpose: This function finds a model file's label and type from its name or its
/// directory.
///
/// Inputs:
/// - `path`: the file's path in the repo.
///
/// Outputs: the label, and the type or `None`.
fn classify(path: &str) -> (String, Option<Ftype>) {
    let mut comps = path.rsplit('/');
    let name = comps.next().unwrap_or(path);
    let stem = model_stem(name);
    if let Some(found) = find_ftype(stem) {
        return found;
    }
    if let Some(found) = comps.find_map(find_ftype) {
        return found;
    }
    // No llama ftype: the last token is the label, with a UD- prefix kept.
    let tokens: Vec<&str> = stem.split(['-', '.']).filter(|t| !t.is_empty()).collect();
    let label = match tokens.as_slice() {
        [.., ud, last] if ud.eq_ignore_ascii_case("UD") => format!("UD-{last}"),
        [.., last] => (*last).to_owned(),
        [] => stem.to_owned(),
    };
    (label, None)
}

/// Purpose: This function finds the longest type name in a file or directory name.
///
/// Inputs:
/// - `text`: the name.
///
/// Outputs: the label and type, or `None`.
fn find_ftype(text: &str) -> Option<(String, Option<Ftype>)> {
    let up = text.to_ascii_uppercase();
    let names = Ftype::ALL.iter().map(|f| (f.name(), *f)).chain(NAME_ALIASES);
    // (length, end, start, ftype) of the best match so far: longer wins, then the one
    // later in the name.
    let mut best: Option<(usize, usize, usize, Ftype)> = None;
    for (name, ftype) in names {
        for (start, _) in up.match_indices(name) {
            let end = start + name.len();
            let starts = start == 0 || matches!(up.as_bytes()[start - 1], b'-' | b'.' | b'_');
            let ends = end == up.len() || matches!(up.as_bytes()[end], b'-' | b'.');
            if starts && ends && best.is_none_or(|(len, e, _, _)| (name.len(), end) > (len, e)) {
                best = Some((name.len(), end, start, ftype));
            }
        }
    }
    let (_, _, start, ftype) = best?;
    let before = &up[..start];
    let dynamic = before.ends_with("UD-") || before.ends_with("UD_");
    Some(if dynamic { (format!("UD-{}", ftype.name()), None) } else { (ftype.name().to_owned(), Some(ftype)) })
}

/// Purpose: This function chooses one model from a Hub repo's file list.
///
/// Inputs:
/// - `files`: the file paths in the Hub repo.
/// - `want`: which type is wanted.
///
/// Outputs: the chosen model, or an error.
pub fn select(files: &[String], want: Want) -> Result<Level, SelectError> {
    let all = levels(files);
    if all.is_empty() {
        return Err(SelectError::NoGguf);
    }
    let of = |t: Ftype| all.iter().filter(|l| l.ftype == Some(t)).cloned().collect::<Vec<_>>();
    match want {
        Want::Ftype(t) => match of(t) {
            found if found.is_empty() => Err(SelectError::NotOffered { want: t, offered: all }),
            found => one(found, want),
        },
        Want::Unquantized => {
            if let Some(found) = [Ftype::BF16, Ftype::F16, Ftype::F32].into_iter().map(of).find(|f| !f.is_empty()) {
                return one(found, want);
            }
            // `all` is sorted most precise first, so the first with an ftype is the best.
            if let Some(best) = all.iter().find_map(|l| l.ftype) {
                return one(of(best), want);
            }
            match all.as_slice() {
                [only] => Ok(only.clone()),
                _ => Err(SelectError::Unclear { offered: all }),
            }
        }
    }
}

/// Purpose: This function returns the only model in a list.
///
/// Inputs:
/// - `found`: the models that match.
/// - `want`: what was asked for.
///
/// Outputs: the model, or an error when there is more than one.
fn one(mut found: Vec<Level>, want: Want) -> Result<Level, SelectError> {
    match found.len() {
        1 => Ok(found.remove(0)),
        _ => Err(SelectError::Ambiguous { want, candidates: found }),
    }
}

/// Purpose: This function chooses the model a file or directory name points to.
///
/// Inputs:
/// - `files`: the file paths in the Hub repo.
/// - `name`: the file or directory.
///
/// Outputs: the chosen model, or an error.
pub fn select_file(files: &[String], name: &str) -> Result<Level, SelectError> {
    let name = name.trim_matches('/');
    let all = levels(files);
    if let Some(level) = all.iter().find(|l| l.files.iter().any(|f| f == name)) {
        return Ok(level.clone());
    }
    if files.iter().any(|f| f == name) {
        return Err(SelectError::NotAModel(name.to_owned()));
    }
    let prefix = format!("{name}/");
    let mut inside: Vec<Level> = all.iter().filter(|l| l.files.iter().all(|f| f.starts_with(&prefix))).cloned().collect();
    match inside.len() {
        0 => Err(SelectError::NoSuchFile { file: name.to_owned(), offered: all }),
        1 => Ok(inside.remove(0)),
        _ => Err(SelectError::Ambiguous { want: Want::Unquantized, candidates: inside }),
    }
}

/// Purpose: This enum lists the errors from choosing a model in a Hub repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectError {
    /// The repo holds no GGUF model files.
    NoGguf,
    /// The repo does not offer the type asked for.
    NotOffered { want: Ftype, offered: Vec<Level> },
    /// More than one model fits.
    Ambiguous { want: Want, candidates: Vec<Level> },
    /// Several models, none with a type dovata can rank, and no file named.
    Unclear { offered: Vec<Level> },
    /// The named file or directory is not in the repo.
    NoSuchFile { file: String, offered: Vec<Level> },
    /// The named file is a GGUF, but a projector or imatrix rather than a model.
    NotAModel(String),
}

/// Purpose: This function lists models as text for an error message.
///
/// Inputs:
/// - `levels`: the models.
///
/// Outputs: the list as text.
fn list(levels: &[Level]) -> String {
    let typed: Vec<String> = levels.iter().filter(|l| l.ftype.is_some()).map(ToString::to_string).collect();
    let named: Vec<String> = levels.iter().filter(|l| l.ftype.is_none()).map(ToString::to_string).collect();
    match (typed.is_empty(), named.is_empty()) {
        (_, true) => typed.join(", "),
        (true, false) => format!("{}, by file name only", named.join(", ")),
        (false, false) => format!("{}; and by file name only, {}", typed.join(", "), named.join(", ")),
    }
}

impl fmt::Display for SelectError {
    /// Purpose: This function formats model selection errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SelectError::NoGguf => write!(
                f,
                "it has no GGUF model files. Repos holding only safetensors are not supported yet; \
                 convert one with llama.cpp's convert_hf_to_gguf.py, or use a GGUF repo of the same model"
            ),
            SelectError::NotOffered { want, offered } => write!(
                f,
                "it does not offer {want}. It offers {}. To get {want}, fetch the unquantized model \
                 with Want::Unquantized and pass it to dovata::quantize::quantize",
                list(offered)
            ),
            SelectError::Ambiguous { want, candidates } => {
                let files: Vec<&str> = candidates.iter().map(|l| l.files[0].as_str()).collect();
                write!(f, "more than one model fits {want}: {}. Name the file to fetch", files.join(", "))
            }
            SelectError::Unclear { offered } => write!(
                f,
                "none of its models has a type in its name, so there is no telling which is unquantized. \
                 It offers {}. Name the file to fetch",
                list(offered)
            ),
            SelectError::NoSuchFile { file, offered } => {
                write!(f, "it has no file or directory {file}. It offers {}", list(offered))
            }
            SelectError::NotAModel(file) => {
                write!(f, "{file} is a vision projector or an importance matrix, not a model")
            }
        }
    }
}

impl std::error::Error for SelectError {}

/// Purpose: This struct holds a fetch's cache directory and access token.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct FetchOptions {
    /// The hf-hub cache directory. `None` is `HF_HUB_CACHE`, else `HF_HOME/hub`, else
    /// `~/.cache/huggingface/hub`, the same as `huggingface-cli`.
    pub cache_dir: Option<PathBuf>,
    /// A Hub access token, for gated and private repos. `None` reads `HF_TOKEN`, else
    /// the token file `huggingface-cli login` writes.
    pub token: Option<String>,
}

/// Debug without the token, so a logged options value never leaks it.
impl fmt::Debug for FetchOptions {
    /// Purpose: This function formats fetch options with the access token hidden.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FetchOptions")
            .field("cache_dir", &self.cache_dir)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

/// Purpose: This struct holds a fetched model's files, type, repo and commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    /// Every part, part 1 first. For a Hub model these are paths in the hf-hub cache.
    pub parts: Vec<PathBuf>,
    /// The `general.file_type` the model declares, when it has one dovata knows.
    pub ftype: Option<Ftype>,
    /// The Hub repo, for a Hub source.
    pub repo: Option<String>,
    /// The commit the files were downloaded at, for a Hub source: the resolved sha,
    /// even when the source named a branch.
    pub revision: Option<String>,
}

/// Purpose: This enum lists the errors a fetch can return.
#[derive(Debug, Clone, PartialEq)]
pub enum FetchError {
    /// This build has no Hub client: the `hub` feature is off.
    NotBuilt { source: String },
    /// The source cannot be fetched as written, such as a malformed repo id.
    BadSource(String),
    /// The repo was listed, but no model in it could be chosen.
    Select { repo: String, error: SelectError },
    /// Talking to the Hub failed.
    Hub { repo: String, detail: String },
    /// The files do not open as one model.
    Model(model::Error),
    /// A downloaded file's `general.file_type` is not the type its name says.
    Mismatch { file: String, named: Ftype, declared: u32 },
}

impl From<model::Error> for FetchError {
    /// Purpose: This function wraps a model loading error as a fetch error.
    ///
    /// Inputs:
    /// - `e`: the model loading error.
    ///
    /// Outputs: the fetch error.
    fn from(e: model::Error) -> FetchError {
        FetchError::Model(e)
    }
}

impl fmt::Display for FetchError {
    /// Purpose: This function formats fetch errors.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FetchError::NotBuilt { source } => write!(
                f,
                "cannot fetch {source}: this build of dovata has no Hugging Face client, since it was \
                 compiled without the `hub` cargo feature. Enable it, or download the GGUF and pass its path"
            ),
            FetchError::BadSource(detail) => f.write_str(detail),
            FetchError::Select { repo, error } => write!(f, "{repo}: {error}"),
            FetchError::Hub { repo, detail } => write!(f, "{repo}: {detail}"),
            FetchError::Model(e) => write!(f, "{e}"),
            FetchError::Mismatch { file, named, declared } => {
                let declared = Ftype::from_file_type(*declared).map_or_else(|| format!("type {declared}"), |t| t.to_string());
                write!(
                    f,
                    "{file} is named {named}, but its header says {declared}. Refusing a file whose name \
                     and contents disagree; name the file you want to fetch it anyway"
                )
            }
        }
    }
}

impl std::error::Error for FetchError {}

/// Purpose: This function gets one model's files on disk.
///
/// Inputs:
/// - `source`: where the model comes from.
/// - `want`: which model of a repo to take.
/// - `opts`: the cache directory and the access token.
///
/// Outputs: the model's file paths, part 1 first, or an error.
pub fn fetch(source: &ModelSource, want: Want, opts: &FetchOptions) -> Result<Fetched, FetchError> {
    match source {
        ModelSource::Path(path) => local(path),
        ModelSource::Hub { endpoint, repo, revision, file } => {
            #[cfg(feature = "hub")]
            {
                fetch::from_hub(endpoint, repo, revision.as_deref(), file.as_deref(), want, opts)
            }
            #[cfg(not(feature = "hub"))]
            {
                let _ = (endpoint, repo, revision, file, want, opts);
                Err(FetchError::NotBuilt { source: source.to_string() })
            }
        }
    }
}

/// Purpose: This function lists the models a Hub repo offers, without downloading any.
///
/// Inputs:
/// - `source`: a Hub repo, with no file named.
/// - `opts`: the cache directory and the access token.
///
/// Outputs: one [`Level`] per model, most precise first, or an error. A local path or a
/// link that names a file is one model, so it is refused as [`FetchError::BadSource`].
pub fn offered(source: &ModelSource, opts: &FetchOptions) -> Result<Vec<Level>, FetchError> {
    match source {
        ModelSource::Hub { endpoint, repo, revision, file: None } => {
            #[cfg(feature = "hub")]
            {
                fetch::offered(endpoint, repo, revision.as_deref(), opts)
            }
            #[cfg(not(feature = "hub"))]
            {
                let _ = (endpoint, repo, revision, opts);
                Err(FetchError::NotBuilt { source: source.to_string() })
            }
        }
        _ => Err(FetchError::BadSource(format!("{source} names one model; only a Hub repo offers several"))),
    }
}

/// Purpose: This function gets a local model file and the other parts of a split model.
///
/// Inputs:
/// - `path`: the file, or any part of a split model.
///
/// Outputs: the model's file paths, or an error.
fn local(path: &Path) -> Result<Fetched, FetchError> {
    let model = match later_part_count(path) {
        Some(count) => {
            let parts = split_paths(path, count).ok_or_else(|| model::Error::SplitName(path.display().to_string()))?;
            Model::from_parts(&parts)?
        }
        None => Model::open(path)?,
    };
    Ok(fetched(&model, None, None))
}

/// Purpose: This function reads the part count of a split model part after the first.
///
/// Inputs:
/// - `path`: the file.
///
/// Outputs: the part count, or `None` for any other file.
fn later_part_count(path: &Path) -> Option<u16> {
    use std::io::Read;
    let mut bytes = Vec::new();
    std::fs::File::open(path).ok()?.take(crate::load::preflight::HEADER_BUDGET).read_to_end(&mut bytes).ok()?;
    let split = crate::gguf::Gguf::parse(&bytes).ok()?.split()?;
    (split.no != 0).then_some(split.count)
}

/// Purpose: This function builds a fetch result from an opened model.
///
/// Inputs:
/// - `model`: the opened model.
/// - `repo`: the Hub repo, if any.
/// - `revision`: the commit, if any.
///
/// Outputs: the fetch result.
fn fetched(model: &Model, repo: Option<String>, revision: Option<String>) -> Fetched {
    Fetched {
        parts: model.paths.clone(),
        ftype: model.meta().file_type().and_then(Ftype::from_file_type),
        repo,
        revision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: This function reads a saved Hub file list from `testdata/`.
    ///
    /// Inputs:
    /// - `name`: the list's short name.
    ///
    /// Outputs: the file paths.
    fn repo(name: &str) -> Vec<String> {
        let text = match name {
            "coder" => include_str!("testdata/unsloth--Qwen3-Coder-30B-A3B-Instruct-GGUF.txt"),
            "235b" => include_str!("testdata/unsloth--Qwen3-235B-A22B-Instruct-2507-GGUF.txt"),
            "gemma" => include_str!("testdata/unsloth--gemma-3-4b-it-GGUF.txt"),
            "thebloke" => include_str!("testdata/TheBloke--Llama-2-7B-GGUF.txt"),
            "bartowski" => include_str!("testdata/bartowski--Qwen_Qwen3-8B-GGUF.txt"),
            "qwen25" => include_str!("testdata/Qwen--Qwen2.5-7B-Instruct-GGUF.txt"),
            "safetensors" => include_str!("testdata/Qwen--Qwen3-8B.txt"),
            other => panic!("no test list {other}"),
        };
        text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_owned).collect()
    }

    /// Purpose: This function turns string slices into a file list.
    ///
    /// Inputs:
    /// - `files`: the paths.
    ///
    /// Outputs: the paths as strings.
    fn names(files: &[&str]) -> Vec<String> {
        files.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Purpose: This function builds a huggingface.co model source for the tests.
    ///
    /// Inputs:
    /// - `repo`: the repo id.
    /// - `revision`: the revision, if any.
    /// - `file`: the file, if any.
    ///
    /// Outputs: the source.
    fn hub(repo: &str, revision: Option<&str>, file: Option<&str>) -> ModelSource {
        ModelSource::Hub {
            endpoint: "https://huggingface.co".into(),
            repo: repo.into(),
            revision: revision.map(Into::into),
            file: file.map(Into::into),
        }
    }

    const CODER: &str = "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF";

    #[test]
    fn parses_urls_blob_and_resolve_links() {
        let p = |s: &str| ModelSource::parse(s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(p(&format!("https://huggingface.co/{CODER}")), hub(CODER, None, None));
        assert_eq!(p(&format!("https://huggingface.co/{CODER}/")), hub(CODER, None, None));
        assert_eq!(p(&format!("https://hf.co/{CODER}")), hub(CODER, None, None), "an alias, at the host it redirects to");
        assert_eq!(p(&format!("http://huggingface.co/{CODER}")), hub(CODER, None, None), "always https");
        assert_eq!(
            p(&format!("https://huggingface.co/{CODER}/blob/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")),
            hub(CODER, Some("main"), Some("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"))
        );
        assert_eq!(
            p(&format!("https://huggingface.co/{CODER}/resolve/main/BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf?download=true")),
            hub(CODER, Some("main"), Some("BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf")),
            "a subdirectory path, and the download query dropped"
        );
        assert_eq!(
            p(&format!("https://huggingface.co/{CODER}/tree/b17cb02dd882d5b6ab62fc777ad2995f19668350")),
            hub(CODER, Some("b17cb02dd882d5b6ab62fc777ad2995f19668350"), None)
        );
        assert_eq!(p(&format!("https://huggingface.co/{CODER}/tree/main/BF16")), hub(CODER, Some("main"), Some("BF16")));
        assert_eq!(
            p("https://huggingface.co/org/repo/blob/refs%2Fpr%2F1/m.gguf"),
            hub("org/repo", Some("refs/pr/1"), Some("m.gguf")),
            "an escaped revision"
        );

        assert_eq!(p("/models/x.gguf"), ModelSource::Path("/models/x.gguf".into()));
        assert_eq!(p("./x.gguf"), ModelSource::Path("./x.gguf".into()));
        assert_eq!(p("../models/x"), ModelSource::Path("../models/x".into()), "a leading dot is always a path");
        assert_eq!(p("x.gguf"), ModelSource::Path("x.gguf".into()));
        assert_eq!(p("org/model.gguf"), ModelSource::Path("org/model.gguf".into()), "a .gguf name is not a repo");
        if let Some(home) = std::env::var_os("HOME") {
            assert_eq!(p("~/m.gguf"), ModelSource::Path(PathBuf::from(home).join("m.gguf")));
        }
        // `src/hub` has the shape of a repo id, but it exists here, the package root.
        assert_eq!(p("src/hub"), ModelSource::Path("src/hub".into()), "an existing path wins");

        // A bare repo id names no server, so it is refused, with the link it meant.
        let err = ModelSource::parse(CODER).unwrap_err();
        assert!(err.reason.contains(&format!("https://huggingface.co/{CODER}")), "{err}");

        for bad in [
            "https://example.com/org/repo",
            "https://huggingface.co/datasets/org/repo",
            "https://huggingface.co/onlyorg",
            &format!("https://huggingface.co/{CODER}/blob/main/README.md"),
            &format!("https://huggingface.co/{CODER}/tree"),
            &format!("https://huggingface.co/{CODER}/discussions/3"),
            "",
        ] {
            assert!(ModelSource::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn display_reads_back() {
        for s in [
            hub(CODER, None, None),
            hub(CODER, Some("main"), None),
            hub(CODER, Some("main"), Some("BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf")),
            hub(CODER, Some("main"), Some("BF16")),
            hub("org/repo", Some("refs/pr/1"), Some("m.gguf")),
        ] {
            assert_eq!(ModelSource::parse(&s.to_string()).unwrap(), s, "{s}");
        }
        assert_eq!(hub(CODER, None, Some("x.gguf")).to_string(), format!("https://huggingface.co/{CODER}/blob/main/x.gguf"));
    }

    #[test]
    fn exact_quant_file_selected() {
        let pick = |list: &str, t: Ftype| select(&repo(list), Want::Ftype(t)).unwrap_or_else(|e| panic!("{e}")).files;
        assert_eq!(pick("coder", Ftype::Q4_K_M), ["Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"]);
        assert_eq!(pick("coder", Ftype::IQ4_XS), ["Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf"]);
        assert_eq!(pick("coder", Ftype::Q2_K), ["Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf"], "not Q2_K_L");
        assert_eq!(pick("thebloke", Ftype::Q8_0), ["llama-2-7b.Q8_0.gguf"], "a dot before the type");
        assert_eq!(pick("bartowski", Ftype::Q4_K_M), ["Qwen_Qwen3-8B-Q4_K_M.gguf"], "not Q4_K_L");
        assert_eq!(pick("bartowski", Ftype::Q6_K), ["Qwen_Qwen3-8B-Q6_K.gguf"], "not Q6_K_L");
        assert_eq!(pick("bartowski", Ftype::IQ3_XXS), ["Qwen_Qwen3-8B-IQ3_XXS.gguf"], "not IQ3_XS");
        assert_eq!(
            pick("qwen25", Ftype::Q4_K_M),
            ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"],
            "lower case, and split"
        );
        assert_eq!(pick("235b", Ftype::Q4_K_M).len(), 3);
    }

    #[test]
    fn unquantized_prefers_bf16() {
        let un = |files: &[String]| select(files, Want::Unquantized).unwrap_or_else(|e| panic!("{e}"));
        let coder = un(&repo("coder"));
        assert_eq!((coder.label.as_str(), coder.ftype), ("BF16", Some(Ftype::BF16)));
        assert_eq!(
            coder.files,
            ["BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00001-of-00002.gguf", "BF16/Qwen3-Coder-30B-A3B-Instruct-BF16-00002-of-00002.gguf"]
        );
        assert_eq!(un(&repo("gemma")).files, ["gemma-3-4b-it-BF16.gguf"], "not mmproj-BF16.gguf");
        assert_eq!(un(&repo("bartowski")).files, ["Qwen_Qwen3-8B-bf16.gguf"]);
        let qwen = un(&repo("qwen25"));
        assert_eq!((qwen.ftype, qwen.files.len()), (Some(Ftype::F16), 4), "fp16 is F16");

        assert_eq!(un(&names(&["m-F32.gguf", "m-F16.gguf", "m-BF16.gguf", "m-Q8_0.gguf"])).files, ["m-BF16.gguf"]);
        assert_eq!(un(&names(&["m-F32.gguf", "m-F16.gguf", "m-Q8_0.gguf"])).files, ["m-F16.gguf"], "F16 before F32");
        assert_eq!(un(&names(&["m-F32.gguf", "m-Q8_0.gguf"])).files, ["m-F32.gguf"]);
    }

    #[test]
    fn missing_unquantized_picks_highest_precision() {
        let un = |files: &[String]| select(files, Want::Unquantized).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(un(&repo("thebloke")).files, ["llama-2-7b.Q8_0.gguf"]);
        // A dynamic quant has no rank, however large: Q5_K_M wins over UD-Q8_K_XL.
        let files = names(&["m-Q4_K_M.gguf", "m-Q5_K_M.gguf", "m-UD-Q8_K_XL.gguf", "m-Q4_1.gguf"]);
        assert_eq!(un(&files).files, ["m-Q5_K_M.gguf"]);
        // One model with no type in its name is still the only choice.
        assert_eq!(un(&names(&["README.md", "stories260K.gguf"])).files, ["stories260K.gguf"]);
        let err = select(&names(&["a.gguf", "b.gguf"]), Want::Unquantized).unwrap_err();
        assert!(matches!(err, SelectError::Unclear { .. }), "{err:?}");
    }

    #[test]
    fn split_parts_are_one_level() {
        let all = levels(&repo("235b"));
        assert_eq!(all.len(), 20, "{all:#?}");
        assert_eq!(all.iter().map(|l| l.files.len()).sum::<usize>(), 76, "every part of every model, nothing else");
        let bf16 = &all[0];
        assert_eq!(bf16.ftype, Some(Ftype::BF16), "most precise first");
        assert_eq!(bf16.files.len(), 10);
        for (i, f) in bf16.files.iter().enumerate() {
            assert_eq!(*f, format!("BF16/Qwen3-235B-A22B-Instruct-2507-BF16-{:05}-of-00010.gguf", i + 1), "parts in order");
        }
        // Listed out of order, the parts still come back in order.
        let mut shuffled = names(&["m-00003-of-00003.gguf", "m-00001-of-00003.gguf", "m-00002-of-00003.gguf"]);
        shuffled.reverse();
        assert_eq!(levels(&shuffled)[0].files, ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"]);

        // A named part brings the whole model.
        let level = select_file(&repo("235b"), "Q8_0/Qwen3-235B-A22B-Instruct-2507-Q8_0-00004-of-00006.gguf").unwrap();
        assert_eq!(level.files.len(), 6);
        assert_eq!(level.files[0], "Q8_0/Qwen3-235B-A22B-Instruct-2507-Q8_0-00001-of-00006.gguf");
    }

    #[test]
    fn subdirectory_quants_grouped() {
        let all = levels(&repo("235b"));
        for l in &all {
            let dir = l.files[0].split('/').next().unwrap();
            assert_eq!(dir, l.label, "each model is its own directory, named for its label");
            assert!(l.files.iter().all(|f| f.starts_with(&format!("{dir}/"))));
        }
        let labels: Vec<&str> = all.iter().map(|l| l.label.as_str()).collect();
        assert_eq!(
            labels,
            [
                "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_1", "Q4_K_M", "Q4_K_S", "Q4_0", "IQ4_XS", "Q3_K_M", "Q3_K_S", "Q2_K",
                "Q2_K_L", "UD-Q2_K_XL", "UD-Q3_K_XL", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL",
            ]
        );
        // A directory link takes the model inside it.
        assert_eq!(select_file(&repo("coder"), "BF16").unwrap().files.len(), 2);
        assert_eq!(select_file(&repo("coder"), "BF16/").unwrap().label, "BF16");
        // The type can come from the directory when the file name has none.
        let l = levels(&names(&["Q4_K_M/model-00001-of-00002.gguf", "Q4_K_M/model-00002-of-00002.gguf"]));
        assert_eq!((l.len(), l[0].ftype), (1, Some(Ftype::Q4_K_M)));
    }

    #[test]
    fn labels_that_are_not_ftypes_are_kept() {
        let all = levels(&repo("coder"));
        let get = |label: &str| all.iter().find(|l| l.label == label).unwrap_or_else(|| panic!("{label} in {all:#?}"));
        assert_eq!(get("UD-Q4_K_XL").ftype, None);
        assert_eq!(get("Q2_K_L").ftype, None);
        assert_eq!(get("UD-IQ1_M").ftype, None, "a dynamic quant is not llama-quantize's IQ1_M");
        assert_eq!(get("UD-TQ1_0").ftype, None);
        assert_eq!(get("Q4_K_M").ftype, Some(Ftype::Q4_K_M));
        assert_eq!(all.len(), 27, "27 models; imatrix_unsloth.gguf_file is not one");
        let bart = levels(&repo("bartowski"));
        assert!(bart.iter().any(|l| l.label == "Q4_K_L" && l.ftype.is_none()));
        assert!(bart.iter().any(|l| l.label == "Q3_K_XL" && l.ftype.is_none()));
        assert!(bart.iter().any(|l| l.label == "IQ3_M" && l.ftype == Some(Ftype::IQ3_M)));
        // Old ARM repacks are no ftype today; the end rule keeps Q4_0 from matching.
        assert_eq!(levels(&names(&["m-Q4_0_4_4.gguf"]))[0].ftype, None);
        assert_eq!(levels(&names(&["gpt-oss-20b-mxfp4.gguf"]))[0].ftype, Some(Ftype::MXFP4_MOE));
        assert_eq!(levels(&names(&["model_q8_0.gguf"]))[0].ftype, Some(Ftype::Q8_0), "an underscore before the type");
    }

    #[test]
    fn mmproj_ignored() {
        let all = levels(&repo("gemma"));
        assert!(all.iter().all(|l| !l.files[0].starts_with("mmproj")), "{all:#?}");
        assert_eq!(all.iter().filter(|l| l.ftype == Some(Ftype::BF16)).count(), 1);
        let err = select_file(&repo("gemma"), "mmproj-F16.gguf").unwrap_err();
        assert_eq!(err, SelectError::NotAModel("mmproj-F16.gguf".into()));
        assert!(levels(&names(&["imatrix_unsloth.gguf", "m-Q8_0.gguf"])).len() == 1, "imatrix GGUFs are skipped too");
    }

    #[test]
    fn not_offered_lists_levels() {
        let err = select(&repo("coder"), Want::Ftype(Ftype::IQ3_M)).unwrap_err();
        let SelectError::NotOffered { want, offered } = &err else { panic!("{err:?}") };
        assert_eq!(*want, Ftype::IQ3_M);
        assert_eq!(offered.len(), 27);
        let msg = err.to_string();
        for s in ["does not offer IQ3_M", "BF16 (2 parts), Q8_0", "UD-Q4_K_XL", "by file name only", "Want::Unquantized", "quantize::quantize"] {
            assert!(msg.contains(s), "{s:?} missing from: {msg}");
        }
    }

    #[test]
    fn no_gguf_is_refused() {
        let err = select(&repo("safetensors"), Want::Unquantized).unwrap_err();
        assert_eq!(err, SelectError::NoGguf);
        assert!(err.to_string().contains("safetensors are not supported yet"), "{err}");
        assert_eq!(select(&names(&["mmproj-F16.gguf"]), Want::Unquantized).unwrap_err(), SelectError::NoGguf);
    }

    #[test]
    fn ambiguity_and_missing_files_are_reported() {
        let dup = names(&["m-Q8_0.gguf", "old/m-Q8_0.gguf"]);
        assert!(matches!(select(&dup, Want::Ftype(Ftype::Q8_0)), Err(SelectError::Ambiguous { .. })));
        let err = select_file(&repo("coder"), "Q9_9.gguf").unwrap_err();
        assert!(matches!(err, SelectError::NoSuchFile { .. }), "{err:?}");
        let err = select_file(&repo("235b"), "nowhere").unwrap_err();
        assert!(err.to_string().contains("no file or directory nowhere"), "{err}");
    }

    // ── local paths ──────────────────────────────────────────────────────────

    #[test]
    fn a_local_file_is_fetched_as_it_is() {
        let dir = crate::quantize::fixture::scratch_dir("hub-local");
        let path = dir.join("tiny.gguf");
        crate::quantize::fixture::write_tiny_llama(&path);
        let got = fetch(&ModelSource::Path(path.clone()), Want::Ftype(Ftype::Q4_K_M), &FetchOptions::default()).unwrap();
        assert_eq!(got, Fetched { parts: vec![path], ftype: Some(Ftype::F32), repo: None, revision: None }, "want does not apply");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn any_local_part_brings_its_siblings() {
        let dir = crate::quantize::fixture::scratch_dir("hub-split");
        let parts = crate::quantize::fixture::write_tiny_llama_split(&dir, "tiny");
        for given in &parts {
            let got = fetch(&ModelSource::Path(given.clone()), Want::Unquantized, &FetchOptions::default()).unwrap();
            assert_eq!(got.parts, parts, "from {}", given.display());
        }
        std::fs::remove_file(&parts[1]).unwrap();
        let err = fetch(&ModelSource::Path(parts[0].clone()), Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert!(matches!(err, FetchError::Model(_)), "a missing part is refused: {err:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_local_file_is_unreadable() {
        let err = fetch(&ModelSource::Path("/nonexistent/m.gguf".into()), Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert!(err.to_string().contains("cannot read the model at /nonexistent/m.gguf"), "{err}");
    }

    #[test]
    fn options_never_print_the_token() {
        let opts = FetchOptions { cache_dir: None, token: Some("hf_secret".into()) };
        assert!(!format!("{opts:?}").contains("hf_secret"));
    }

    #[test]
    #[cfg(not(feature = "hub"))]
    fn hub_source_is_refused_without_feature() {
        let err = fetch(&hub(CODER, None, None), Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert_eq!(err, FetchError::NotBuilt { source: format!("https://huggingface.co/{CODER}") });
        assert!(err.to_string().contains("`hub` cargo feature"), "{err}");
    }
}
