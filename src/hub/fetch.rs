//! Downloading one model from the Hugging Face Hub, through hf-hub's blocking client.
//!
//! # What a fetch does
//!
//! 1. List the repo at the requested revision, `main` by default. The listing also
//!    returns the commit that revision points at.
//! 2. Choose one model with the pure [`super::select`] or [`super::select_file`].
//! 3. Get every part of it at that commit. A part already in the cache is used as it
//!    is; any other is downloaded. Pinning the commit means a branch that moves halfway
//!    through a ten-part download cannot mix parts of two uploads.
//! 4. Open the parts as one [`Model`], and refuse a file whose `general.file_type`
//!    disagrees with the type its name gave.
//!
//! The commit is returned as [`Fetched::revision`], so a caller can record exactly what
//! it built from.
//!
//! # Why hf-hub 0.4 and its blocking client
//!
//! The 0.4 line has a client over `ureq`, so the Jade compiler's build does not pull in
//! tokio; see the note in `Cargo.toml`. Files land in the same cache `huggingface-cli`
//! and the Python library use, `models--org--repo/{blobs,snapshots,refs}`, so a model
//! already downloaded by either is not downloaded again.
//!
//! hf-hub 0.4.3 finds a cached file only through `refs/<revision>`. A model cached by
//! another tool under `refs/main` would look missing when asked for by commit, and
//! hf-hub would download all of it again. So step 3 looks in
//! `snapshots/<commit>/<file>` first, which is where every tool puts it.
//!
//! # Settings read from the environment
//!
//! These match the Python `huggingface_hub`, so one setup serves both:
//!
//! - The cache is [`FetchOptions::cache_dir`], else `HF_HUB_CACHE`, else
//!   `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else
//!   `~/.cache/huggingface/hub`.
//! - The token is [`FetchOptions::token`], else `HF_TOKEN`, else the older
//!   `HUGGING_FACE_HUB_TOKEN`, else the file `huggingface-cli login` writes: the path
//!   in `HF_TOKEN_PATH`, else `token` in the `HF_HOME` directory found the same way as
//!   the cache, normally `~/.cache/huggingface/token`. Reading these is a choice: public
//!   repos need no token, but gated ones such as Meta's Llama need one, and a build tool
//!   that ignored the login a user already did would fail for no reason they could see.
//!   The token is only ever sent to the Hub endpoint, and [`FetchOptions`] never prints it.
//! - The endpoint is `HF_ENDPOINT`, for a mirror, else `https://huggingface.co`.
//!
//! hf-hub draws a progress bar on stderr while downloading, and hides it when stderr is
//! not a terminal, so a build log gets no bar.

use std::path::{Path, PathBuf};

use hf_hub::api::sync::{ApiBuilder, ApiError};
use hf_hub::{Cache, Repo, RepoType};

use super::{fetched, is_repo_id, select, select_file, FetchError, FetchOptions, Fetched, Want};
use crate::load::Model;

const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Fetch one model of `repo`. See the module header.
pub(super) fn from_hub(
    repo: &str,
    revision: Option<&str>,
    file: Option<&str>,
    want: Want,
    opts: &FetchOptions,
) -> Result<Fetched, FetchError> {
    if !is_repo_id(repo) {
        return Err(FetchError::BadSource(format!("{repo:?} is not a Hub repo id; one looks like org/repo")));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let cache = cache_dir(opts, env, home.as_deref()).ok_or_else(|| {
        FetchError::BadSource("there is no cache directory: set HF_HUB_CACHE or HOME, or pass FetchOptions::cache_dir".into())
    })?;
    let api = ApiBuilder::from_cache(Cache::new(cache.clone()))
        .with_token(token(opts, env, read_token_file, home.as_deref()))
        .with_endpoint(env("HF_ENDPOINT").unwrap_or_else(|| DEFAULT_ENDPOINT.to_owned()))
        .build()
        .map_err(|e| hub_error(repo, "cannot start a Hub client", &e))?;

    let rev = revision.unwrap_or("main");
    let info = api
        .repo(Repo::with_revision(repo.to_owned(), RepoType::Model, rev.to_owned()))
        .info()
        .map_err(|e| hub_error(repo, &format!("cannot list revision {rev}"), &e))?;
    let names: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();
    let level = match file {
        Some(f) => select_file(&names, f),
        None => select(&names, want),
    }
    .map_err(|error| FetchError::Select { repo: repo.to_owned(), error })?;

    let pinned = api.repo(Repo::with_revision(repo.to_owned(), RepoType::Model, info.sha.clone()));
    let snapshot = cache.join(Repo::model(repo.to_owned()).folder_name()).join("snapshots").join(&info.sha);
    let mut parts = Vec::with_capacity(level.files.len());
    for f in &level.files {
        let cached = snapshot.join(f);
        let path = if cached.exists() {
            cached
        } else {
            pinned.get(f).map_err(|e| hub_error(repo, &format!("cannot download {f}"), &e))?
        };
        parts.push(path);
    }

    let model = Model::from_parts(&parts)?;
    if let (Some(named), Some(declared)) = (level.ftype, model.meta().file_type()) {
        if declared != named.value() {
            return Err(FetchError::Mismatch { file: level.files[0].clone(), named, declared });
        }
    }
    Ok(fetched(&model, Some(repo.to_owned()), Some(info.sha)))
}

/// One environment variable, with an empty value treated as unset.
fn env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

fn read_token_file(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// The `HF_HOME` directory: `HF_HOME`, else `$XDG_CACHE_HOME/huggingface`, else
/// `~/.cache/huggingface`.
fn hf_home(env: impl Fn(&str) -> Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    env("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| env("XDG_CACHE_HOME").map(|x| PathBuf::from(x).join("huggingface")))
        .or_else(|| home.map(|h| h.join(".cache").join("huggingface")))
}

/// The hf-hub cache directory, by the precedence in the module header.
fn cache_dir(opts: &FetchOptions, env: impl Fn(&str) -> Option<String>, home: Option<&Path>) -> Option<PathBuf> {
    opts.cache_dir
        .clone()
        .or_else(|| env("HF_HUB_CACHE").map(PathBuf::from))
        .or_else(|| hf_home(&env, home).map(|h| h.join("hub")))
}

/// The access token, by the precedence in the module header.
fn token(
    opts: &FetchOptions,
    env: impl Fn(&str) -> Option<String>,
    read: impl Fn(&Path) -> Option<String>,
    home: Option<&Path>,
) -> Option<String> {
    let clean = |t: String| Some(t.trim().to_owned()).filter(|t| !t.is_empty());
    opts.token
        .clone()
        .and_then(clean)
        .or_else(|| env("HF_TOKEN").and_then(clean))
        .or_else(|| env("HUGGING_FACE_HUB_TOKEN").and_then(clean))
        .or_else(|| {
            let path = env("HF_TOKEN_PATH").map(PathBuf::from).or_else(|| hf_home(&env, home).map(|h| h.join("token")))?;
            read(&path).and_then(clean)
        })
}

/// A Hub failure, with a hint for the statuses a user can do something about.
fn hub_error(repo: &str, what: &str, e: &ApiError) -> FetchError {
    let msg = e.to_string();
    let hint = match status_code(&msg) {
        Some(401 | 403) => {
            " The repo is gated or private: accept its terms on huggingface.co, then set HF_TOKEN or run \
             huggingface-cli login"
        }
        Some(404) => " Check the repo id, the revision and the file name",
        Some(429) => " The Hub is limiting requests; wait and retry, or set HF_TOKEN",
        _ => "",
    };
    FetchError::Hub { repo: repo.to_owned(), detail: format!("{what}: {msg}.{hint}") }
}

/// The HTTP status in an hf-hub error message. ureq prints a status error as
/// `<url>: status code 404`; reading it from the text avoids taking ureq on as a
/// direct dependency just to match its error type, and a miss only loses the hint.
fn status_code(msg: &str) -> Option<u16> {
    let (_, rest) = msg.rsplit_once("status code ")?;
    rest.get(..3)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::{fetch, ModelSource};
    use crate::quantize::Ftype;

    fn vars<'a>(set: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| set.iter().find(|(n, _)| *n == k).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn the_cache_follows_huggingface_hub() {
        let home = Some(Path::new("/home/u"));
        let none = FetchOptions::default();
        assert_eq!(cache_dir(&none, vars(&[]), home), Some("/home/u/.cache/huggingface/hub".into()));
        assert_eq!(cache_dir(&none, vars(&[("XDG_CACHE_HOME", "/x")]), home), Some("/x/huggingface/hub".into()));
        assert_eq!(cache_dir(&none, vars(&[("HF_HOME", "/h"), ("XDG_CACHE_HOME", "/x")]), home), Some("/h/hub".into()));
        assert_eq!(cache_dir(&none, vars(&[("HF_HUB_CACHE", "/c"), ("HF_HOME", "/h")]), home), Some("/c".into()));
        let mine = FetchOptions { cache_dir: Some("/mine".into()), token: None };
        assert_eq!(cache_dir(&mine, vars(&[("HF_HUB_CACHE", "/c")]), home), Some("/mine".into()), "the option wins");
        assert_eq!(cache_dir(&none, vars(&[]), None), None);
    }

    #[test]
    fn the_token_follows_huggingface_hub() {
        let home = Some(Path::new("/home/u"));
        let file = |p: &Path| (p == Path::new("/home/u/.cache/huggingface/token")).then(|| "hf_file\n".to_owned());
        let none = FetchOptions::default();
        assert_eq!(token(&none, vars(&[]), file, home).as_deref(), Some("hf_file"), "the login file, trimmed");
        assert_eq!(token(&none, vars(&[("HF_TOKEN", "hf_env")]), file, home).as_deref(), Some("hf_env"));
        assert_eq!(token(&none, vars(&[("HUGGING_FACE_HUB_TOKEN", "hf_old")]), file, home).as_deref(), Some("hf_old"));
        let given = FetchOptions { cache_dir: None, token: Some("hf_given".into()) };
        assert_eq!(token(&given, vars(&[("HF_TOKEN", "hf_env")]), file, home).as_deref(), Some("hf_given"));
        let elsewhere = |p: &Path| (p == Path::new("/t")).then(|| "hf_path".to_owned());
        assert_eq!(token(&none, vars(&[("HF_TOKEN_PATH", "/t")]), elsewhere, home).as_deref(), Some("hf_path"));
        assert_eq!(token(&none, vars(&[]), |_| None, home), None, "no token is fine for a public repo");
        let blank = FetchOptions { cache_dir: None, token: Some("  ".into()) };
        assert_eq!(token(&blank, vars(&[]), |_| None, home), None);
    }

    #[test]
    fn statuses_are_read_from_ureq_messages() {
        assert_eq!(status_code("request error: https://huggingface.co/api/models/x/y/revision/main: status code 401"), Some(401));
        assert_eq!(status_code("request error: https://h/x: status code 404 (redirected from https://h/y)"), Some(404));
        assert_eq!(status_code("I/O error connection refused"), None);
    }

    #[test]
    fn a_malformed_repo_id_is_refused_before_the_network() {
        let src = ModelSource::Hub { repo: "not a repo".into(), revision: None, file: None };
        let err = fetch(&src, Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert!(matches!(err, FetchError::BadSource(_)), "{err:?}");
    }

    /// Run with `cargo test -p dovata --lib -- --ignored fetches_a_small_public_gguf`.
    #[test]
    #[ignore = "downloads about 28 MB from huggingface.co"]
    fn fetches_a_small_public_gguf() {
        let dir = crate::quantize::fixture::scratch_dir("hub-network");
        let opts = FetchOptions { cache_dir: Some(dir.join("hub")), token: None };

        // A named file: exactly that one, at the commit main points to.
        let src = ModelSource::parse("https://huggingface.co/ggml-org/tiny-llamas/blob/main/stories260K.gguf").unwrap();
        let got = fetch(&src, Want::Unquantized, &opts).unwrap();
        let sha = got.revision.clone().unwrap();
        assert_eq!(sha.len(), 40, "the resolved commit, not the branch name: {sha}");
        assert_eq!(got.repo.as_deref(), Some("ggml-org/tiny-llamas"));
        assert_eq!(got.parts, [dir.join("hub/models--ggml-org--tiny-llamas/snapshots").join(&sha).join("stories260K.gguf")]);

        // Again: served from the cache, the same file untouched.
        let modified = std::fs::metadata(&got.parts[0]).unwrap().modified().unwrap();
        let again = fetch(&src, Want::Unquantized, &opts).unwrap();
        assert_eq!(again, got);
        assert_eq!(std::fs::metadata(&again.parts[0]).unwrap().modified().unwrap(), modified);

        // By type: the Q8_0 file, checked against its header.
        let q8 = fetch(&ModelSource::parse("ggml-org/tiny-llamas").unwrap(), Want::Ftype(Ftype::Q8_0), &opts).unwrap();
        assert_eq!(q8.ftype, Some(Ftype::Q8_0));
        assert!(q8.parts[0].ends_with("stories15M-q8_0.gguf"), "{:?}", q8.parts);

        // A type the repo lacks is refused with the list, and nothing is downloaded.
        let err = fetch(&ModelSource::parse("ggml-org/tiny-llamas").unwrap(), Want::Ftype(Ftype::Q6_K), &opts).unwrap_err();
        assert!(err.to_string().contains("does not offer Q6_K"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
