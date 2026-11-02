//! Author: Joe Ricks
//! File: src/hf_hub/fetch.rs
//! Description: Downloads one model from the Hugging Face Hub into the shared Hugging
//! Face cache.

use std::path::{Path, PathBuf};

// The leading `::` names the hf-hub library, not this module, which has the same name.
use ::hf_hub::api::sync::{ApiBuilder, ApiError};
use ::hf_hub::{Cache, Repo, RepoType};

use super::{fetched, is_repo_id, levels, select, select_file, FetchError, FetchOptions, Fetched, Level, Want};
use crate::load::Model;

/// Purpose: This struct holds a Hub client for one repo, its cache directory, and the
/// repo's file list at one commit.
struct Listing {
    /// The hf-hub client.
    api: ::hf_hub::api::sync::Api,
    /// The hf-hub cache directory.
    cache: PathBuf,
    /// The commit the revision resolved to.
    sha: String,
    /// Every file path in the repo at that commit.
    names: Vec<String>,
}

/// Purpose: This function connects to the Hub and lists a repo's files at one revision.
///
/// Inputs:
/// - `endpoint`: the Hub server, taken from the link.
/// - `repo`: the repo id.
/// - `revision`: the branch, tag or commit, if any; `main` when `None`.
/// - `opts`: the cache directory and the access token.
///
/// Outputs: the [`Listing`], or an error.
fn list(
    endpoint: &str,
    repo: &str,
    revision: Option<&str>,
    opts: &FetchOptions,
) -> Result<Listing, FetchError>
{
    // `ModelSource::parse` already checks both of these, but a program can build a
    // `ModelSource::Hub` by hand, so they are checked again here. The endpoint must be
    // `https` because the access token is sent to it.
    if !is_repo_id(repo) {
        return Err(FetchError::BadSource(format!("{repo:?} is not a Hub repo id; one looks like org/repo")));
    }
    if !endpoint.starts_with("https://") {
        return Err(FetchError::BadSource(format!("the Hub endpoint {endpoint:?} is not an https:// link")));
    }
    // The user's home folder, such as /Users/joericks, or `None` when HOME is not set.
    let home: Option<PathBuf> = std::env::var_os("HOME").map(PathBuf::from);
    let Some(cache) = cache_dir(opts, env, home.as_deref()) else {
        return Err(FetchError::BadSource(
            "there is no cache directory: set HF_HUB_CACHE or HOME, or pass FetchOptions::cache_dir".into(),
        ));
    };
    let api = ApiBuilder::from_cache(Cache::new(cache.clone()))
        .with_token(token(opts, env, read_token_file, home.as_deref()))
        .with_endpoint(endpoint.trim_end_matches('/').to_owned())
        .build()
        .map_err(|e| hub_error(repo, "cannot start a Hub client", &e))?;

    let rev = revision.unwrap_or("main");
    let info = api
        .repo(Repo::with_revision(repo.to_owned(), RepoType::Model, rev.to_owned()))
        .info()
        .map_err(|e| hub_error(repo, &format!("cannot list revision {rev}"), &e))?;
    let names = info.siblings.into_iter().map(|s| s.rfilename).collect();
    Ok(Listing { api, cache, sha: info.sha, names })
}

/// Purpose: This function lists the models a Hub repo offers, without downloading any.
///
/// Inputs:
/// - `endpoint`: the Hub server, taken from the link.
/// - `repo`: the repo id.
/// - `revision`: the branch, tag or commit, if any.
/// - `opts`: the cache directory and the access token.
///
/// Outputs: one [`Level`] per model, most precise first, or an error.
pub(super) fn offered(
    endpoint: &str,
    repo: &str,
    revision: Option<&str>,
    opts: &FetchOptions,
) -> Result<Vec<Level>, FetchError>
{
    Ok(levels(&list(endpoint, repo, revision, opts)?.names))
}

/// Purpose: This function downloads one model of a Hub repo.
///
/// Inputs:
/// - `endpoint`: the Hub server, taken from the link.
/// - `repo`: the repo id.
/// - `revision`: the branch, tag or commit, if any.
/// - `file`: the named file or directory, if any.
/// - `want`: which model to take when no file is named.
/// - `opts`: the cache directory and the access token.
///
/// Outputs: the model's file paths and commit, or an error.
pub(super) fn from_hub(
    endpoint: &str,
    repo: &str,
    revision: Option<&str>,
    file: Option<&str>,
    want: Want,
    opts: &FetchOptions,
) -> Result<Fetched, FetchError>
{
    let Listing { api, cache, sha, names } = list(endpoint, repo, revision, opts)?;
    let level = match file {
        Some(f) => select_file(&names, f),
        None => select(&names, want),
    }
    .map_err(|error| FetchError::Select { repo: repo.to_owned(), error })?;

    let pinned = api.repo(Repo::with_revision(repo.to_owned(), RepoType::Model, sha.clone()));
    let snapshot = cache.join(Repo::model(repo.to_owned()).folder_name()).join("snapshots").join(&sha);
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
    Ok(fetched(&model, Some(repo.to_owned()), Some(sha)))
}

/// Purpose: This function reads an environment variable, treating an empty one as unset.
///
/// Inputs:
/// - `key`: the variable's name.
///
/// Outputs: the value, or `None`.
fn env(
    key: &str,
) -> Option<String>
{
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// Purpose: This function reads a token file.
///
/// Inputs:
/// - `path`: the file.
///
/// Outputs: the file's text, or `None`.
fn read_token_file(
    path: &Path,
) -> Option<String>
{
    std::fs::read_to_string(path).ok()
}

/// Purpose: This function finds the Hugging Face home directory.
///
/// Inputs:
/// - `env`: reads an environment variable.
/// - `home`: the home directory, if known.
///
/// Outputs: the directory, or `None`.
fn hf_home(
    env: impl Fn(&str) -> Option<String>,
    home: Option<&Path>,
) -> Option<PathBuf>
{
    // The first of these that is set wins.
    if let Some(dir) = env("HF_HOME") {
        return Some(PathBuf::from(dir));
    }
    if let Some(xdg) = env("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg).join("huggingface"));
    }
    if let Some(home) = home {
        return Some(home.join(".cache").join("huggingface"));
    }
    None
}

/// Purpose: This function finds the Hugging Face cache directory.
///
/// Inputs:
/// - `opts`: the fetch options.
/// - `env`: reads an environment variable.
/// - `home`: the home directory, if known.
///
/// Outputs: the directory, or `None`.
fn cache_dir(
    opts: &FetchOptions,
    env: impl Fn(&str) -> Option<String>,
    home: Option<&Path>,
) -> Option<PathBuf>
{
    // The first of these that is set wins, in the order Hugging Face's Python library
    // uses, so both find the same folder.
    if let Some(dir) = &opts.cache_dir {
        return Some(dir.clone());
    }
    if let Some(dir) = env("HF_HUB_CACHE") {
        return Some(PathBuf::from(dir));
    }
    if let Some(hf_home) = hf_home(&env, home) {
        return Some(hf_home.join("hub"));
    }
    None
}

/// Purpose: This function finds the Hub access token.
///
/// Inputs:
/// - `opts`: the fetch options.
/// - `env`: reads an environment variable.
/// - `read`: reads a file.
/// - `home`: the home directory, if known.
///
/// Outputs: the token, or `None`.
fn token(
    opts: &FetchOptions,
    env: impl Fn(&str) -> Option<String>,
    read: impl Fn(&Path) -> Option<String>,
    home: Option<&Path>,
) -> Option<String>
{
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

/// Purpose: This function turns an hf-hub error into a fetch error with a hint.
///
/// Inputs:
/// - `repo`: the repo id.
/// - `what`: what was being done.
/// - `e`: the hf-hub error.
///
/// Outputs: the fetch error.
fn hub_error(
    repo: &str,
    what: &str,
    e: &ApiError,
) -> FetchError
{
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

/// Purpose: This function reads the HTTP status from an hf-hub error message.
///
/// Inputs:
/// - `msg`: the message.
///
/// Outputs: the status, or `None`.
fn status_code(
    msg: &str,
) -> Option<u16>
{
    let (_, rest) = msg.rsplit_once("status code ")?;
    rest.get(..3)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_hub::{fetch, ModelSource};
    use crate::quantize::Ftype;

    /// Purpose: This function builds a stand-in environment from fixed pairs.
    ///
    /// Inputs:
    /// - `set`: the variables and their values.
    ///
    /// Outputs: a lookup function.
    fn vars<'a>(
        set: &'a [(&'a str, &'a str)],
    ) -> impl Fn(&str) -> Option<String> + 'a
    {
        move |k| set.iter().find(|(n, _)| *n == k).map(|(_, v)| (*v).to_owned())
    }

    #[test]
    fn the_cache_follows_huggingface_hub()
    {
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
    fn the_token_follows_huggingface_hub()
    {
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
    fn statuses_are_read_from_ureq_messages()
    {
        assert_eq!(status_code("request error: https://huggingface.co/api/models/x/y/revision/main: status code 401"), Some(401));
        assert_eq!(status_code("request error: https://h/x: status code 404 (redirected from https://h/y)"), Some(404));
        assert_eq!(status_code("I/O error connection refused"), None);
    }

    /// Purpose: This function builds a Hub source for the tests.
    ///
    /// Inputs:
    /// - `endpoint`: the Hub server.
    /// - `repo`: the repo id.
    ///
    /// Outputs: the source.
    fn source(
        endpoint: &str,
        repo: &str,
    ) -> ModelSource
    {
        ModelSource::Hub { endpoint: endpoint.into(), repo: repo.into(), revision: None, file: None }
    }

    #[test]
    fn a_malformed_repo_id_is_refused_before_the_network()
    {
        let err = fetch(&source("https://huggingface.co", "not a repo"), Want::Unquantized, &FetchOptions::default()).unwrap_err();
        assert!(matches!(err, FetchError::BadSource(_)), "{err:?}");
    }

    #[test]
    fn an_endpoint_that_is_not_https_is_refused_before_the_network()
    {
        for endpoint in ["", "http://huggingface.co", "huggingface.co"] {
            let err = fetch(&source(endpoint, "org/repo"), Want::Unquantized, &FetchOptions::default()).unwrap_err();
            assert!(matches!(&err, FetchError::BadSource(m) if m.contains("https://")), "{endpoint:?}: {err:?}");
        }
    }

    /// Run with `cargo test -p dovata --lib -- --ignored fetches_a_small_public_gguf`.
    #[test]
    #[ignore = "downloads about 28 MB from huggingface.co"]
    fn fetches_a_small_public_gguf()
    {
        let dir = crate::quantize::fixture::scratch_dir("hf-hub-network");
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
        let repo = ModelSource::parse("https://huggingface.co/ggml-org/tiny-llamas").unwrap();
        let q8 = fetch(&repo, Want::Ftype(Ftype::Q8_0), &opts).unwrap();
        assert_eq!(q8.ftype, Some(Ftype::Q8_0));
        assert!(q8.parts[0].ends_with("stories15M-q8_0.gguf"), "{:?}", q8.parts);

        // A type the repo lacks is refused with the list, and nothing is downloaded.
        let err = fetch(&repo, Want::Ftype(Ftype::Q6_K), &opts).unwrap_err();
        assert!(err.to_string().contains("does not offer Q6_K"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
