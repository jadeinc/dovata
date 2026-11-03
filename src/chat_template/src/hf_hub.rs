//! Author: Joe Ricks
//! File: src/chat_template/src/hf_hub.rs
//! Description: Builds a `Chat` from a model repo on the Hugging Face Hub.

use std::io::Read;

// The leading `::` names the hf-hub library, not this module, which has the same name.
use ::hf_hub::api::sync::{ApiBuilder, ApiError};
use ::hf_hub::{Cache, Repo, RepoType};
use serde::Deserialize;

use crate::{header, Chat, ChatError, Gguf, Result, Tokenizer};

const TOKENIZER_JSON: &str = "tokenizer.json";
const KEY_BASE_REPO_URL: &str = "general.base_model.0.repo_url";

/// Purpose: This struct holds the parts of a Hub repo's info that `from_hub` reads.
#[derive(Deserialize)]
pub(crate) struct RepoInfo {
    siblings: Vec<Sibling>,
    #[serde(rename = "cardData", default)]
    card: Option<Card>,
}

/// Purpose: This struct holds one file name from a Hub repo's file list.
#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

/// Purpose: This struct holds the part of a Hub repo's model card that `from_hub` reads.
#[derive(Deserialize)]
struct Card {
    /// A repo id, or a list of them for a merge. Read loosely, since the card is
    /// free-form YAML written by whoever published the repo.
    #[serde(default)]
    base_model: serde_json::Value,
}

impl RepoInfo {
    /// Purpose: This function reads the base model's repo id from a Hub repo's model
    /// card.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the repo id, such as `Qwen/Qwen3-0.6B`, or `None`.
    pub(crate) fn card_base_model(&self) -> Option<String>
    {
        let value = &self.card.as_ref()?.base_model;
        let id = value.as_str().or_else(|| value.as_array()?.first()?.as_str())?;
        let (org, name) = id.split_once('/')?;
        (!org.is_empty() && !name.is_empty() && !name.contains('/')).then(|| id.to_owned())
    }
}

/// Purpose: This function builds a [`Chat`] from a GGUF repo on the Hugging Face Hub.
///
/// Inputs:
/// - `repo`: the GGUF repo id.
/// - `revision`: a branch, tag or commit of `repo`, or `None` for `main`.
///
/// Outputs: the [`Chat`], or an error if a download fails or the files do not match.
pub(crate) fn from_hub(
    repo: &str,
    revision: Option<&str>,
) -> Result<Chat>
{
    let revision = revision.unwrap_or("main");
    let cache = Cache::from_env();
    // `with_token` replaces the token hf-hub read from the login file, so pass that one
    // on when HF_TOKEN is unset rather than clearing it.
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()).or_else(|| cache.token());
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .with_token(token.clone())
        .with_user_agent("dovata-chat-template", env!("CARGO_PKG_VERSION"))
        .build()
        .map_err(hub_error(repo))?;

    let at = Repo::with_revision(repo.to_owned(), RepoType::Model, revision.to_owned());
    let remote = api.repo(at.clone());
    let info: RepoInfo = remote
        .info_request()
        .call()
        .map_err(|e| request_error(repo, &e))?
        .into_json()
        .map_err(|e| ChatError::Hub(format!("{repo}: unreadable repo info: {e}")))?;
    let files: Vec<&str> = info.siblings.iter().map(|s| s.rfilename.as_str()).collect();

    let cached = cache.repo(at);
    let file = pick_gguf(&files, |f| cached.get(f).is_some()).ok_or_else(|| {
        ChatError::Hub(format!(
            "{repo} has no GGUF; dovata-chat-template reads the chat template from a GGUF header. \
             For a repo without one, download its tokenizer.json and use Chat::new"
        ))
    })?;
    let header = match cached.get(file) {
        Some(path) => header::read(&path)?,
        None => range_fetch(&remote.url(file), token.as_deref())?,
    };

    let tokenizer_path = if files.contains(&TOKENIZER_JSON) {
        remote.get(TOKENIZER_JSON).map_err(hub_error(repo))?
    } else {
        let base = base_repo(&header).or_else(|| info.card_base_model()).ok_or_else(|| {
            ChatError::Hub(format!(
                "{repo} has no tokenizer.json, and neither {file}'s {KEY_BASE_REPO_URL} nor the \
                 model card's base_model names a Hugging Face repo to take one from; download the \
                 model's tokenizer.json and use Chat::from_files"
            ))
        })?;
        api.model(base.clone()).get(TOKENIZER_JSON).map_err(hub_error(&base))?
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| ChatError::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;

    Chat::from_gguf(&header, tokenizer)
}

/// Purpose: This function picks which GGUF file in a Hub repo to read the header of.
///
/// Inputs:
/// - `files`: the file names in the repo.
/// - `cached`: tells whether a file is already in the local Hugging Face cache.
///
/// Outputs: the chosen file name, or `None` if the repo has no model GGUF.
pub(crate) fn pick_gguf<'a>(
    files: &[&'a str],
    cached: impl Fn(&str) -> bool,
) -> Option<&'a str>
{
    let mut candidates: Vec<&str> = files.iter().copied().filter(|f| is_model_gguf(f)).collect();
    candidates.sort_unstable();
    candidates.iter().copied().find(|f| cached(f)).or_else(|| candidates.first().copied())
}

/// Purpose: This function tells whether a file is a GGUF file that holds the model's
/// metadata.
///
/// Inputs:
/// - `name`: a file name from the repo.
///
/// Outputs: `true` for a model GGUF, `false` for anything else.
fn is_model_gguf(
    name: &str,
) -> bool
{
    let lower = name.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    let file = stem.rsplit('/').next().unwrap_or(stem);
    !file.contains("mmproj") && !file.contains("imatrix") && !is_later_split_part(file)
}

/// Purpose: This function tells whether a file name is a second or later part of a split
/// GGUF.
///
/// Inputs:
/// - `stem`: the file name without `.gguf`.
///
/// Outputs: `true` for a part after the first.
fn is_later_split_part(
    stem: &str,
) -> bool
{
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, part)) = head.rsplit_once('-') else {
        return false;
    };
    digits(total) && digits(part) && part.trim_start_matches('0') != "1"
}

/// Purpose: This function reads the base model's repo id from a GGUF header.
///
/// Inputs:
/// - `gguf`: the parsed GGUF header.
///
/// Outputs: the repo id, or `None` if the header has no Hugging Face URL.
pub(crate) fn base_repo(
    gguf: &Gguf,
) -> Option<String>
{
    let url = gguf.string(KEY_BASE_REPO_URL)?;
    let id = ["https://huggingface.co/", "http://huggingface.co/"]
        .iter()
        .find_map(|p| url.strip_prefix(p))?
        .trim_end_matches('/');
    let (org, name) = id.split_once('/')?;
    (!org.is_empty() && !name.is_empty() && !name.contains('/')).then(|| format!("{org}/{name}"))
}

/// Purpose: This function downloads and parses just the header of a GGUF file on the
/// Hub.
///
/// Inputs:
/// - `url`: the file's download URL.
/// - `token`: the Hugging Face access token, if any.
///
/// Outputs: the parsed header, or an error if a request fails or the header does not
/// parse.
fn range_fetch(
    url: &str,
    token: Option<&str>,
) -> Result<Gguf>
{
    // Keep the token on same-host redirects, which the Hub uses between revisions, and
    // drop it on the hop to the storage CDN, whose URL is already signed. That is
    // hf-hub's own policy.
    let agent = ureq::AgentBuilder::new()
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .try_proxy_from_env(true)
        .user_agent(concat!("dovata-chat-template/", env!("CARGO_PKG_VERSION")))
        .build();
    let io = |e: std::io::Error| ChatError::Hub(format!("{url}: {e}"));

    header::grow(|buf, want| {
        let start = buf.len() as u64;
        let mut req = agent.get(url).set("Range", &format!("bytes={start}-{}", want - 1));
        if let Some(t) = token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            // The previous read ended exactly at the end of the file.
            Err(ureq::Error::Status(416, _)) => return Ok(true),
            Err(e) => return Err(ChatError::Hub(format!("{url}: {e}"))),
        };
        if resp.status() == 200 {
            // The server ignored the range and is sending the file from byte 0. Read
            // the prefix wanted and drop the connection, rather than the whole file.
            buf.clear();
            resp.into_reader().take(want).read_to_end(buf).map_err(io)?;
            return Ok((buf.len() as u64) < want);
        }
        let need = want - start;
        let before = buf.len();
        resp.into_reader().take(need).read_to_end(buf).map_err(io)?;
        Ok(((buf.len() - before) as u64) < need)
    })
}

/// Purpose: This function makes a converter from hf-hub errors to `ChatError::Hub`.
///
/// Inputs:
/// - `repo`: the repo id to name in the message.
///
/// Outputs: the converter.
fn hub_error(
    repo: &str,
) -> impl Fn(ApiError) -> ChatError + '_
{
    move |e| {
        let hint = match &e {
            ApiError::RequestError(inner) => status_hint(inner),
            _ => "",
        };
        ChatError::Hub(format!("{repo}: {e}{hint}"))
    }
}

/// Purpose: This function converts an HTTP request error into a `ChatError::Hub`.
///
/// Inputs:
/// - `repo`: the repo id to name in the message.
/// - `e`: the request error.
///
/// Outputs: the `ChatError::Hub`.
fn request_error(
    repo: &str,
    e: &ureq::Error,
) -> ChatError
{
    ChatError::Hub(format!("{repo}: {e}{}", status_hint(e)))
}

/// Purpose: This function picks a fix-it hint for a Hub HTTP status.
///
/// Inputs:
/// - `e`: the request error.
///
/// Outputs: the hint, or an empty string.
fn status_hint(
    e: &ureq::Error,
) -> &'static str
{
    match e {
        // The Hub answers 401 for a repo that does not exist, as well as for one that is
        // private or gated, so as not to reveal which.
        ureq::Error::Status(401 | 403, _) => {
            "; the repo does not exist, is private, or is gated. For a gated one, accept its \
             license on huggingface.co, then set HF_TOKEN or run `huggingface-cli login`"
        }
        ureq::Error::Status(404, _) => "; check the repo id and the revision",
        _ => "",
    }
}
