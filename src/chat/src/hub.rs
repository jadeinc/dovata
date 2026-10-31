//! [`Chat::from_hub`](crate::Chat::from_hub): the GGUF header and `tokenizer.json` from
//! the Hugging Face Hub.
//!
//! # The rule
//!
//! `repo` is a GGUF repo, the same id dovata's own Hub fetch takes, such as
//! `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`.
//!
//! 1. The template and the special tokens come from a GGUF header in that repo, exactly
//!    as [`Chat::from_files`](crate::Chat::from_files) reads them. Every quant of one
//!    model carries the same tokenizer metadata, so any of its GGUFs will do. Projector
//!    (`mmproj`) and importance-matrix (`imatrix`) files are skipped, since neither is
//!    the language model, and so are split parts after the first, which carry no
//!    metadata. A GGUF already in the local hf-hub cache is preferred: it is the file
//!    the program will run, and reading it costs no network. Otherwise the first one in
//!    name order is range-fetched, its header only, never its tensor data.
//! 2. `tokenizer.json` comes from the same repo when it has one at its root. GGUF repos
//!    usually do not, so otherwise it comes from the model the GGUF was converted from.
//!    That is the header's `general.base_model.0.repo_url` when the converter recorded
//!    it, as Unsloth's files do, and otherwise the `base_model` the repo's model card
//!    names, which is the only place Qwen's own GGUF repos record it. The base repo is
//!    read at `main`: `revision` names a commit of the GGUF repo, and the base repo has
//!    a history of its own.
//! 3. The pair is checked like any other, by [`vocab::check`](crate::vocab::check).
//!    That check is what makes step 2 safe. A base repo whose tokenizer has since
//!    changed is refused rather than used.
//!
//! The alternative was to take the template from the base repo's
//! `tokenizer_config.json` and need no GGUF at all. It was rejected because the GGUF is
//! what the program runs, and GGUF publishers do fix templates: the Unsloth
//! Qwen3-Coder GGUF carries a corrected one, so the base repo's would be the older
//! template.
//!
//! # Tokens
//!
//! A Hub token comes from `HF_TOKEN`, or else from the file `huggingface-cli login`
//! writes, which is where hf-hub looks. Gated base models, such as Meta's Llama, need
//! one.

use std::io::Read;

use hf_hub::api::sync::{ApiBuilder, ApiError};
use hf_hub::{Cache, Repo, RepoType};
use serde::Deserialize;

use crate::{header, Chat, ChatError, Gguf, Result, Tokenizer};

const TOKENIZER_JSON: &str = "tokenizer.json";
const KEY_BASE_REPO_URL: &str = "general.base_model.0.repo_url";

/// The part of the Hub's repo info this module reads. hf-hub's own `RepoInfo` keeps
/// only the file list and the commit, and the model card's `base_model` is needed too.
#[derive(Deserialize)]
pub(crate) struct RepoInfo {
    siblings: Vec<Sibling>,
    #[serde(rename = "cardData", default)]
    card: Option<Card>,
}

#[derive(Deserialize)]
struct Sibling {
    rfilename: String,
}

#[derive(Deserialize)]
struct Card {
    /// A repo id, or a list of them for a merge. Read loosely, since the card is
    /// free-form YAML written by whoever published the repo.
    #[serde(default)]
    base_model: serde_json::Value,
}

impl RepoInfo {
    /// The first `base_model` the model card names, if it is a plain `org/name` id.
    pub(crate) fn card_base_model(&self) -> Option<String> {
        let value = &self.card.as_ref()?.base_model;
        let id = value.as_str().or_else(|| value.as_array()?.first()?.as_str())?;
        let (org, name) = id.split_once('/')?;
        (!org.is_empty() && !name.is_empty() && !name.contains('/')).then(|| id.to_owned())
    }
}

pub(crate) fn from_hub(repo: &str, revision: Option<&str>) -> Result<Chat> {
    let revision = revision.unwrap_or("main");
    let cache = Cache::from_env();
    // `with_token` replaces the token hf-hub read from the login file, so pass that one
    // on when HF_TOKEN is unset rather than clearing it.
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()).or_else(|| cache.token());
    let api = ApiBuilder::from_env()
        .with_progress(false)
        .with_token(token.clone())
        .with_user_agent("dovata-chat", env!("CARGO_PKG_VERSION"))
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
            "{repo} has no GGUF; dovata-chat reads the chat template from a GGUF header. \
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

/// The GGUF to read the header of, by the rule in the module docs. `cached` says
/// whether a file is in the local hf-hub cache.
pub(crate) fn pick_gguf<'a>(files: &[&'a str], cached: impl Fn(&str) -> bool) -> Option<&'a str> {
    let mut candidates: Vec<&str> = files.iter().copied().filter(|f| is_model_gguf(f)).collect();
    candidates.sort_unstable();
    candidates.iter().copied().find(|f| cached(f)).or_else(|| candidates.first().copied())
}

fn is_model_gguf(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    let Some(stem) = lower.strip_suffix(".gguf") else {
        return false;
    };
    let file = stem.rsplit('/').next().unwrap_or(stem);
    !file.contains("mmproj") && !file.contains("imatrix") && !is_later_split_part(file)
}

/// `<name>-00002-of-00003`: a split part after the first, which has no metadata.
fn is_later_split_part(stem: &str) -> bool {
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let Some((head, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((_, part)) = head.rsplit_once('-') else {
        return false;
    };
    digits(total) && digits(part) && part.trim_start_matches('0') != "1"
}

/// The base model's repo id, from the URL the converter records. Only a
/// `huggingface.co/<org>/<name>` URL counts; anything else is somewhere this package
/// cannot fetch from.
pub(crate) fn base_repo(gguf: &Gguf) -> Option<String> {
    let url = gguf.string(KEY_BASE_REPO_URL)?;
    let id = ["https://huggingface.co/", "http://huggingface.co/"]
        .iter()
        .find_map(|p| url.strip_prefix(p))?
        .trim_end_matches('/');
    let (org, name) = id.split_once('/')?;
    (!org.is_empty() && !name.is_empty() && !name.contains('/')).then(|| format!("{org}/{name}"))
}

/// Read a remote GGUF's header with HTTP range requests, in the growing prefixes
/// [`header::grow`] asks for. hf-hub has no range API of its own, so this is `ureq`, the
/// client hf-hub is built on.
fn range_fetch(url: &str, token: Option<&str>) -> Result<Gguf> {
    // Keep the token on same-host redirects, which the Hub uses between revisions, and
    // drop it on the hop to the storage CDN, whose URL is already signed. That is
    // hf-hub's own policy.
    let agent = ureq::AgentBuilder::new()
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .try_proxy_from_env(true)
        .user_agent(concat!("dovata-chat/", env!("CARGO_PKG_VERSION")))
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

/// Map an hf-hub error to [`ChatError::Hub`].
fn hub_error(repo: &str) -> impl Fn(ApiError) -> ChatError + '_ {
    move |e| {
        let hint = match &e {
            ApiError::RequestError(inner) => status_hint(inner),
            _ => "",
        };
        ChatError::Hub(format!("{repo}: {e}{hint}"))
    }
}

/// Map a request error to [`ChatError::Hub`].
fn request_error(repo: &str, e: &ureq::Error) -> ChatError {
    ChatError::Hub(format!("{repo}: {e}{}", status_hint(e)))
}

/// A hint for the statuses that mean the caller has to do something.
fn status_hint(e: &ureq::Error) -> &'static str {
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
