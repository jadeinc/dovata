//! Author: Joe Ricks
//! File: src/chat/src/tests/hub.rs
//! Description: Tests how `from_hub` picks its files, plus two network tests that are
//! off by default.

use serde_json::json;

use super::fixture::Header;
use crate::hub::{base_repo, pick_gguf, RepoInfo};
use crate::{Chat, Message};

#[test]
fn picks_a_cached_gguf_first_then_the_first_by_name() {
    let files = ["README.md", "Model-Q8_0.gguf", "Model-BF16.gguf", "Model-Q4_K_M.gguf"];
    assert_eq!(pick_gguf(&files, |_| false), Some("Model-BF16.gguf"));
    assert_eq!(pick_gguf(&files, |f| f == "Model-Q4_K_M.gguf"), Some("Model-Q4_K_M.gguf"));
}

#[test]
fn skips_projectors_imatrix_and_later_split_parts() {
    let files = [
        "mmproj-Model-F16.gguf",
        "Model.imatrix.gguf",
        "BF16/Model-BF16-00002-of-00002.gguf",
        "BF16/Model-BF16-00001-of-00002.gguf",
        "params",
    ];
    assert_eq!(pick_gguf(&files, |_| false), Some("BF16/Model-BF16-00001-of-00002.gguf"));
    assert_eq!(pick_gguf(&["mmproj-F16.gguf", "tokenizer.json"], |_| false), None);
}

#[test]
fn base_repo_comes_from_the_header_url() {
    let with_url = |url: &str| Header::chatml().str("general.base_model.0.repo_url", url).parse();
    assert_eq!(
        base_repo(&with_url("https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct")).as_deref(),
        Some("Qwen/Qwen3-Coder-30B-A3B-Instruct")
    );
    assert_eq!(base_repo(&with_url("https://huggingface.co/Qwen/Qwen3-0.6B/")).as_deref(), Some("Qwen/Qwen3-0.6B"));
    assert_eq!(base_repo(&with_url("https://example.com/Qwen/Qwen3-0.6B")), None);
    assert_eq!(base_repo(&with_url("https://huggingface.co/Qwen")), None);
    assert_eq!(base_repo(&with_url("https://huggingface.co/Qwen/Qwen3/tree/main")), None);
    assert_eq!(base_repo(&Header::chatml().parse()), None);
}

#[test]
fn base_model_comes_from_the_model_card_otherwise() {
    let card = |base: serde_json::Value| -> Option<String> {
        let info: RepoInfo =
            serde_json::from_value(json!({ "siblings": [], "cardData": { "base_model": base } })).unwrap();
        info.card_base_model()
    };
    // Qwen's own GGUF repos write a string, Unsloth's a list.
    assert_eq!(card(json!("Qwen/Qwen3-0.6B")).as_deref(), Some("Qwen/Qwen3-0.6B"));
    assert_eq!(card(json!(["Qwen/Qwen3-Coder-30B-A3B-Instruct", "x/y"])).as_deref(), Some("Qwen/Qwen3-Coder-30B-A3B-Instruct"));
    assert_eq!(card(json!(null)), None);
    assert_eq!(card(json!({ "name": "odd" })), None);
    assert_eq!(card(json!("not-a-repo-id")), None);
    let no_card: RepoInfo = serde_json::from_value(json!({ "siblings": [{ "rfilename": "a.gguf" }] })).unwrap();
    assert_eq!(no_card.card_base_model(), None);
}

/// Range-fetches the header of `Qwen/Qwen3-0.6B-GGUF`, which has no tokenizer.json and
/// whose header names no base model, so the tokenizer comes from the model card's
/// `base_model`, `Qwen/Qwen3-0.6B`. Downloads 8 MB of header and 11 MB of tokenizer.
#[test]
#[ignore = "network: fetches from huggingface.co"]
fn from_hub_small_repo() {
    let chat = Chat::from_hub("Qwen/Qwen3-0.6B-GGUF", None).unwrap();
    assert_eq!(chat.eos(), Some("<|im_end|>"));
    let prompt = chat.render(&[Message::user("Hi")], true).unwrap();
    assert!(prompt.ends_with("<|im_start|>assistant\n"), "{prompt}");
    let ids = chat.encode(&prompt, false).unwrap();
    assert_eq!(ids.first(), Some(&151_644), "<|im_start|>");
    assert_eq!(chat.decode(&ids, false).unwrap(), prompt);
}

/// The other fallback: Unsloth's header names its base model in
/// `general.base_model.0.repo_url`. Also a repo with many quants and a subfolder.
#[test]
#[ignore = "network: fetches from huggingface.co"]
fn from_hub_reads_base_model_from_the_header() {
    let chat = Chat::from_hub("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", None).unwrap();
    assert!(chat.template().contains("Unsloth Chat template fixes"));
    let prompt = chat.render(&[Message::user("Hi")], true).unwrap();
    assert_eq!(prompt, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
}
