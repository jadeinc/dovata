//! Author: Joe Ricks
//! File: src/chat/src/tests/real.rs
//! Description: Tests against real model files and the jinja2 cases, when environment
//! variables name them.

use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::json;

use super::fixture;
use crate::{header, Chat, ChatError, Message};

/// Purpose: This function reads a file path from an environment variable.
///
/// Inputs:
/// - `var`: the variable's name.
///
/// Outputs: the path, or `None` if the variable is not set.
fn env_path(var: &str) -> Option<PathBuf> {
    let path = std::env::var_os(var).map(PathBuf::from);
    if path.is_none() {
        eprintln!("skipping: set {var} to run this test against a real model");
    }
    path
}

/// Purpose: This function builds a test conversation that uses every part of the
/// Qwen3-Coder template.
///
/// Inputs: none.
///
/// Outputs: the messages and the tool definitions.
fn conversation() -> (Vec<Message>, serde_json::Value) {
    let messages = vec![
        Message::system("You are a coding assistant."),
        Message::user("What is the weather in Paris?"),
        Message::tool_calls(json!([{
            "type": "function",
            "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
        }])),
        Message::tool("18C and sunny"),
        Message::user("Thanks."),
    ];
    let extra = json!({ "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string", "description": "City name" } },
                "required": ["city"]
            }
        }
    }]});
    (messages, extra)
}

#[test]
fn qwen3_coder_header_renders() {
    let Some(path) = env_path("DOVATA_TEST_HEADER") else {
        return;
    };
    let gguf = header::read(&path).unwrap();
    let template = gguf.string("tokenizer.chat_template").expect("header has a chat template").to_owned();
    let pieces = header::Pieces::of(&gguf).unwrap();
    assert_eq!(pieces.eos.as_deref(), Some("<|im_end|>"));
    assert_eq!(pieces.bos, None, "Qwen ships no BOS id");

    // The fixture tokenizer cannot tokenize Qwen, but rendering never touches it.
    let chat = Chat::new(template, fixture::tokenizer(), pieces.bos.clone(), pieces.eos.clone()).unwrap();
    let (messages, extra) = conversation();

    let prompt = chat.render_with(&messages, true, &extra).unwrap();
    println!("{prompt}");
    assert!(prompt.starts_with("<|im_start|>system\nYou are a coding assistant."), "{prompt}");
    assert!(prompt.contains("<tools>\n<function>\n<name>get_weather</name>"), "{prompt}");
    assert!(prompt.contains("<parameter>\n<name>city</name>\n<type>string</type>"), "{prompt}");
    assert!(prompt.contains("<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>"), "{prompt}");
    assert!(prompt.contains("<|im_start|>user\n<tool_response>\n18C and sunny\n</tool_response>\n<|im_end|>"), "{prompt}");
    assert!(prompt.ends_with("<|im_start|>user\nThanks.<|im_end|>\n<|im_start|>assistant\n"), "{prompt}");

    let finished = chat.render_with(&messages, false, &extra).unwrap();
    assert!(!finished.ends_with("<|im_start|>assistant\n"), "{finished}");
    assert_eq!(format!("{finished}<|im_start|>assistant\n"), prompt);

    // And the check refuses the fixture tokenizer against the real vocabulary.
    let err = Chat::from_gguf(&gguf, fixture::tokenizer()).unwrap_err();
    assert!(matches!(err, ChatError::VocabMismatch { gguf: 151_936, .. }), "{err:?}");
}

#[test]
fn qwen3_coder_tokenizer_matches_its_header() {
    let (Some(head), Some(tok)) = (env_path("DOVATA_TEST_HEADER"), env_path("DOVATA_TEST_TOKENIZER")) else {
        return;
    };
    // 151,936 GGUF tokens against 151,665 in tokenizer.json: the padding rule, on the
    // real file.
    let chat = Chat::from_files(&head, &tok).unwrap();
    let (messages, extra) = conversation();
    let prompt = chat.render_with(&messages, true, &extra).unwrap();
    let ids = chat.encode(&prompt, false).unwrap();
    // <|im_start|> is 151644 and <|im_end|> is 151645 in every Qwen2-family vocabulary.
    assert_eq!(ids.first(), Some(&151_644));
    assert_eq!(ids.iter().filter(|&&id| id == 151_645).count(), 5);
    assert_eq!(chat.decode(&ids, false).unwrap(), prompt);
}

/// Purpose: This struct holds one case written by `jinja2_cases.py`.
#[derive(Deserialize)]
struct Case {
    name: String,
    template: String,
    bos: Option<String>,
    eos: Option<String>,
    messages: Vec<Message>,
    add_generation_prompt: bool,
    extra: serde_json::Value,
    expected: Expected,
}

/// Purpose: This struct holds what jinja2 produced for one case: a prompt or a refusal.
#[derive(Deserialize)]
struct Expected {
    ok: Option<String>,
    raised: Option<String>,
}

#[test]
fn matches_jinja2_cases() {
    let Some(path) = env_path("DOVATA_CHAT_CASES") else {
        return;
    };
    if !cfg!(feature = "preserve_order") {
        // The cases' tools are in Python's dict order, which a serde_json::Value keeps
        // only with the feature on; without it, tool JSON differs in key order alone.
        eprintln!("skipping: run with --features preserve_order to compare against jinja2");
        return;
    }
    let cases: Vec<Case> = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut failures = Vec::new();
    for case in &cases {
        let chat = Chat::new(case.template.clone(), fixture::tokenizer(), case.bos.clone(), case.eos.clone())
            .unwrap()
            .with_time(UNIX_EPOCH + Duration::from_secs(1_721_984_707));
        let got = chat.render_with(&case.messages, case.add_generation_prompt, &case.extra);
        let same = match (&got, &case.expected) {
            (Ok(out), Expected { ok: Some(want), .. }) => out == want,
            (Err(ChatError::Raised(msg)), Expected { raised: Some(want), .. }) => msg == want,
            _ => false,
        };
        if !same {
            failures.push(format!("{}\n  jinja2: {:?}\n  ours:   {got:?}", case.name, case.expected.ok));
        }
    }
    assert!(failures.is_empty(), "{} of {} cases differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
    println!("all {} cases match jinja2", cases.len());
}
