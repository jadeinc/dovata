//! Author: Joe Ricks
//! File: src/chat_template/src/tests/mod.rs
//! Description: Tests `Chat` itself and lists the other test files.

mod fixture;
#[cfg(feature = "hf-hub")]
mod hf_hub;
mod real;
mod template;
mod vocab;

use serde_json::json;

use crate::{Chat, ChatError, Message};
use fixture::{Header, Scratch, BOS_ID, CHATML, EOS_ID};

#[test]
fn encode_decode_round_trip()
{
    let dir = Scratch::new();
    let gguf = dir.write("model.gguf", Header::chatml().bytes());
    let tok = dir.write("tokenizer.json", fixture::tokenizer_json());
    let chat = Chat::from_files(&gguf, &tok).unwrap();

    let ids = chat.encode("hello world", false).unwrap();
    assert_eq!(ids, [4, 5]);
    assert_eq!(chat.decode(&ids, false).unwrap(), "hello world");

    // A rendered prompt: the turn markers are special tokens and come out as single
    // ids, and skipping specials on the way back drops them.
    let prompt = chat.render(&[Message::user("hello world")], false).unwrap();
    let ids = chat.encode(&prompt, false).unwrap();
    assert_eq!(ids, [1, 7, 4, 5, 2]);
    assert_eq!(chat.decode(&ids, true).unwrap(), "user hello world");
}

#[test]
fn add_special_runs_the_post_processor()
{
    let chat = fixture::chat();
    assert_eq!(chat.encode("hello", true).unwrap(), [BOS_ID, 4]);
    assert_eq!(chat.encode("hello", false).unwrap(), [4]);
}

#[test]
fn chatml_renders_exactly()
{
    let chat = fixture::chat();
    let messages = [Message::system("Be brief."), Message::user("Hi"), Message::assistant("hello world")];
    assert_eq!(
        chat.render(&messages, false).unwrap(),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n\
         <|im_start|>assistant\nhello world<|im_end|>\n"
    );
}

#[test]
fn generation_prompt_appended()
{
    let chat = fixture::chat();
    let messages = [Message::user("Hi")];
    let without = chat.render(&messages, false).unwrap();
    let with = chat.render(&messages, true).unwrap();
    assert_eq!(without, "<|im_start|>user\nHi<|im_end|>\n");
    assert_eq!(with, format!("{without}<|im_start|>assistant\n"));
}

#[test]
fn missing_template_is_a_clear_error()
{
    let header = Header::chatml()
        .without("tokenizer.chat_template")
        .str("tokenizer.chat_template.tool_use", CHATML)
        .parse();
    let err = Chat::from_gguf(&header, fixture::tokenizer()).unwrap_err();
    assert!(matches!(&err, ChatError::MissingTemplate { named } if named == &["tool_use"]), "{err:?}");
    let msg = err.to_string();
    assert!(msg.contains("tokenizer.chat_template"), "{msg}");
    assert!(msg.contains("tool_use"), "{msg}");
    assert!(msg.contains("Chat::new"), "{msg}");
}

#[test]
fn bos_and_eos_come_from_the_header()
{
    let chat = fixture::chat();
    assert_eq!(chat.bos(), Some("<s>"));
    assert_eq!(chat.eos(), Some("<|im_end|>"));
    assert_eq!(chat.template(), CHATML);

    let printed = Chat::new("{{ bos_token }}|{{ eos_token }}".into(), fixture::tokenizer(), chat.bos.clone(), chat.eos.clone())
        .unwrap()
        .render(&[], false)
        .unwrap();
    assert_eq!(printed, "<s>|<|im_end|>");
}

#[test]
fn absent_bos_is_none_and_undefined_in_template()
{
    // Qwen ships no BOS id. transformers then leaves `bos_token` undefined, and a
    // template that prints it prints nothing.
    let header = Header::chatml().without("tokenizer.ggml.bos_token_id").parse();
    let chat = Chat::from_gguf(&header, fixture::tokenizer()).unwrap();
    assert_eq!(chat.bos(), None);
    let chat = Chat::new("[{{ bos_token }}]{{ bos_token is defined }}".into(), fixture::tokenizer(), None, None).unwrap();
    assert_eq!(chat.render(&[], false).unwrap(), "[]False");
}

#[test]
fn out_of_range_eos_is_refused()
{
    let header = Header::chatml().u32("tokenizer.ggml.eos_token_id", 99).parse();
    let err = Chat::from_gguf(&header, fixture::tokenizer()).unwrap_err();
    assert!(matches!(err, ChatError::BadTokenId { id: 99, vocab: 12, .. }), "{err:?}");
}

#[test]
fn from_files_reads_a_header_only_file()
{
    // A range-fetched prefix: metadata and nothing after it. Also proves the reader
    // stops at the end of a short file instead of asking for more.
    let dir = Scratch::new();
    let mut bytes = Header::chatml().bytes();
    bytes.extend_from_slice(&[0xAB; 1000]);
    let gguf = dir.write("head.gguf", bytes);
    let tok = dir.write("tokenizer.json", fixture::tokenizer_json());
    assert!(Chat::from_files(&gguf, &tok).is_ok());
}

#[test]
fn metadata_longer_than_the_first_read_is_read_whole()
{
    // Metadata that ends past the first 8 MiB read, as a 250k-token vocabulary can.
    let dir = Scratch::new();
    let padding = "x".repeat(usize::try_from(crate::header::FIRST_READ).unwrap() + 4096);
    let gguf = dir.write("big.gguf", Header::chatml().str("general.description", &padding).bytes());
    let tok = dir.write("tokenizer.json", fixture::tokenizer_json());
    let chat = Chat::from_files(&gguf, &tok).unwrap();
    assert_eq!(chat.eos(), Some("<|im_end|>"));
}

#[test]
fn truncated_metadata_is_a_header_error()
{
    let dir = Scratch::new();
    let bytes = Header::chatml().bytes();
    let gguf = dir.write("cut.gguf", &bytes[..bytes.len() / 2]);
    let tok = dir.write("tokenizer.json", fixture::tokenizer_json());
    let err = Chat::from_files(&gguf, &tok).unwrap_err();
    assert!(matches!(err, ChatError::Header(_)), "{err:?}");
}

#[test]
fn missing_files_name_the_path()
{
    let dir = Scratch::new();
    let missing = dir.path().join("nope.gguf");
    let tok = dir.write("tokenizer.json", fixture::tokenizer_json());
    let err = Chat::from_files(&missing, &tok).unwrap_err();
    assert!(matches!(&err, ChatError::Io { path, .. } if path == &missing), "{err:?}");

    let gguf = dir.write("model.gguf", Header::chatml().bytes());
    let err = Chat::from_files(&gguf, &dir.path().join("nope.json")).unwrap_err();
    assert!(matches!(&err, ChatError::Tokenizer(m) if m.contains("nope.json")), "{err:?}");
}

#[test]
fn extra_reaches_the_template_and_overrides_defaults()
{
    let template = "{{ tools | length }} {{ enable_thinking }} {{ bos_token }}";
    let chat = Chat::new(template.into(), fixture::tokenizer(), Some("<s>".into()), None).unwrap();
    let extra = json!({ "tools": [{"name": "a"}, {"name": "b"}], "enable_thinking": false, "bos_token": "B" });
    // Booleans print as Python prints them, as they do under jinja2.
    assert_eq!(chat.render_with(&[], false, &extra).unwrap(), "2 False B");
}

#[test]
fn extra_must_be_a_map()
{
    let chat = fixture::chat();
    let err = chat.render_with(&[], false, &json!([1, 2])).unwrap_err();
    assert!(matches!(&err, ChatError::BadInput(m) if m.contains("map")), "{err:?}");
    // Nothing at all is fine: that is what `render` passes.
    assert!(chat.render_with(&[], false, &json!(null)).is_ok());
}

#[test]
fn extra_cannot_override_the_arguments()
{
    let chat = fixture::chat();
    for key in ["messages", "add_generation_prompt"] {
        let err = chat.render_with(&[], false, &json!({ key: true })).unwrap_err();
        assert!(matches!(&err, ChatError::BadInput(m) if m.contains(key)), "{err:?}");
    }
}

#[test]
fn message_shadowing_a_field_is_refused()
{
    let chat = fixture::chat();
    let bad = Message::user("Hi").with("content", json!("other"));
    let err = chat.render(&[bad], false).unwrap_err();
    assert!(matches!(&err, ChatError::BadInput(m) if m.contains("message 0")), "{err:?}");
}

#[test]
fn tool_call_message_has_null_content_and_its_calls()
{
    // DeepSeek V3 renders tool calls only for `content is none`.
    let template = "{% for m in messages %}{{ m.content is none }} {{ m.tool_calls[0].function.name }}{% endfor %}";
    let chat = Chat::new(template.into(), fixture::tokenizer(), None, None).unwrap();
    let call = Message::tool_calls(json!([{ "type": "function", "function": { "name": "f", "arguments": "{}" } }]));
    assert_eq!(chat.render(&[call], false).unwrap(), "True f");
}

#[test]
fn message_reads_openai_json()
{
    let messages: Vec<Message> = serde_json::from_value(json!([
        { "role": "user", "content": "Hi" },
        { "role": "assistant", "content": null, "tool_calls": [{ "id": "c1" }] },
        { "role": "tool", "tool_call_id": "c1", "content": "42" }
    ]))
    .unwrap();
    assert_eq!(messages[0], Message::user("Hi"));
    assert_eq!(messages[1].content, None);
    assert_eq!(messages[1].extra["tool_calls"], json!([{ "id": "c1" }]));
    assert_eq!(messages[2], Message::tool("42").with("tool_call_id", json!("c1")));
    // And back: `content` stays an explicit null, which templates test for.
    assert_eq!(serde_json::to_value(&messages[1]).unwrap()["content"], json!(null));
}

#[test]
fn chat_is_debug_without_dumping_the_vocabulary()
{
    let shown = format!("{:?}", fixture::chat());
    assert!(shown.contains("vocab_size: 12"), "{shown}");
    assert!(shown.len() < 200, "{shown}");
}

#[test]
fn bos_and_eos_ids_match_the_fixture()
{
    // Guards the fixture itself: the header's ids point at the strings tests expect.
    assert_eq!(fixture::WORDS[BOS_ID as usize], "<s>");
    assert_eq!(fixture::WORDS[EOS_ID as usize], "<|im_end|>");
}
