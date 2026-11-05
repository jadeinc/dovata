//! Author: Joe Ricks
//! File: src/tokenize/tests.rs
//! Description: Tests describing a tokenizer for a plan, finding a `tokenizer.json`, and
//! running it, on a small GGUF and tokenizer written here, and on Qwen3 0.6B when its
//! files are in the Hugging Face cache.

use std::path::PathBuf;

use serde_json::json;

use super::*;
use crate::gguf::write::Writer;
use crate::gguf::Value as Kv;
use crate::hf_hub::FetchOptions;
use crate::load::Model;

/// The fixture's vocabulary, in id order. Ids 0 to 3 are special.
const WORDS: &[&str] = &["<unk>", "<|im_start|>", "<|im_end|>", "<s>", "hello", "world", "user", "Hi"];
/// The BOS token's id.
const BOS: u32 = 3;
/// The EOS token's id, `<|im_end|>`.
const EOS: u32 = 2;
/// The end-of-turn token's id the GGUF names apart from EOS, `<unk>` here only so it is
/// a different id.
const EOT: u32 = 0;
/// The embedding table's rows: the vocabulary padded to 12, as converters pad it.
const ROWS: usize = 12;

/// Purpose: This function builds the fixture's `tokenizer.json`: a word-level tokenizer
/// over [`WORDS`] that puts BOS before a sequence.
///
/// Inputs:
/// - `words`: the vocabulary, in id order.
///
/// Outputs: the JSON text.
fn tokenizer_json(
    words: &[&str],
) -> String
{
    let added: Vec<_> = words[..4]
        .iter()
        .enumerate()
        .map(|(id, w)| json!({"id": id, "content": w, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}))
        .collect();
    let vocab: serde_json::Map<String, serde_json::Value> = words.iter().enumerate().map(|(id, w)| ((*w).to_owned(), json!(id))).collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": added, "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{ "SpecialToken": { "id": "<s>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } }],
            "pair": [{ "SpecialToken": { "id": "<s>", "type_id": 0 } }, { "Sequence": { "id": "A", "type_id": 0 } }, { "Sequence": { "id": "B", "type_id": 1 } }],
            "special_tokens": { "<s>": { "id": "<s>", "ids": [BOS], "tokens": ["<s>"] } }
        },
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<unk>" }
    })
    .to_string()
}

/// Purpose: This function builds the fixture model: a llama header whose vocabulary is
/// [`WORDS`] padded to [`ROWS`], and a token embedding table with [`ROWS`] rows.
///
/// Inputs:
/// - `eos`: the id to write as `tokenizer.ggml.eos_token_id`.
///
/// Outputs: the model, opened from the header bytes.
fn model(
    eos: u32,
) -> Model
{
    let s = |v: &str| Kv::String(v.into());
    let mut tokens: Vec<Kv> = WORDS.iter().map(|w| s(w)).collect();
    let mut types: Vec<Kv> = (0..WORDS.len()).map(|i| Kv::I32(if i < 4 { 3 } else { 1 })).collect();
    for id in WORDS.len()..ROWS {
        tokens.push(s(&format!("[PAD{id}]")));
        types.push(Kv::I32(5));
    }
    let mut w = Writer::new();
    w.kv("general.architecture", s("llama"))
        .kv("llama.block_count", Kv::U32(1))
        .kv("llama.context_length", Kv::U32(64))
        .kv("llama.embedding_length", Kv::U32(8))
        .kv("llama.feed_forward_length", Kv::U32(16))
        .kv("llama.attention.head_count", Kv::U32(1))
        .kv("llama.attention.head_count_kv", Kv::U32(1))
        .kv("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-5))
        .kv("tokenizer.ggml.model", s("gpt2"))
        .kv("tokenizer.ggml.tokens", Kv::Array(tokens))
        .kv("tokenizer.ggml.token_type", Kv::Array(types))
        .kv("tokenizer.ggml.bos_token_id", Kv::U32(BOS))
        .kv("tokenizer.ggml.eos_token_id", Kv::U32(eos))
        .kv("tokenizer.ggml.eot_token_id", Kv::U32(EOT));
    w.f32_tensor("token_embd.weight", &[8, ROWS as u64], &vec![0.0; 8 * ROWS]);
    let bytes = w.to_bytes().unwrap();
    Model::from_headers(&[(PathBuf::from("fixture.gguf"), bytes)]).unwrap()
}

#[test]
fn a_tokenizer_is_described_from_the_gguf_and_its_tokenizer_json()
{
    let text = tokenizer_json(WORDS);
    let t = describe(&model(EOS), text.clone()).unwrap();
    assert_eq!(t.kind, "gpt2");
    assert_eq!((t.tokens, t.embedding_rows), (WORDS.len() as u32, ROWS as u32));
    assert_eq!((t.bos, t.eos), (Some(BOS), Some(EOS)));
    assert_eq!(t.stop, [EOS, EOT], "EOS first, then the end-of-turn token");
    assert_eq!(t.definition, text, "kept exactly as the file has it");
    assert_eq!(t.sha256, sha256_hex(&text));

    // It travels as JSON, as the plan and the process send it.
    let back: TokenizerPlan = serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
}

#[test]
fn a_tokenizer_from_another_model_or_a_bad_id_is_refused()
{
    let mut swapped = WORDS.to_vec();
    swapped.swap(4, 5);
    let e = describe(&model(EOS), tokenizer_json(&swapped)).unwrap_err();
    assert!(matches!(&e, TokenizeError::Vocab(v) if v.count == 2 && v.examples[0].id == 4), "{e:?}");
    assert!(e.to_string().contains("id 4: GGUF \"hello\", tokenizer.json \"world\""), "{e}");

    let e = describe(&model(99), tokenizer_json(WORDS)).unwrap_err();
    assert_eq!(e, TokenizeError::BadTokenId { key: "tokenizer.ggml.eos_token_id", id: 99, tokens: ROWS });
    assert!(matches!(describe(&model(EOS), "{".into()), Err(TokenizeError::Unreadable(_))));
}

#[test]
fn a_plan_s_tokenizer_encodes_decodes_and_streams()
{
    let plan = describe(&model(EOS), tokenizer_json(WORDS)).unwrap();
    let t = Tokenizer::from_plan(&plan).unwrap();
    assert_eq!(t.encode("hello world", false).unwrap(), [4, 5]);
    assert_eq!(t.encode("hello world", true).unwrap(), [BOS, 4, 5], "tokenizer.json's post-processor adds BOS");
    assert_eq!(t.encode("<|im_start|>user Hi<|im_end|>", false).unwrap(), [1, 6, 7, 2], "special tokens in the text are single ids");
    assert_eq!(t.decode(&[4, 5], false).unwrap(), "hello world");
    assert_eq!(t.decode(&[BOS, 4, 2], true).unwrap(), "hello", "special tokens left out");
    assert!(t.is_stop(EOS) && t.is_stop(EOT) && !t.is_stop(4));
    assert_eq!((t.bos(), t.eos(), t.stop()), (Some(BOS), Some(EOS), &[EOS, EOT][..]));

    let mut s = t.stream(true);
    let streamed: String = [4u32, 5, 4].iter().map(|id| s.push(*id).unwrap()).collect();
    assert_eq!(streamed, t.decode(&[4, 5, 4], true).unwrap());

    let tampered = TokenizerPlan { definition: plan.definition.replace("hello", "howdy"), ..plan };
    assert!(matches!(Tokenizer::from_plan(&tampered), Err(TokenizeError::Unreadable(m)) if m.contains("sha256")));
}

#[test]
fn a_tokenizer_json_beside_the_model_is_found_first()
{
    let dir = std::env::temp_dir().join(format!("dovata-tok-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let gguf_path = dir.join("m.gguf");
    let m = model(EOS);
    let e = find(&gguf_path, m.meta(), None, &FetchOptions::default()).unwrap_err();
    assert!(matches!(&e, TokenizeError::NotFound(msg) if msg.contains("--tokenizer FILE") && msg.contains("base model")), "{e:?}");
    std::fs::write(dir.join(TOKENIZER_JSON), tokenizer_json(WORDS)).unwrap();
    assert_eq!(find(&gguf_path, m.meta(), None, &FetchOptions::default()).unwrap(), dir.join(TOKENIZER_JSON));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Purpose: This function finds a file in the Hugging Face cache's newest snapshot of a
/// repo.
///
/// Inputs:
/// - `repo`: the repo's cache folder name, such as `models--Qwen--Qwen3-0.6B`.
/// - `file`: the file's name.
///
/// Outputs: the path, or `None` when the cache does not have it.
fn cached(
    repo: &str,
    file: &str,
) -> Option<PathBuf>
{
    let snapshots = PathBuf::from(std::env::var_os("HOME")?).join(".cache/huggingface/hub").join(repo).join("snapshots");
    std::fs::read_dir(snapshots).ok()?.flatten().map(|e| e.path().join(file)).find(|p| p.is_file())
}

#[test]
fn qwen3_s_real_tokenizer_streams_whole_characters()
{
    // Runs only when both files are in the local Hugging Face cache, as they are after
    // `dovata add https://huggingface.co/unsloth/Qwen3-0.6B-GGUF --quant Q4_K_M`.
    let (Some(gguf), Some(json)) = (cached("models--unsloth--Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"), cached("models--Qwen--Qwen3-0.6B", TOKENIZER_JSON)) else {
        eprintln!("skipped: Qwen3 0.6B's GGUF or tokenizer.json is not in the Hugging Face cache");
        return;
    };
    let model = Model::open(&gguf).unwrap();
    let plan = describe(&model, std::fs::read_to_string(json).unwrap()).unwrap();
    assert_eq!((plan.kind.as_str(), plan.bos, plan.eos), ("gpt2", None, Some(151645)));
    assert!(plan.tokens <= plan.embedding_rows && plan.embedding_rows == 151936, "{} {}", plan.tokens, plan.embedding_rows);
    let t = Tokenizer::from_plan(&plan).unwrap();
    let text = "<|im_start|>user\nKV caches 缓存 🚀<|im_end|>";
    let ids = t.encode(text, false).unwrap();
    assert_eq!(ids[0], 151644, "<|im_start|> is one id");
    assert_eq!(t.decode(&ids, false).unwrap(), text);
    // Byte-level BPE splits a character such as 🚀 over several ids. The stream holds
    // back the pieces until the character is whole, so every piece it gives is valid.
    let mut s = t.stream(false);
    let pieces: Vec<String> = ids.iter().map(|id| s.push(*id).unwrap()).collect();
    assert_eq!(pieces.concat(), text);
    assert!(pieces.iter().all(|p| !p.contains('\u{fffd}')), "{pieces:?}");
}
