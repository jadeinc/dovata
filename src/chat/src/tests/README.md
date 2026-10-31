# src/chat/src/tests: the dovata-chat tests

This folder holds the unit tests for dovata-chat. It is a module of the crate, compiled
only under `cargo test`, so the tests can reach private items such as the header reader
and the template environment.

Almost everything here runs offline against fixtures that the tests build themselves, so
`cargo test -p dovata-chat` needs no model file, no network and no GPU. The exceptions
are the tests in `real.rs`, which run only when environment variables name real model
files, and two tests in `hub.rs`, which are `#[ignore]`d because they download from
huggingface.co.

## The fixtures

`fixture.rs` builds a tiny GGUF header and a `tokenizer.json` in the test, and writes
them to a scratch folder when a test needs files on disk. The header is GGUF version 3
with no tensors, built key by key from the few metadata types the fixtures need: strings,
`u32` values, and arrays of strings or of `i32` values. Both model files are made from
the same word list, `WORDS`, so they agree by construction, and a test breaks the
agreement on purpose when it wants a mismatch. The tokenizer is a WordLevel model over
words split on whitespace. That is the smallest real `tokenizer.json` that still has
added special tokens and a post-processor that adds a BOS token.

The scratch folder is made by hand rather than with a crate: the standard library has no
temporary directory type, and one function's worth of code is not worth a dependency.
Each folder is named by process id and a counter, so tests that run at the same time do
not collide, and it is deleted when the test ends.

## Expected values come from Python

The expected strings for `tojson` and `strftime_now` in `template.rs` were produced by
CPython's `json.dumps` and `datetime.strftime`, not written by hand. The goal is to match
what `transformers` prints, so Python is the reference. The weather tool there is
written as a Rust struct, with its keys in the order an OpenAI-style client sends them,
because serde keeps a struct's field order while a default `serde_json::Value` sorts its
keys.

## Tests against real models

`real.rs` runs against real model files when they are on disk. None of these files
belongs in the repository, so with a variable unset its test prints a note and passes.

- `DOVATA_TEST_HEADER` names a GGUF file, or just its first megabytes as a range request
  fetches them. The tests were written against Unsloth's
  `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`.
- `DOVATA_TEST_TOKENIZER` optionally names that model's `tokenizer.json`, from
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`.
- `DOVATA_CHAT_CASES` names the output of `src/chat/jinja2_cases.py`. The comparison runs
  only with `--features preserve_order`, because the cases' tools are in Python's dict
  order, which a `serde_json::Value` keeps only with that feature on.

## Files

- `mod.rs`: the tests of `Chat` itself, such as encoding, rendering, the special tokens
  and bad input, and the list of the other test modules.
- `fixture.rs`: the fixture header, tokenizer and scratch folder described above.
- `hub.rs`: the Hub rule's pure parts, offline: which GGUF in a repo is picked, and where
  the base model's id is read from. It also holds the two network tests, which fetch
  `Qwen/Qwen3-0.6B-GGUF` and `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF`. Run them with
  `cargo test -p dovata-chat -- --ignored`.
- `real.rs`: the tests against real model files and the jinja2 cases.
- `template.rs`: the template environment, `tojson` and `strftime_now`, against what
  `transformers`, jinja2 and CPython produce.
- `vocab.rs`: the vocabulary check, which refuses a mismatched pair and accepts converter
  padding.

## Who uses it

Only `cargo test`. Nothing outside this folder imports it.
