# tokenize: the model's tokenizer, as part of its plan

`tokenize` gives a plan its tokenizer, and runs it. A model's plan carries everything a
program needs to run the model: where each tensor lives, the step's ops, and now the
tokenizer that turns text into the token ids the step takes and the ids it produces back
into text. The Jade compiler never reads a GGUF's tokenizer or a `tokenizer.json`
itself. It reads the plan, and calls this capability to tokenize.

```rust
use dovata::tokenize::{self, Tokenizer};

// When planning: find the model's tokenizer.json, check it against the GGUF, and plan
// with it. The plan's table hash covers it.
let path = tokenize::find(&parts[0], model.meta(), origin.as_ref(), &fetch)?;
let t = tokenize::describe(&model, std::fs::read_to_string(path)?)?;
let plan = dovata::memplan::plan(&model, &layout, &step, &target, Some(&t))?;

// When running: everything comes from the plan.
let tok = Tokenizer::from_plan(plan.tokenizer.as_ref().unwrap())?;
let ids = tok.encode(&prompt, false)?;
let mut out = tok.stream(true);
// the program's kernels produce the next id
if tok.is_stop(id) { /* the model's turn is over */ }
print!("{}", out.push(id)?);            // whole characters only
```

A program that maps a model from the dovata process gets the same tokenizer with
`Process::tokenizer(name, table_hash)`, which refuses a plan other than the one the
program was compiled against.

## Why it exists, and why here

Tokenizing runs on the CPU, before and after each step, and is not one of the GPU ops
the compiler turns into kernels. It still belongs to running the model: the wrong
tokenizer raises no error, and the model just answers worse. So it sits with the rest of
what a model needs, in dovata, and the plan is the one place the compiler reads it from.

It is its own capability, not part of `memplan/`, because the planner places memory and
the tokenizer runs text. `memplan/` only carries the tokenizer's description and hashes
it with the rest of the plan. It is not `src/chat_template/`, the optional
`dovata-chat-template` package, because chat templates stay the programmer's to choose:
a template builds a prompt string from messages, and any program can do that its own way.
That package uses this capability's vocabulary check rather than keeping a copy.

## What the plan carries

`TokenizerPlan`, the plan's `tokenizer` key:

| field | what it is |
|---|---|
| `kind` | the GGUF's `tokenizer.ggml.model`: `gpt2` for byte-level BPE, as Qwen, Llama 3 and DeepSeek use, or `llama` for SentencePiece |
| `tokens` | the tokens `tokenizer.json` defines: one past its highest id |
| `embedding_rows` | the rows of `token_embd.weight`; every id must be below it. Converters pad the table, so it can be larger than `tokens`: Qwen3 has 151,669 tokens and 151,936 rows |
| `bos`, `eos` | the GGUF's BOS and EOS ids. Qwen has no BOS |
| `stop` | the ids that end the model's turn: EOS, and the end-of-turn and end-of-message ids the GGUF names in `tokenizer.ggml.eot_token_id` and `eom_token_id` |
| `sha256` | the SHA-256 of `definition` |
| `definition` | the model's `tokenizer.json`, exactly as the file has it |

The definition is kept as text, not as parsed JSON. The plan's table hash is a SHA-256
of the plan's JSON, and a parsed `serde_json::Value` orders its keys one way or another
depending on a cargo feature any crate in the build can turn on. Text has the same bytes
in every build. For Qwen3 the definition is 10.6 MB, so a plan with its tokenizer is that
much larger.

`stop` holds only what the GGUF names. A model card can name more, such as Qwen3's
`<|endoftext|>`, which ends raw completions; a program that wants it adds it. Lists of
model-specific stop strings, the way llama.cpp keeps them, are model specifics, which
belong in data rather than in dovata.

## Where `tokenizer.json` comes from

The GGUF has the vocabulary and the merges, but running them takes the pre-tokenizer,
normalizer and decoder `tokenizer.json` describes, so dovata uses that file. `find`
looks in three places, in order:

1. `tokenizer.json` beside the model file;
2. the model's own Hub repo, when the model came from the Hub;
3. the base model's repo: the one the GGUF header names in
   `general.base_model.0.repo_url`, as Unsloth's files do, or else the `base_model` in
   the model repo's card. GGUF repos rarely carry a `tokenizer.json`, so this is the
   usual source.

The Hugging Face cache is checked before the network, so a tokenizer fetched once is
found offline. `--tokenizer FILE` on `dovata plan` and `dovata add`, or
`AddOptions::tokenizer`, names one instead.

What is found is checked, not trusted: `describe` refuses a `tokenizer.json` whose
vocabulary differs from the GGUF's.

### The vocabulary check

The model's embedding table is indexed by token id. A tokenizer from a different model,
or from a different revision of the same one, still produces ids, and they still index
rows of the table. Nothing fails; the model just reads the wrong tokens. Qwen2.5 and
Qwen3 share most of their vocabulary, so a mixed pair is nearly right, which is the worst
kind of wrong. A base repo whose tokenizer changed after the GGUF was made is the same
case.

`check_vocab` in `vocab.rs` compares every id, not a sample. The sizes may differ in one
direction: the converter pads the GGUF's vocabulary to the embedding table with
`[PAD<id>]` entries of type UNUSED, and those are accepted. The converter also rewrites
some added tokens, and an added token that matches only after the same rewrite is
accepted. Anything else that differs is refused, with the first ids that disagree.

## Running it

`Tokenizer` runs the definition with Hugging Face's `tokenizers` crate, built with its
pure-Rust regex backend and without its C and C++ code.

- `encode(text, add_special)`: `add_special` adds what `tokenizer.json`'s post-processor
  puts around a sequence, such as a BOS. Pass `false` for a prompt whose chat template
  already wrote it. Special tokens written in the text become their single ids either
  way. An id at or past `embedding_rows` is refused.
- `decode(ids, skip_special)`.
- `stream(skip_special)` and `Stream::push(id)`: byte-level BPE splits a character such
  as 🚀 over several ids, and the stream holds the pieces back until the character is
  whole, so every piece it gives is valid text.
- `is_stop(id)`, `bos()`, `eos()`, `stop()`.

## Files

| File | What it does |
|---|---|
| `mod.rs` | `TokenizerPlan`, `Mismatch`, `VocabMismatch` and `TokenizeError`; reading the GGUF's vocabulary, token types and special ids; the hash. With the feature off, `describe` returns `TokenizeError::NotBuilt`. |
| `find.rs` | `find`: the three places a `tokenizer.json` comes from. |
| `vocab.rs` | `check_vocab`, behind the `tokenize` feature. |
| `engine.rs` | `describe`, `Tokenizer` and `Stream`, behind the `tokenize` feature. |
| `tests.rs` | A small GGUF and a matching `tokenizer.json` written in the test: the description, a mismatched vocabulary and a bad id refused, encoding, decoding and streaming, a tampered definition refused, and the file beside the model found first. Qwen3 0.6B's real files, when they are in the Hugging Face cache: byte-level BPE, and a stream that never splits a character. |

## Who uses it

`memplan/` carries `TokenizerPlan` in the plan and hashes it. `src/cli/plan.rs` finds and
describes the tokenizer for `dovata plan` and `dovata add`. `host/` checks each added
model's tokenizer into its plan and builds a `Tokenizer` for a program from it.
`src/chat_template/` uses `check_vocab`. It depends on `gguf`, `load` and `hf_hub`, and on
the `tokenizers` crate.
