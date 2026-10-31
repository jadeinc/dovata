# dovata-chat: prompts and token ids for a model planned with dovata

`dovata-chat` turns a conversation into the prompt string a model expects, and turns that
prompt into token ids and back. It reads the chat template from the model's GGUF header
and runs the model's own `tokenizer.json`. It is a Rust package that sits next to dovata
in this repository, and it is optional.

## Why it is separate, and why it is optional

dovata plans where a model's tensors live in memory. It runs at build time and never runs
the model, so it has no use for a tokenizer, and it does not depend on one. Keeping this
package out of dovata keeps it that way: the Jade compiler can plan a model without
building a tokenizer, a template engine, or anything else here.

A program that runs the model still has to build prompts, and that is where mistakes are
quiet. A prompt with the wrong turn markers still runs; the model only answers worse.
Token ids from the wrong vocabulary also still run, for the same reason. Up to 2.x, the
engine took care of all of this itself, with tokenizing, templates and tool-call parsing
built in. 3.0 dropped that hand-holding. This package is where the help went, for a Jade
user who wants it. A user who would rather read the model card and write the prompt
string by hand can do that, and loses nothing by not installing this.

It is written in Rust rather than Jade because the Jade language is about to change, and
because the two hard parts, the tokenizer and the template engine, already exist as Rust
crates.

## Using it

```rust
use std::path::Path;

use dovata_chat::{Chat, Message};
use serde_json::json;

// From files on disk. Only the GGUF's header is read, never its tensor data.
let chat = Chat::from_files(Path::new("model.gguf"), Path::new("tokenizer.json"))?;
// Or from the Hub, with the same repo id dovata fetches the model from:
// let chat = Chat::from_hub("unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", None)?;

let messages = [Message::system("Be brief."), Message::user("What is a KV cache?")];
let prompt = chat.render(&messages, true)?;
let ids = chat.encode(&prompt, false)?;

// The program runs the model on `ids` until it produces `chat.eos()`, then:
let reply = chat.decode(&generated, true)?;
```

A few things in that example are deliberate:

- `render(.., true)` appends the opening of an assistant turn, so the model's next
  tokens are its reply. Pass `false` to render a finished conversation.
- `encode(.., false)` does not add special tokens. A template that prints `bos_token`
  has already put it in the prompt, and adding it again gives the model two.
- `decode(.., true)` drops special tokens such as turn markers, which is what a person
  reading the reply wants.

Values a template reads besides the messages go in `render_with`'s `extra` argument:
`tools` for function definitions, `enable_thinking` for Qwen3, `date_string` for
Llama 3, or anything else a model card names. `extra` takes any `Serialize` value.

Key order matters there. A template's `tojson` prints a tool schema's keys in the order
they arrive, as Python does, and models were trained on Python's order. A Rust struct
keeps its field order. A `serde_json::Value` sorts its keys unless serde_json's own
`preserve_order` feature is on. This package's `preserve_order` feature turns it on, but
it is off by default because that feature changes key order for every crate in the
build, dovata's own JSON included. Pass structs, or turn the feature on knowingly.

Llama 3 templates print today's date. Rendered at build time, that bakes the build date
into the program. `Chat::with_time` pins the clock so two builds match.

### How a Jade user reaches it

The Jade compiler and dovata are both Rust, so the direct use is from the Rust side of a
build. For example, a build step can render a fixed system prompt and embed its token
ids in the program. How a Jade program calls this package at run time depends on how the
package is distributed, which is still open; see the last section.

## Where each piece comes from

The chat template is the GGUF's `tokenizer.chat_template`. The BOS and EOS strings are
`tokenizer.ggml.bos_token_id` and `tokenizer.ggml.eos_token_id`, looked up in
`tokenizer.ggml.tokens`. The GGUF is the file the program runs, so its template is the
one that goes with the weights. Some GGUF publishers fix the template on the way: the
Unsloth Qwen3-Coder file carries a corrected one. Only the header is read. The reader
starts with 8 MiB and doubles until the metadata parses, up to dovata's
`HEADER_BUDGET` of 64 MiB, so a header fetched with an HTTP range request works as well
as the whole file.

The tokenizer is the model's `tokenizer.json`, run by Hugging Face's `tokenizers` crate
with its pure-Rust regex backend. Its default features compile C and C++ code: the
Oniguruma regex library and a suffix array used only for training. This package turns
them off.

Because the template and the tokenizer come from two files, every constructor that has
both checks that they describe the same vocabulary, and refuses a pair that does not. A
`tokenizer.json` from a sibling model gives plausible token ids that are wrong, with no
error anywhere. The check compares every id, not a sample. The GGUF may be longer than
the tokenizer, because `convert_hf_to_gguf.py` pads the vocabulary to the size of the
embedding table: Qwen3-Coder's GGUF has 151,936 tokens and its tokenizer 151,665. Those
padding entries are accepted, and anything else that differs is not.

### The `from_hub` rule

`Chat::from_hub(repo, revision)` takes a GGUF repo, the same id dovata's fetch takes.

1. The template and special tokens come from a GGUF header in that repo. Every quant of
   a model carries the same tokenizer metadata, so any GGUF will do. Projector files,
   importance-matrix files, and split parts after the first are skipped. A GGUF already
   in the local Hugging Face cache is preferred, since it is the file the program runs.
   Otherwise the first one by name is range-fetched, its header only.
2. `tokenizer.json` comes from the same repo if it has one. GGUF repos usually do not, so
   otherwise it comes from the base model: the one the header names in
   `general.base_model.0.repo_url`, or else the `base_model` in the repo's model card.
   Qwen's own GGUF repos record it only in the card. The base repo is read at `main`.
3. The pair is checked like any other, which is what makes step 2 safe.

A Hub token comes from `HF_TOKEN`, or from `huggingface-cli login`. Gated base models,
such as Meta's Llama, need one.

## Templates it was checked against

Templates are Jinja written for Python's `jinja2`, the way Hugging Face's `transformers`
renders them. This package uses `minijinja`, set up the way `transformers` sets up
`jinja2` in `_compile_jinja_template`: `trim_blocks` and `lstrip_blocks` on, loop
controls, a `tojson` that prints what Python's `json.dumps` prints and takes its keyword
arguments, `raise_exception`, `strftime_now`, and Python's string and dict methods
through minijinja-contrib's `pycompat`. `tools` and `documents` are always defined, as
none when not given, which is also what `transformers` does.

One real difference between the engines turned up and is corrected here. In jinja2,
`none is iterable` is false. In minijinja it is true, and Qwen3-Coder's template then
fails on `none | length` whenever no tools are passed.

These templates render byte for byte the same as jinja2 3.1.6 configured like
`transformers`, across 50 conversations covering system prompts, multi-turn chats, tool
definitions, tool calls and results, thinking content, Unicode, and the templates' own
refusals:

| Model | Template source |
|---|---|
| Qwen3 | `Qwen/Qwen3-0.6B` |
| Qwen3-Coder-30B-A3B-Instruct | the GGUF header of `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` |
| Llama 3.2 Instruct, the Llama 3.1 and 3.3 template family | `unsloth/Llama-3.2-1B-Instruct` |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` |

To rerun the comparison, for example after bumping minijinja:

```sh
pip install jinja2
python3 src/chat/jinja2_cases.py /tmp/cases.json
DOVATA_CHAT_CASES=/tmp/cases.json cargo test -p dovata-chat --features preserve_order \
    matches_jinja2_cases -- --nocapture
```

The script fetches the templates from the public repos, so no third-party template is
kept here. With `DOVATA_TEST_HEADER` set to a GGUF header, it adds that file's template.

## What it does not do

- It does not parse tool calls out of a reply, stream tokens, or sample. Those belong to
  whatever runs the model.
- It does not run a model, and never will; dovata does not either.
- Message content is text or null. Content made of parts, such as images, is not
  supported.
- `strftime_now` formats in UTC. `transformers` uses local time, so a prompt rendered
  near midnight can carry a different date.
- Templates that use `{% generation %}`, which marks assistant tokens for training, do
  not compile.
- The token ids are the reference tokenizer's. llama.cpp tokenizes from the GGUF with
  its own implementation, which aims to match and can disagree on rare inputs.

## Features

- `hub`, on by default: `Chat::from_hub`. It uses hf-hub 0.4.3 over `ureq`, the same pin
  dovata uses. Its TLS stack, `ring`, compiles C and assembly; `tokenizers` compiles none.
  Built without `hub`, this package compiles no C at all.
- `preserve_order`, off by default: keep the key order of a `serde_json::Value`, as
  described above.

## Files

- `Cargo.toml`: dependencies and features, with the reason for each.
- `src/lib.rs`: `Chat`, the constructors, `render`, `encode` and `decode`.
- `src/message.rs`: `Message`, one turn, which serializes to the dict a template sees.
- `src/header.rs`: reading the template, vocabulary and special tokens from a GGUF
  header, in growing reads.
- `src/vocab.rs`: the check that a `tokenizer.json` and a GGUF agree.
- `src/template/mod.rs`: the minijinja environment, set up like `transformers`.
- `src/template/pyjson.rs`: `tojson` as Python's `json.dumps`.
- `src/template/clock.rs`: `strftime_now`'s formatting, in UTC.
- `src/hub.rs`: `from_hub`, behind the `hub` feature.
- `src/error.rs`: `ChatError`.
- `src/tests/`: the tests. They build a tiny GGUF header and a WordLevel
  `tokenizer.json` in the test. `real.rs` runs against real model files when
  `DOVATA_TEST_HEADER`, `DOVATA_TEST_TOKENIZER` or `DOVATA_CHAT_CASES` is set, and
  `hub.rs` has two `#[ignore]` tests that use the network.
- `jinja2_cases.py`: writes the cases for the jinja2 comparison.

## What it depends on, and what depends on it

It depends on dovata, with default features off, for the GGUF reader in `src/gguf/` and
for `HEADER_BUDGET` in `src/load/preflight.rs`. With default features off, dovata builds
no llama.cpp and no network code. Nothing in dovata depends on this package, and nothing
should.

## Open questions

- Distribution. How a Jade user gets this package is not decided. It could be a crate
  on crates.io or a git dependency for the Rust side of a build, a prebuilt library with
  a C interface that a Jade program loads, or a Jade package wrapping it once the
  language settles.
- Whether `strftime_now` should use local time, as `transformers` does, at the cost of a
  time-zone dependency and a build that depends on the build machine's zone.
- Whether `from_hub` should read the base model's tokenizer at a pinned revision rather
  than `main`. The vocabulary check catches a tokenizer that has drifted, but only by
  refusing it.
