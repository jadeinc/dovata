# dovata-chat-template: prompts and token ids for a model planned with dovata

`dovata-chat-template` turns a conversation into the prompt string a model expects, and
turns that prompt into token ids and back. It reads the chat template from the model's
GGUF header and runs the model's own `tokenizer.json`. It is a Rust package that sits
next to dovata in this repository, and it is optional. It never runs the model: a
program that does gives it the token ids the model produced, and gets text back.

## Why it is separate, and why it is optional

dovata itself now tokenizes: since 2026-11-04 a model's plan carries its tokenizer, and
dovata's [`src/tokenize/`](../tokenize/README.md) runs it, so the Jade compiler never
reads a tokenizer itself. What stays here is the chat template, which is the
programmer's to choose: it builds the prompt string from messages, and dovata's
tokenizer turns that string into ids. This package also tokenizes, for a program that
uses it on its own, and its vocabulary check is dovata's.

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

use dovata_chat_template::{Chat, Message};
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

Special tokens written in the text, such as `<|im_start|>`, become their single ids
whether or not `encode` adds special tokens of its own.

Values a template reads besides the messages go in `render_with`'s `extra` argument:
`tools` for function definitions, `enable_thinking` for Qwen3, `date_string` for
Llama 3, or anything else a model card names. `extra` takes any `Serialize` value
that serializes to a map, or to nothing. It cannot set `messages` or
`add_generation_prompt`, which `render_with` takes as its own arguments: a map that tries
is refused, since one of the two values would be lost without a word.

Key order matters there. A template's `tojson` prints a tool schema's keys in the order
they arrive, as Python does, and models were trained on Python's order. A Rust struct
keeps its field order. A `serde_json::Value` sorts its keys unless serde_json's own
`preserve_order` feature is on. This package's `preserve_order` feature turns it on, but
it is off by default because that feature changes key order for every crate in the
build, dovata's own JSON included. Pass structs, or turn the feature on knowingly.

Llama 3 templates print today's date. Rendered at build time, that bakes the build date
into the program. `Chat::with_time` pins the clock so two builds match.

### The four constructors

- `Chat::from_files` is for a GGUF file and a `tokenizer.json` that are both on disk.
- `Chat::from_hub` fetches both from a Hugging Face repo, and needs the `hf-hub` feature.
- `Chat::from_gguf` takes a GGUF header that is already parsed, for a program that read
  it with dovata already, such as the Jade compiler after preflight, so the file is not
  read twice.
- `Chat::new` takes a template, a tokenizer and the BOS and EOS strings directly. It is
  for a model whose GGUF has no template, or for a template other than the one the model
  shipped. It checks nothing against a vocabulary, because there is none to check
  against, and it fails only when the template does not compile.

The first three check the tokenizer against the GGUF's vocabulary; see below. `bos()`
returns `None` for a model with no BOS token, as Qwen models have none. `eos()` is the
token that ends the model's turn, where a program stops generating.

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

### Reading the header

Everything this package needs from the GGUF is metadata, which sits at the front of the
file ahead of the tensor index. So `src/header.rs` starts with a small read and doubles
it until the metadata parses. For Qwen3-Coder-30B, the metadata, with its 151,936-token
vocabulary and as many merges, ends near 6 MB into an 18 GB file, so the first read of 8
MiB has been enough for every model checked so far. The same loop serves a local file and
a header fetched from the Hub with range requests.

Any parse error before the end of the file and before the budget means: read more. A
buffer cut short can fail as `Truncated`, or as `Implausible` when a long array claims
more elements than the bytes so far can hold. Only the error from the last and largest
read is reported.

A BOS or EOS id that the header does not set is fine, since Qwen ships no BOS. An id
past the end of the vocabulary is not: the header is wrong about something, and the
template would print the wrong marker. The token strings are borrowed from the parsed
header rather than copied, since a vocabulary runs to hundreds of thousands of strings.

The tokenizer is the model's `tokenizer.json`, run by Hugging Face's `tokenizers` crate
with its pure-Rust regex backend. Its default features compile C and C++ code: the
Oniguruma regex library and a suffix array used only for training. This package turns
them off.

### The vocabulary check

The check is dovata's `tokenize::check_vocab`, which dovata also runs when a plan takes a
tokenizer; this package calls it. Because the template and the tokenizer come from two
files, every constructor that has
both checks that they describe the same vocabulary, and refuses a pair that does not. A
`tokenizer.json` from a sibling model gives plausible token ids that are wrong, with no
error anywhere.

It is a refusal and not a warning for this reason. The model's embedding table is indexed
by token id. A tokenizer from a different model, or from a different revision of the same
one, still produces ids, and they still index rows of the table. Nothing fails; the model
just reads the wrong tokens. Qwen2.5 and Qwen3 share most of their vocabulary, for
instance, so a mixed pair is nearly right, which is the worst kind of wrong.

The check in `src/vocab.rs` compares every id, not a sample. It costs a few milliseconds
for a vocabulary of 150,000 tokens, and a sample can miss a pair that differs only in its
added tokens, which is how two revisions of a model most often differ.

The sizes may differ in one direction. `convert_hf_to_gguf.py` sizes the GGUF vocabulary
to the model's embedding table, `vocab_size` in `config.json`, which is often larger than
the tokenizer: Qwen3-Coder's GGUF has 151,936 tokens and its tokenizer 151,665. The
converter fills the gap with `[PAD<id>]` entries of type UNUSED, and those are accepted;
in a GGUF without token types, the name alone marks an entry as padding. A tokenizer with
ids the GGUF lacks is refused, since the model has no row for them. Anything else that
differs is refused too.

The converter also rewrites some added tokens on the way in. One whose `normalized` flag
is off goes through an encode and decode round trip, and a user-defined one has
SentencePiece's `▁` replaced with a space. An added token that matches only after the
same rewrite is accepted. The rewrites are tried only when the plain strings differ, so
the common case costs one string compare. The token type values come from gguf-py's
`TokenType`.

### The `from_hub` rule

`Chat::from_hub(repo, revision)` takes a GGUF repo, the same id dovata's fetch takes.

1. The template and special tokens come from a GGUF header in that repo. Every quant of
   a model carries the same tokenizer metadata, so any GGUF will do. Projector files,
   named `mmproj`, and importance-matrix files, named `imatrix`, are skipped, since
   neither is the language model, and so are split parts after the first, such as
   `Model-00002-of-00003.gguf`, which carry no metadata. A GGUF already in the local
   Hugging Face cache is preferred, since it is the file the program runs and reading it
   costs no network. Otherwise the first one by name is range-fetched, its header only.
2. `tokenizer.json` comes from the same repo if it has one at its root. GGUF repos
   usually do not, so otherwise it comes from the base model: the one the header names
   in `general.base_model.0.repo_url`, as Unsloth's files do, or else the `base_model`
   in the repo's model card. Qwen's own GGUF repos record it only in the card, as a
   string; Unsloth's cards give a list, and the first entry is used. Only a
   `huggingface.co/<org>/<name>` address counts, since anything else is somewhere this
   package cannot download from. The base repo is read at `main`, because `revision`
   names a commit of the GGUF repo, and the base repo has a history of its own.
3. The pair is checked like any other, which is what makes step 2 safe. A base repo whose
   tokenizer has changed since the GGUF was made is refused rather than used.

The other design was to take the template from the base repo's `tokenizer_config.json`
and need no GGUF at all. It was rejected because the GGUF is what the program runs, and
GGUF publishers do fix templates: the Unsloth Qwen3-Coder GGUF carries a corrected one,
so the base repo's would be the older template.

The repo's file list and model card come from one request that `src/hf_hub.rs` makes itself,
because hf-hub's own repo info keeps only the file list and the commit. The header is
fetched with HTTP range requests through `ureq`, the client hf-hub is built on, because
hf-hub has no range API of its own. The token goes along on redirects within the Hub and
is dropped on the hop to the storage CDN, whose address is already signed, which is
hf-hub's own policy.

A Hub token comes from `HF_TOKEN`, or else from the file `huggingface-cli login` writes,
which is where hf-hub looks. Gated base models, such as Meta's Llama, need one.

## Messages

Templates are written against the OpenAI-style message dicts that `transformers` passes
them: `role` and `content`, plus whatever else a turn carries, such as `tool_calls` on an
assistant turn or `reasoning_content` for a thinking model. `Message` in `src/message.rs`
keeps the two fields every template reads as fields, and everything else in a map, so a
new model's extra key needs no change to the package. It serializes to the dict a
template sees as `message`, with every extra key at the same level, and it reads from the
same shape, so an OpenAI-style JSON message list can be read straight into a
`Vec<Message>`.

Templates differ on whether a tool call's `arguments` is a JSON string or an object:
DeepSeek V3 joins it into the prompt as a string, Qwen3-Coder walks it as a map, and
Qwen3 takes either. `Message::tool_calls` passes on whatever it is given, so pass what
the model's template expects.

A message whose extra map holds `role` or `content` is refused at render time. With
`serde(flatten)`, it would serialize a second key of the same name, and which one a
template saw would depend on how the map is stored.

## How it fails

Most errors are refusals when a `Chat` is built rather than failures later. A wrong
tokenizer or a missing template is cheap to report then, and expensive to discover from a
model's odd replies. Each message in `src/error.rs` names the file or GGUF key at fault
and, where there is one, the fix. A template that refuses a conversation through
`raise_exception` comes back as `ChatError::Raised`, apart from `ChatError::Template`,
which means the template itself is broken; `src/template/README.md` says how the two are
told apart.

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
python3 src/chat_template/jinja2_cases.py /tmp/cases.json
DOVATA_CHAT_TEMPLATE_CASES=/tmp/cases.json cargo test -p dovata-chat-template --features preserve_order \
    matches_jinja2_cases -- --nocapture
```

The script fetches the templates from the public repos, so no third-party template is
kept here. With `DOVATA_TEST_HEADER` set to a GGUF header, it adds that file's template,
as the `gguf` cases, which pass no BOS or EOS. Run with no arguments, it prints its
usage text, which it keeps in its `USAGE` constant.

The script's environment mirrors `transformers`' `_compile_jinja_template`: an
`ImmutableSandboxedEnvironment` with `trim_blocks`, `lstrip_blocks` and `loopcontrols`,
the `transformers` `tojson`, `raise_exception` and `strftime_now`. Its render call
mirrors `render_jinja_template`: `tools` and `documents` are always passed, and the
special tokens the tokenizer has are passed as keyword arguments. When a model's
`tokenizer_config.json` holds a list of named templates, the one named `default` is used.
`strftime_now` is pinned to 2024-07-26 09:05:07 UTC, and the test pins the same instant.

How the template environment matches `transformers`, and where it does not, is written
out in `src/template/README.md`.

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

- `hf-hub`, on by default: `Chat::from_hub`. It uses hf-hub 0.4.3 over `ureq`, the same pin
  dovata uses. Its TLS stack, `ring`, compiles C and assembly; `tokenizers` compiles none.
  Built without `hf-hub`, this package compiles no C at all.
- `preserve_order`, off by default: keep the key order of a `serde_json::Value`, as
  described above.

## Files

- `Cargo.toml`: dependencies and features, with the reason for each.
- `src/lib.rs`: `Chat`, the constructors, `render`, `encode` and `decode`.
- `src/message.rs`: `Message`, one turn, which serializes to the dict a template sees.
- `src/header.rs`: reading the template, vocabulary and special tokens from a GGUF
  header, in growing reads.
- `src/vocab.rs`: the check that a `tokenizer.json` and a GGUF agree.
- `src/template/`: the minijinja environment, set up like `transformers`, with `tojson`
  as Python's `json.dumps` and `strftime_now` in UTC. Its own README describes each file.
- `src/hf_hub.rs`: `from_hub`, behind the `hf-hub` feature.
- `src/error.rs`: `ChatError`.
- `src/tests/`: the tests. They build a tiny GGUF header and a WordLevel
  `tokenizer.json` in the test. `real.rs` runs against real model files when
  `DOVATA_TEST_HEADER`, `DOVATA_TEST_TOKENIZER` or `DOVATA_CHAT_TEMPLATE_CASES` is set, and
  `hf_hub.rs` has two `#[ignore]` tests that use the network. Its own README describes each
  file.
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
