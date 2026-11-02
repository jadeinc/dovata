# cli: the `dovata` command line

`dovata` is the package's command-line program. It has two groups of commands:
`dovata machine` reports the current machine's memory and GPUs through
[`src/machine/`](../machine/README.md), and `dovata plan` places a model's tensors in
memory through [`src/memplan/`](../memplan/README.md). `gguf-inspect` can join it later as
`dovata inspect`, so dovata has one program with subcommands instead of one program per
job.

## `dovata machine`

```text
dovata machine info [--json]                    memory capacity, page size, GPU memory limits
dovata machine usage [--json] [--process N]     memory in use right now
dovata machine watch [--every S] [--process N]  memory in use, refreshed in place until Ctrl-C
dovata machine target [--json] [--cpu]          the memory and limits the planner will use here
```

On a Mac Mini with an M4:

```text
$ dovata machine info
Apple M4, macOS 26.5.2
  memory      24.0 GiB, 16 KiB pages
  Metal GPU   17.8 GiB working set, 13.3 GiB largest buffer, unified memory
  planner     apple target, 17.8 GiB, buffers up to 13.3 GiB

$ dovata machine usage --process zsh
  memory     18.8 GiB used of 24.0 GiB, 5.2 GiB available
             immovable 2.2 GiB, compressed 8.2 GiB, cached 4.3 GiB
  swap       0 B used of 0 B
  pressure   55% free
  process    92385 /bin/zsh, 2.4 MiB resident
```

`watch` shows the same reading as `usage`, cleared and redrawn every `--every` seconds,
one second by default. On a machine with NVIDIA GPUs, each GPU's line shows its memory,
load, temperature and clocks. When its output is a file or a pipe instead of a terminal,
`watch` writes one line per reading, with the seconds since it started, so a log of a
long run can be kept with `dovata machine watch > memory.log`. It stops when Ctrl-C
ends it or when its output is closed, as when it is piped into `head`.

`--process NAME` matches running programs whose command, or the last part of its path,
is exactly `NAME`, and shows each one's resident memory, the bytes it holds in physical
memory.

`info` and `target` answer different questions. `info` describes the hardware: all of
its memory, its page size and every GPU. `target` describes the budget the planner gets
from that hardware, the `target::Target` that `Machine::target` builds: Apple on a Mac
with Metal, CUDA for the first NVIDIA GPU, and CPU otherwise. On a Mac, the budget is
Metal's working set, not all of memory:

```text
$ dovata machine target
apple target
  memory          17.8 GiB the plan may use
  largest buffer  13.3 GiB
  page size       16 KiB
```

`--json` prints the same target in bytes, the form a program reads. `--cpu` asks for
the CPU target even when there is a GPU.

## `dovata plan`

```text
dovata plan <model> [--quant TYPE] [--allow-requantize] [--layout FILE]
                    [--target apple|cpu|cuda] [--memory SIZE] [--max-buffer SIZE] [--page SIZE]
                    [--ctx N] [--tokens N] [--outputs N] [--kv TYPE] [--kv-k TYPE] [--kv-v TYPE]
                    [--json]
```

The model is a GGUF file, or a Hugging Face link to a repo or to one file in it. The
machine is this one, as `dovata machine target` shows it, unless `--target` and
`--memory` describe another. The step defaults to llama.cpp's: a KV cache of 4096
tokens in f16, 512 tokens per step, and logits for one of them.

```text
$ dovata plan Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
Qwen3-Coder-30B-A3B-Instruct, qwen3moe, Q4_K_M
  machine    apple target, 17.8 GiB, buffers up to 13.3 GiB
  step       512 tokens, 1 with logits, a KV cache of 4096 tokens in f16 for K and f16 for V
  graph      3030 ops over 48 layers, 3711 tensors
  regions
    0   mmap       13.3 GiB  weights, part 1 from byte 0
    1   mmap        4.0 GiB  weights, part 1 from byte 14284275712
    2   unified   384.0 MiB  KV cache
    3   unified    72.0 MiB  step inputs, scratch and outputs
  total      17.3 GiB mapped from the model file, 456.0 MiB allocated
  table hash e9f5585e748e20f13fb48c4000287fb96556a372133e02b9e0d6708c552294b6
Add --json for every tensor's region and offset.
```

`--json` prints the whole plan, the form the Jade compiler reads.

### Choosing a quantization

`--quant` names the quantization to plan, and dovata gets it the cheapest way it can:

- A Hub repo that offers that type: dovata downloads only that file.
- A Hub repo that does not: dovata downloads its unquantized model and quantizes it
  with llama.cpp's `llama-quantize`, into a `dovata` directory inside the repo's
  folder in the Hugging Face cache. A later run reuses the copy.
- A local file of another type: dovata quantizes it next to the file, and reuses that
  copy later too.

A model that is already quantized is not quantized a second time unless
`--allow-requantize` says so, since that loses more quality than quantizing once.

Without `--quant`, a Hub repo that offers several types stops the command before
anything is downloaded, and lists them:

```text
$ dovata plan https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF
dovata: https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF offers 7 models. Choose one with --quant, or link to its file:
  F16          SmolLM2-135M-Instruct-F16.gguf
  Q8_0         SmolLM2-135M-Instruct-Q8_0.gguf
  ...
```

The unquantized model of a 30B repo is 60 GB, so dovata asks instead of guessing.

## Exit codes

A bad or incomplete command line, including a repo that needs `--quant`, exits with 2.
A command that fails, such as a model that does not fit or a machine that cannot be
read, exits with 1.

## Files

- `main.rs` dispatches to the two groups and holds what they share: the `Fail` type
  that picks the exit code, `out`, which writes to standard output and stops quietly
  when the output is closed, and the size formatting and parsing.
- `machine.rs` is `dovata machine`: argument parsing, the four commands, and their text.
- `plan.rs` is `dovata plan`: argument parsing, getting and quantizing the model,
  building the target, and the plan's summary.

Each file's tests cover its parser and its text.

## Building

The binary needs the `machine` feature, which is on by default:

```sh
cargo run --bin dovata -- machine info
cargo run --bin dovata -- plan /path/to/model.gguf
```

## Who uses it

People, at a terminal. Programs such as the Jade compiler call `dovata::machine` and
`dovata::memplan` directly.
