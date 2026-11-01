# cli: the `dovata` command line

`dovata` is the package's command-line program. Today it holds the `machine` commands,
which report the current machine's memory and GPUs through
[`src/machine/`](../machine/README.md). The planned plan command and `gguf-inspect` can
join it later as `dovata plan` and `dovata inspect`, so dovata has one program with
subcommands instead of one program per job.

## Commands

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
long run can be kept with `dovata machine watch > memory.log`. It stops when Ctrl-C ends
it or when its output is closed, as when it is piped into `head`.

`--process NAME` matches running programs whose command, or the last part of its path,
is exactly `NAME`, and shows each one's resident memory, the bytes it holds in physical
memory.

`info` and `target` answer different questions. `info` describes the hardware: all of
its memory, its page size and every GPU. `target` describes the budget the planner gets
from that hardware, the `plan::Target` that `Machine::target` builds: Apple on a Mac
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

A bad command line prints the help and exits with 2. A failure to read the machine
exits with 1.

## Files

- `main.rs` is the whole program: argument parsing, the four commands, and the text
  formatting. Its tests cover the parser, the size formatting and the usage text.

## Building

The binary needs the `machine` feature, which is on by default:

```sh
cargo run --bin dovata -- machine info
```

## Who uses it

People, at a terminal. Programs such as the Jade compiler call `dovata::machine`
directly.
