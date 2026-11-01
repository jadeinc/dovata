# cli: the `dovata` command line

`dovata` is the package's command-line program. Today it holds the `hw` commands, which
report the current machine's memory and GPUs through [`src/hw/`](../hw/README.md). The
planned plan command and `gguf-inspect` can join it later as `dovata plan` and
`dovata inspect`, so dovata has one program with subcommands instead of one program per
job.

## Commands

```text
dovata hw info [--json]                  memory capacity, page size, GPU memory limits
dovata hw usage [--json] [--process N]   memory in use right now
dovata hw watch [--every S] [--process N]  memory in use, refreshed in place until Ctrl-C
dovata hw target [--cpu]                 this machine as the planner's target, as JSON
```

On a Mac Mini with an M4:

```text
$ dovata hw info
Apple M4, macOS 26.5.2
  memory      24.0 GiB, 16 KiB pages
  Metal GPU   17.8 GiB working set, 13.3 GiB largest buffer, unified memory
  planner     apple target, 17.8 GiB, buffers up to 13.3 GiB

$ dovata hw usage --process zsh
  memory     18.8 GiB used of 24.0 GiB, 5.2 GiB available
             wired 2.2 GiB, compressed 8.2 GiB, cached 4.3 GiB
  swap       0 B used of 0 B
  pressure   55% free
  process    92385 /bin/zsh, 2.4 MiB resident
```

`watch` shows the same reading as `usage`, cleared and redrawn every `--every` seconds,
one second by default. On a machine with NVIDIA GPUs, each GPU's line shows its memory,
load, temperature and clocks. When its output is a file or a pipe instead of a terminal,
`watch` writes one line per reading, with the seconds since it started, so a log of a
long run can be kept with `dovata hw watch > memory.log`. It stops when Ctrl-C ends it
or when its output is closed, as when it is piped into `head`.

`--process NAME` matches running programs whose command, or the last part of its path,
is exactly `NAME`, and shows each one's resident memory, the bytes it holds in physical
memory.

`target` prints the `plan::Target` that `Machine::target` builds: Apple on a Mac with
Metal, CUDA for the first NVIDIA GPU, and CPU otherwise. `--cpu` asks for the CPU
target even when there is a GPU.

A bad command line prints the help and exits with 2. A failure to read the machine
exits with 1.

## Files

- `main.rs` is the whole program: argument parsing, the four commands, and the text
  formatting. Its tests cover the parser, the size formatting and the usage text.

## Building

The binary needs the `hw` feature, which is on by default:

```sh
cargo run --bin dovata -- hw info
```

## Who uses it

People, at a terminal. Programs such as the Jade compiler call `dovata::hw` directly.
