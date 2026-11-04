# host: dovata as a process that holds a model in memory

`host` runs dovata as a process of its own. The process loads one model into memory and
keeps it there until it is stopped. A compiled Jade program maps that same memory into
itself and runs its own kernels on it. dovata still never runs a model's math: the
process holds memory, reports on it and hands it out, and the program does the compute.

```rust
use dovata::host::{self, StartOptions};

// Start: download the model if needed, plan it for this machine, load it.
let model = host::start("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF", &StartOptions::default())?;

// Map: every region of the plan, where the program's compiled offsets expect it. The
// process refuses a program whose offsets came from a different plan.
let mem = model.map(PLAN_TABLE_HASH)?;
let weights = mem.region(0).unwrap().as_ptr();   // bind as a no-copy MTLBuffer, say

// Watch and stop.
let status = model.status()?;                    // each region, and how much is resident
model.kill()?;
```

The command line does the same: `dovata start`, `dovata status` and `dovata kill`, in
[`src/cli/`](../cli/README.md). A process started from one is seen and stopped by the
other, since both find it by name.

## Why it exists

dovata 3.0 began as a build-time crate only: the Jade compiler planned a model with it,
and the compiled program set up the plan's memory itself at startup. That made each
program load the model on its own, and left nothing on the machine that owned the
model's memory or could say where it was. Since 2026-11-03, dovata is also a process
that hosts a model, started and stopped from the command line or from code, with the
monitoring commands watching it live.

The 2.x daemon was removed because a request crossing a process boundary on every
inference step was pure cost. This design keeps the boundary out of the step: the
program asks the process for its memory once, as file descriptors, and maps them. From
then on its kernels read and write the same physical pages the process holds, with no
message per step. Messages are only for control: start, status, map and stop.

## How it works

### One process per model

Each process holds one model and has a name, derived from the model, such as
`qwen3-0.6b`, or given with `--name`. Its socket is `<run dir>/<name>.sock`, and its log
is `<run dir>/<name>.log`. The run directory is `DOVATA_RUN_DIR` when it is set, else
`~/.dovata/run`, so a terminal, a launchd job and a Jade program all find the same
processes. The directory is readable by its owner only, and so is each socket.

A process holds its name with a lock on `<run dir>/<name>.lock` for as long as it runs,
and the system releases the lock when it exits, however it exits. A second process of
the same name is refused while the lock is held. A process that was killed leaves its
socket file behind; the next program to find nothing answering removes it, and the next
process of that name replaces it. The lock is what makes that replacement safe when two
starts race.

### Starting

`start` gets the model the way `dovata plan` does: a local file, or one GGUF from the
Hugging Face Hub, the unquantized one unless `StartOptions::want` names another. It
never quantizes. It then runs `dovata host` in the background, in a process group of its
own so a Ctrl-C at the terminal does not reach it, with the `Spec` as JSON in its
argument. The process:

1. plans the model for this machine, as `dovata machine target` describes it;
2. listens on its socket, so status and stop work while it loads;
3. sets up the plan's memory: each part file mapped once, read-only, and each allocated
   region as a shared memory object with no name, `memfd_create` on Linux and an
   unlinked `shm_open` object on macOS;
4. reads every page of the model file and writes every page of each allocation, so the
   whole model is in physical memory;
5. writes one line to standard output, ready or the reason it stopped.

`start` returns when that line arrives. When nobody reads the line, because the program
that started it was stopped during the load, the process stops itself.

### Mapping

`map` sends the plan's table hash and gets back one file descriptor per part file and
per allocation, passed over the socket with `SCM_RIGHTS`, and where each region lies in
them. Each region is mapped with `MAP_SHARED`: weights read-only, allocations
read-write. A write through one program's mapping is seen by every other mapping and by
the process. The mappings stay valid after the process stops; the memory goes away when
the last program unmaps it.

Two more checks keep a program from crashing on a model file changed underneath it.
The process records each part file's size and time when it maps it, and refuses `map`
once either changes, since a file rewritten in place changes the bytes under every
mapping. The program also checks each region against its file's size before mapping
it, since reading a mapped page past the end of a file crashes with SIGBUS. On Linux
each memfd is sealed at its size, so no program handed one can shrink it under the
others; macOS refuses a second `ftruncate` of a shared memory object on its own.

The table hash check is what keeps the compiled offsets right. A program's offsets are
constants from the plan it was compiled against. If the process holds a plan with
another hash, such as a different KV cache size, `map` is refused rather than handing
out memory the offsets do not fit.

A `MappedRegion` gives its base as a raw pointer, which is what a kernel binds. Its
`read` and `write` methods copy bytes one at a time as relaxed atomics, which is slow
but stays defined while another thread or process writes the same bytes; they are for
inputs, outputs and checks, not for weights in bulk.

On a Mac the program wraps each mapped region in a no-copy `MTLBuffer`, the way
[`src/refback/`](../refback/README.md) does. Every base is page-aligned, which Metal
requires; a window's base is moved by the same amount as its file offset is past a page.

### Status

`status` reports the model, the plan's target and table hash, and the memory live: each
part file and each region with its bytes in physical memory right now, counted with
`mincore`. Mapped weights can be dropped by the system under memory pressure and read
back later, so a falling resident count shows memory pressure on the model. It also
lists every program that mapped the memory, by process id, and whether each is still
running. `dovata machine usage` and `watch` show one line per running model beside the
machine's own readings.

### Messages

Each request is a new connection carrying one line of JSON, answered with one line of
JSON, as in `{"protocol":1,"op":"status"}`. The ops are `status`, `plan`, `map` and
`stop`. Both sides refuse another protocol version, so a library and a `dovata` program
of different versions say so instead of misreading each other. The process answers one
request at a time, and gives up on a connection that sends nothing for 10 seconds. A
reply carries at most 64 file descriptors, so a plan of more than 64 part files and
allocations is refused at start. A request with more descriptors attached than the
process reads is answered with an error, and the extra descriptors are closed.

## Files

| File | What it does |
|---|---|
| `mod.rs` | The pure half, built and tested everywhere: `Hosted`, `Spec`, `StartOptions`, `Status` and the types in it, `HostError`, names, the run directory, the messages, and what each region holds. With the feature off, every call returns `HostError::NotBuilt`. |
| `server.rs` | `serve`, the process: planning, claiming the socket, setting up and loading the memory, and answering requests. |
| `client.rs` | `start`, `start_parts`, `attach` and `running`, and the requests `Hosted` makes: status, plan, map and kill. |
| `memory.rs` | Every unsafe memory call: `Map`, a mapping unmapped on drop; reading pages in; `mincore`; shared memory objects, sealed on Linux; a file's size; and whether a process is alive. |
| `wire.rs` | Sending and reading one line with file descriptors attached, finding the process at the other end of a socket, and keeping writes from raising SIGPIPE in a program that does not ignore it. |
| `fixture.rs` | A tiny llama GGUF with f32 weights, for the tests. |
| `tests.rs` | Runs the process on a thread: every weight read through a mapping matches the file, a write shows in a second mapping, a wrong table hash and another protocol are refused, a name holds one process, a stale socket is replaced, an abandoned start stops itself, a request with 100 descriptors attached does no harm, and a model changed on disk is not handed out. |
| `process_test.rs` | An integration test against the real `dovata` program: start from code, map, the command line's `status` and `machine usage`, kill from code, start and kill from the command line, and a failed start reporting the process's own error. |

## What it does not cover yet

- *CUDA.* An NVIDIA plan puts everything in VRAM, which takes a CUDA context in the
  process and `cuIpcGetMemHandle` handles in place of file descriptors. A CUDA plan is
  refused for now; `--cpu` holds the model in host memory instead.
- *More than one GPU,* as for the planner.
- *Moving memory while the model runs,* such as paging MoE experts. Nothing moves.
- *Keeping weights resident.* They are read in at start, and the system may drop them
  under pressure; status shows it. Locking them with `mlock` is a later choice.
- *A model file truncated while mapped.* The checks above catch a change before `map`.
  A file cut short after a program has mapped it still crashes that program with SIGBUS
  on its next read past the new end; no system call on Linux prevents that. Replace a
  model file by writing a new one and renaming it into place, which leaves the old one
  intact for every mapping.
- *Stopping on SIGTERM.* A process killed with a signal exits without removing its
  socket file, which the next program removes. `dovata kill` asks it to stop instead.

## Who uses it

`src/cli/host.rs` builds `dovata start`, `status`, `kill` and the hidden `dovata host`
on it, and `src/cli/machine.rs` adds running models to `usage` and `watch`. The Jade
program uses `start`, `attach`, `map` and `kill`. It depends on `load`, `layout`,
`memplan`, `target`, `hf_hub` and `machine`, and on libc for the system calls.
