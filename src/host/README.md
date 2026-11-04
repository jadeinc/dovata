# host: the dovata process, which holds models in memory

`host` runs dovata as one long-lived process that holds models in memory. The process
starts empty; each model is added to it, planned for this machine, and loaded, and stays
until it is removed or the process stops. A compiled Jade program maps a model's memory
into itself and runs its own kernels on it. dovata still never runs a model's math: the
process holds memory, reports on it and hands it out, and the program does the compute.

```rust
use dovata::host::{self, AddOptions, StartOptions};

// Start the process, with no models, or find the one already running.
let dovata = host::start(&StartOptions::default()).or_else(|_| host::connect())?;

// Add a model: download it if needed, plan it for this machine, load it.
let model = dovata.add("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF", &AddOptions::default())?;

// Map it: every region of its plan, where the program's compiled offsets expect it. The
// process refuses a program whose offsets came from a different plan.
let mem = dovata.map(&model.name, PLAN_TABLE_HASH)?;
let weights = mem.region(0).unwrap().as_ptr();   // bind as a no-copy MTLBuffer, say

// Watch, take the model out, stop.
let status = dovata.status()?;                   // every model, every region, how much is resident
dovata.remove(&model.name)?;
dovata.kill()?;
```

The command line does the same: `dovata start`, `add`, `remove`, `status` and `kill`, in
[`src/cli/`](../cli/README.md). Both talk to the same process, so a model added from one
is seen, mapped and removed from the other.

## Why it exists

dovata 3.0 began as a build-time crate only: the Jade compiler planned a model with it,
and the compiled program set up the plan's memory itself at startup. That made each
program load the model on its own, and left nothing on the machine that owned the
model's memory or could say where it was. Since 2026-11-03, dovata is also the process
that owns that memory, started and stopped from the command line or from code, with the
monitoring commands watching it live.

There is one dovata process, never one per model. Every model's memory then goes
through one place, which can see all of it and refuse a model that does not fit beside
the others. That is also what lets one process hold several models.

The 2.x daemon was removed because a request crossing a process boundary on every
inference step was pure cost. This design keeps the boundary out of the step: the
program asks the process for a model's memory once, as file descriptors, and maps them.
From then on its kernels read and write the same physical pages the process holds, with
no message per step. Messages are only for control: start, add, remove, status, map and
stop.

## How it works

### One process per user

The process's socket is `<run dir>/dovata.sock`, its lock `dovata.lock` and its log
`dovata.log`. The run directory is `DOVATA_RUN_DIR` when it is set, else
`~/.dovata/run`, so a terminal, a launchd job and a Jade program all find the same
process. The directory is readable by its owner only, and so is the socket.

The process holds a lock on `dovata.lock` for as long as it runs, and the system
releases the lock when it exits, however it exits. A second process is refused while the
lock is held, and is told the first one's process id, which the lock file holds. A
process that was killed leaves its socket file behind, and the next one to start
replaces it; the lock is what makes that safe when two starts race.

The run directory is per user, so two users on one machine each have their own process.
Neither sees the other's memory.

### Starting

`start` runs `dovata host` in the background, in a process group of its own so a Ctrl-C
at the terminal does not reach it. The process takes the lock, listens on its socket,
and writes one line to standard output: ready, or the reason it stopped. `start` returns
when that line arrives. When nobody reads the line, because the program that started it
has gone, the process stops itself.

### Adding a model

`add` gets the model the way `dovata plan` does: a local file, or one GGUF from the
Hugging Face Hub, the unquantized one unless `AddOptions::want` names another. It never
quantizes. It sends the process a `Spec`: the model's files, its name, its layout and
the step to plan for. The process:

1. plans the model for this machine, as `dovata machine target` describes it;
2. sets up the plan's memory: each part file mapped once, read-only, and each allocated
   region as a shared memory object with no name, `memfd_create` on Linux and an
   unlinked `shm_open` object on macOS;
3. checks the model fits beside the ones it holds: the files and allocations of every
   model together must fit the new model's budget, Metal's working set on a Mac. The
   planner has already checked the model alone;
4. answers, and reads every page of the model file and writes every page of each
   allocation on a thread of its own, so the whole model ends up in physical memory.

`add` then waits until the model's state is ready and returns its status. A name is
taken once: a second model of the same name is refused until the first is removed.
Names come from the model, such as `qwen3-0.6b`, or from `AddOptions::name`.

`remove` takes a model out. The process drops its mappings at once; programs that
mapped the model keep their mappings, and the memory goes when the last one unmaps it.

### Mapping

`map` names a model and the plan's table hash, and gets back one file descriptor per
part file and per allocation, passed over the socket with `SCM_RIGHTS`, and where each
region lies in them. Each region is mapped with `MAP_SHARED`: weights read-only,
allocations read-write. A write through one program's mapping is seen by every other
mapping of that model and by the process. Two models never share an allocation. The
mappings stay valid after the model is removed or the process stops.

The table hash check is what keeps the compiled offsets right. A program's offsets are
constants from the plan it was compiled against. If the process holds a plan with
another hash, such as a different KV cache size, `map` is refused rather than handing
out memory the offsets do not fit.

Two more checks keep a program from crashing on a model file changed underneath it.
The process records each part file's size and time when it maps it, and refuses `map`
once either changes, since a file rewritten in place changes the bytes under every
mapping. The program also checks each region against its file's size before mapping
it, since reading a mapped page past the end of a file crashes with SIGBUS. On Linux
each memfd is sealed at its size, so no program handed one can shrink it under the
others; macOS refuses a second `ftruncate` of a shared memory object on its own.

A `MappedRegion` gives its base as a raw pointer, which is what a kernel binds. Its
`read` and `write` methods copy bytes one at a time as relaxed atomics, which is slow
but stays defined while another thread or process writes the same bytes; they are for
inputs, outputs and checks, not for weights in bulk.

On a Mac the program wraps each mapped region in a no-copy `MTLBuffer`, the way
[`src/refback/`](../refback/README.md) does. Every base is page-aligned, which Metal
requires; a window's base is moved by the same amount as its file offset is past a page.

### Status

`status` reports the process and each model: its plan's target and table hash, and its
memory live, each part file and each region with its bytes in physical memory right
now, counted with `mincore`. Mapped weights can be dropped by the system under memory
pressure and read back later, so a falling resident count shows memory pressure on the
model. It also lists every program that mapped each model, by process id, and whether
each is still running. `dovata machine usage` and `watch` show the process and one line
per model beside the machine's own readings.

### Messages

Each request is a new connection carrying one line of JSON, answered with one line of
JSON, as in `{"protocol":2,"op":"status"}`. The ops are `status`, `add`, `remove`,
`plan`, `map` and `stop`. Both sides refuse another protocol version, so a library and a
`dovata` program of different versions say so instead of misreading each other. Each
request is answered on a thread of its own, so a model loading does not hold up the
others, and a connection that sends nothing for 10 seconds is dropped. A reply carries
at most 64 file descriptors, so a plan of more than 64 part files and allocations is
refused when it is added. A request with more descriptors attached than the process
reads is answered with an error, and the extra descriptors are closed.

## Files

| File | What it does |
|---|---|
| `mod.rs` | The pure half, built and tested everywhere: `Process`, `Spec`, `StartOptions`, `AddOptions`, `Status`, `ModelStatus` and the types in them, `HostError`, names, the run directory, the budget check, the messages, and what each region holds. With the feature off, every call returns `HostError::NotBuilt`. |
| `server.rs` | `serve`, the process: the lock and socket, the table of models, planning, setting up and loading each model's memory, and answering requests. |
| `client.rs` | `start` and `connect`, and the requests a `Process` makes: status, add, remove, plan, map and kill. |
| `memory.rs` | Every unsafe memory call: `Map`, a mapping unmapped on drop; reading pages in; `mincore`; shared memory objects, sealed on Linux; a file's size; and whether a process is alive. |
| `wire.rs` | Sending and reading one line with file descriptors attached, finding the process at the other end of a socket, and keeping writes from raising SIGPIPE in a program that does not ignore it. |
| `fixture.rs` | A tiny llama GGUF with f32 weights, for the tests. |
| `tests.rs` | Runs the process on a thread: it starts empty; every weight read through a mapping matches the file; a write shows in a second mapping of the same model and not in another model; several models share the process and each name is taken once; a removed model's mapping stays; a second process, a wrong table hash and another protocol are refused; a stale socket is replaced; an abandoned start stops itself; a model that cannot be planned leaves the process running; a request with 100 descriptors attached does no harm; and a model changed on disk is not handed out. |
| `process_test.rs` | An integration test against the real `dovata` program: start from code, add a model, map it, the command line's `start`, `add`, `status`, `remove` and `machine usage` against the same process, kill from code, and start, add and kill from the command line. |

## What it does not cover yet

- *CUDA.* An NVIDIA plan puts everything in VRAM, which takes a CUDA context in the
  process and `cuIpcGetMemHandle` handles in place of file descriptors. A CUDA plan is
  refused for now; `--cpu` holds the model in host memory instead.
- *More than one GPU,* as for the planner.
- *Sharing between models.* Two models added from the same file map it twice; the page
  cache holds the file once, but the budget check counts it twice.
- *Moving memory while a model runs,* such as paging MoE experts. Nothing moves.
- *Keeping weights resident.* They are read in when the model is added, and the system
  may drop them under pressure; status shows it. Locking them with `mlock` is a later
  choice.
- *A model file truncated while mapped.* The checks above catch a change before `map`.
  A file cut short after a program has mapped it still crashes that program with SIGBUS
  on its next read past the new end; no system call on Linux prevents that. Replace a
  model file by writing a new one and renaming it into place, which leaves the old one
  intact for every mapping.
- *Stopping on SIGTERM.* A process killed with a signal exits without removing its
  socket file, which the next start replaces. `dovata kill` asks it to stop instead.

## Who uses it

`src/cli/host.rs` builds `dovata start`, `add`, `remove`, `status`, `kill` and the hidden
`dovata host` on it, and `src/cli/machine.rs` adds the process and its models to `usage`
and `watch`. The Jade program uses `start`, `connect`, `add`, `map`, `remove` and `kill`.
It depends on `load`, `layout`, `memplan`, `target`, `hf_hub` and `machine`, and on libc
for the system calls.
