//! Author: Joe Ricks
//! File: src/host/mod.rs
//! Description: Runs dovata as the one process that holds models in memory, and lets
//! other programs start it, add and remove models, map their memory, read its status and
//! stop it.

use std::fmt;
use std::path::{Path, PathBuf};
#[cfg(all(feature = "host", unix))]
use std::ptr::NonNull;
#[cfg(all(feature = "host", unix))]
use std::sync::atomic::{AtomicU8, Ordering};

use serde::{Deserialize, Serialize};

use crate::hf_hub::{FetchOptions, Want};
use crate::layout::BindOptions;
use crate::load::{KvType, Model};
use crate::memplan::{Backing, Plan, Role};
use crate::target::{RegionKind, Target};

#[cfg(all(feature = "host", unix))]
mod client;
#[cfg(all(feature = "host", unix))]
mod memory;
#[cfg(all(feature = "host", unix))]
mod server;
#[cfg(all(test, feature = "host", unix))]
mod tests;
#[cfg(all(feature = "host", unix))]
mod wire;

#[cfg(all(feature = "host", unix))]
pub use client::{connect, start};
#[cfg(all(feature = "host", unix))]
pub use server::serve;

/// The version of the messages between a dovata process and the programs that talk to
/// it. Both sides refuse a message of another version.
pub const PROTOCOL: u32 = 2;

/// The longest name a model can have.
pub const MAX_NAME: usize = 48;

/// The longest socket path macOS accepts, which is 104 bytes with the closing NUL. Linux
/// accepts 108, so this limit holds on both.
const MAX_SOCKET_PATH: usize = 103;

/// Purpose: This struct holds the running dovata process, as a program talks to it.
/// There is one per user: every model is added to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Process {
    /// The process's socket.
    socket: PathBuf,
}

/// Purpose: This struct holds what the dovata process needs to add a model: the
/// model's files, its layout, the step to plan for, and the model's name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spec {
    /// The model's name in the process. [`spec`] derives one from the model when none is
    /// given.
    pub name: String,
    /// Every part file of the model, part 1 first, as absolute paths.
    pub parts: Vec<PathBuf>,
    /// A layout file to use instead of dovata's built-in one.
    pub layout: Option<PathBuf>,
    /// The model's `tokenizer.json`, which the plan carries.
    pub tokenizer: PathBuf,
    /// The step to plan for.
    pub step: BindOptions,
    /// Plan for the CPU even when the machine has a GPU.
    pub cpu: bool,
}

/// Purpose: This struct holds the options for starting the dovata process from code.
#[derive(Debug, Clone, Default)]
pub struct StartOptions {
    /// The `dovata` program to run. `None` uses `DOVATA_PROGRAM`, else `dovata` on
    /// `PATH`.
    pub program: Option<PathBuf>,
}

/// Purpose: This struct holds the options for adding a model to the dovata process.
#[derive(Debug, Clone)]
pub struct AddOptions {
    /// The model's name in the process. `None` derives one from the model, such as
    /// `qwen3-8b`.
    pub name: Option<String>,
    /// Which model of a Hugging Face repo to download. The default is the unquantized
    /// one; quantizing is a separate call, [`crate::quantize::quantize`].
    pub want: Want,
    /// The Hugging Face cache directory and access token.
    pub fetch: FetchOptions,
    /// A layout file to use instead of dovata's built-in one.
    pub layout: Option<PathBuf>,
    /// The model's `tokenizer.json`. `None` finds it the way
    /// [`crate::tokenize::find`] does: beside the model file, in its Hub repo, or in its
    /// base model's repo.
    pub tokenizer: Option<PathBuf>,
    /// The step to plan for. The default is `dovata plan`'s: a KV cache of 4096 tokens
    /// in f16, 512 tokens per step, and logits for one of them.
    pub step: BindOptions,
    /// Plan for the CPU even when the machine has a GPU.
    pub cpu: bool,
}

impl Default for AddOptions {
    /// Purpose: This function gives the default options for adding a model.
    ///
    /// Inputs: none.
    ///
    /// Outputs: options with no name, the unquantized model, the built-in layout,
    /// `dovata plan`'s step, and this machine's GPU when it has one.
    fn default() -> AddOptions
    {
        AddOptions {
            name: None,
            want: Want::Unquantized,
            fetch: FetchOptions::default(),
            layout: None,
            tokenizer: None,
            step: BindOptions { n_ctx: 4096, n_tokens: 512, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 },
            cpu: false,
        }
    }
}

/// Purpose: This struct holds what the dovata process reports once it answers on its
/// socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ready {
    /// Its process id.
    pub pid: u32,
    /// Its socket.
    pub socket: PathBuf,
    /// [`PROTOCOL`] in the process.
    pub protocol: u32,
    /// The dovata version the process runs, such as `3.1.0`.
    pub version: String,
}

/// Purpose: This enum holds the one line the dovata process started by [`start`] writes
/// to its standard output: that it is ready, or why it stopped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StartLine {
    /// The process answers on its socket.
    Ready(Ready),
    /// The process could not start, and has exited.
    Error(String),
}

/// Purpose: This struct holds the dovata process's status: every model it holds, with
/// each one's memory region by region.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Status {
    /// The process id.
    pub pid: u32,
    /// The dovata version it runs.
    pub version: String,
    /// When the process started, in seconds since 1970.
    pub started_unix: u64,
    /// Every model it holds, sorted by name.
    pub models: Vec<ModelStatus>,
}

/// Purpose: This struct holds one model's status: its plan, its memory region by region
/// with how much of each is in physical memory, and the programs that mapped it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStatus {
    /// The model's name in the process, such as `qwen3-8b`.
    pub name: String,
    /// When it was added, in seconds since 1970.
    pub added_unix: u64,
    /// Whether it is still loading.
    pub state: State,
    /// Which model it is.
    pub model: ModelInfo,
    /// The machine its plan is for.
    pub target: Target,
    /// The plan's table hash. A program's compiled offsets are right for this model
    /// only when they came from a plan with the same hash.
    pub table_hash: String,
    /// Every part file of the model, part 1 first, mapped whole.
    pub files: Vec<FileUse>,
    /// Every region of the plan, in id order. Windows of the same file can overlap, so
    /// the weights' memory is counted in `files`, not here.
    pub regions: Vec<RegionUse>,
    /// Every program that has mapped the model's memory, in the order they mapped it.
    pub clients: Vec<Client>,
    /// The tokenizer the model's plan carries.
    pub tokenizer: Option<TokenizerUse>,
}

/// Purpose: This struct holds a short description of a model's tokenizer, for status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizerUse {
    /// The GGUF's tokenizer kind, such as `gpt2`.
    pub kind: String,
    /// The tokens it defines.
    pub tokens: u32,
    /// The ids that end the model's turn.
    pub stop: Vec<u32>,
    /// The SHA-256 of its `tokenizer.json`.
    pub sha256: String,
}

/// Purpose: This enum lists the states a model in the dovata process can be in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum State {
    /// The process is reading the model into memory.
    Loading {
        /// Bytes read so far.
        loaded_bytes: u64,
        /// Bytes to read in all.
        total_bytes: u64,
    },
    /// The model is in memory, and programs may map it.
    Ready,
}

/// Purpose: This struct holds which model a name in the process stands for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// `general.name`, when the model has one.
    pub name: Option<String>,
    /// `general.architecture`.
    pub architecture: String,
    /// The quantization type, such as `Q4_K_M`, when dovata knows it.
    pub file_type: Option<String>,
}

/// Purpose: This struct holds one part file of a model and how much of it is in physical
/// memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileUse {
    /// The file's path.
    pub path: PathBuf,
    /// Its size in bytes.
    pub bytes: u64,
    /// The bytes of it in physical memory right now, in whole pages, as `mincore`
    /// reports them. The system can drop pages of a mapped file under memory pressure
    /// and read them back later, so this can fall below `bytes`.
    pub resident_bytes: u64,
}

/// Purpose: This struct holds one region of a model and how much of it is in physical
/// memory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionUse {
    /// The region's id in the plan.
    pub id: u16,
    /// The kind of memory.
    pub kind: RegionKind,
    /// What the region holds.
    pub holds: Vec<Holds>,
    /// Its size in bytes.
    pub bytes: u64,
    /// The bytes of it in physical memory right now, in whole pages, as `mincore`
    /// reports them. The system can drop pages of mapped weights under memory pressure
    /// and read them back later, so this can fall below `bytes`.
    pub resident_bytes: u64,
}

/// Purpose: This enum lists what a region can hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Holds {
    /// Weights, mapped from the model file.
    Weights,
    /// The KV cache.
    KvCache,
    /// Recurrent state.
    State,
    /// One step's inputs, scratch and outputs.
    Step,
}

/// Purpose: This struct holds one program that mapped a model's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Client {
    /// The program's process id.
    pub pid: u32,
    /// When it mapped the memory, in seconds since 1970.
    pub mapped_unix: u64,
    /// Whether the program is still running.
    pub alive: bool,
}

impl Status {
    /// Purpose: This function adds up the memory every model in the process holds.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the total in bytes.
    pub fn bytes(&self) -> u64
    {
        self.models.iter().map(ModelStatus::bytes).sum()
    }

    /// Purpose: This function adds up the memory every model holds that is in physical
    /// memory right now.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the total in bytes.
    pub fn resident_bytes(&self) -> u64
    {
        self.models.iter().map(ModelStatus::resident_bytes).sum()
    }

    /// Purpose: This function finds one model by its name.
    ///
    /// Inputs:
    /// - `name`: the model's name in the process.
    ///
    /// Outputs: the model's status, or `None` when the process holds no such model.
    pub fn model(
        &self,
        name: &str,
    ) -> Option<&ModelStatus>
    {
        self.models.iter().find(|m| m.name == name)
    }
}

impl ModelStatus {
    /// Purpose: This function adds up the memory the model holds: every part file once,
    /// and every allocated region.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the total in bytes.
    pub fn bytes(&self) -> u64
    {
        self.files.iter().map(|f| f.bytes).sum::<u64>() + self.allocated().map(|r| r.bytes).sum::<u64>()
    }

    /// Purpose: This function adds up the memory the model holds that is in physical
    /// memory right now.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the total in bytes.
    pub fn resident_bytes(&self) -> u64
    {
        self.files.iter().map(|f| f.resident_bytes).sum::<u64>() + self.allocated().map(|r| r.resident_bytes).sum::<u64>()
    }

    /// Purpose: This function lists the regions the process allocated for the model,
    /// leaving out windows of the part files.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the allocated regions, in id order.
    pub fn allocated(&self) -> impl Iterator<Item = &RegionUse>
    {
        self.regions.iter().filter(|r| r.kind != RegionKind::Mmap)
    }
}

/// Purpose: This struct holds a hosted model's memory, mapped into this program. Each
/// region is mapped where the plan's offsets expect it, and stays mapped until the
/// [`Mapping`] is dropped, even if the dovata process stops.
#[cfg(all(feature = "host", unix))]
#[derive(Debug)]
pub struct Mapping {
    /// The plan's table hash, checked against the one the program asked for.
    table_hash: String,
    /// One per region, in id order.
    regions: Vec<MappedRegion>,
}

/// Purpose: This struct holds one region of a hosted model, mapped into this program.
#[cfg(all(feature = "host", unix))]
#[derive(Debug)]
pub struct MappedRegion {
    /// The region's id in the plan.
    id: u16,
    /// The kind of memory.
    kind: RegionKind,
    /// The region's base: where offset 0 of the plan's entries for it is.
    base: NonNull<u8>,
    /// Its size in bytes, as the plan gives it.
    len: usize,
    /// Whether this program may write it: allocated regions yes, weights never.
    writable: bool,
    /// The mapping the base points into, unmapped when this is dropped.
    _map: memory::Map,
}

// SAFETY: a `MappedRegion` owns its mapping and hands out only raw pointers into it, so
// moving it to another thread, or reading its fields from several, cannot by itself
// cause a data race. What a program does through the pointers is its own to order.
#[cfg(all(feature = "host", unix))]
unsafe impl Send for MappedRegion {}
// SAFETY: as for `Send` above.
#[cfg(all(feature = "host", unix))]
unsafe impl Sync for MappedRegion {}

#[cfg(all(feature = "host", unix))]
impl Mapping {
    /// Purpose: This function gives the plan's table hash.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the hash, in lowercase hex.
    pub fn table_hash(&self) -> &str
    {
        &self.table_hash
    }

    /// Purpose: This function gives every mapped region, in id order.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the regions.
    pub fn regions(&self) -> &[MappedRegion]
    {
        &self.regions
    }

    /// Purpose: This function finds one mapped region by its id.
    ///
    /// Inputs:
    /// - `id`: the region's id in the plan.
    ///
    /// Outputs: the region, or `None` when the plan has no such region.
    pub fn region(
        &self,
        id: u16,
    ) -> Option<&MappedRegion>
    {
        self.regions.get(usize::from(id)).filter(|r| r.id == id)
    }
}

#[cfg(all(feature = "host", unix))]
impl MappedRegion {
    /// Purpose: This function gives the region's id in the plan.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the id.
    pub fn id(&self) -> u16
    {
        self.id
    }

    /// Purpose: This function gives the region's kind of memory.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the kind.
    pub fn kind(&self) -> RegionKind
    {
        self.kind
    }

    /// Purpose: This function gives the region's size.
    ///
    /// Inputs: none.
    ///
    /// Outputs: its size in bytes.
    pub fn len(&self) -> usize
    {
        self.len
    }

    /// Purpose: This function says whether the region is empty.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for a region of no bytes.
    pub fn is_empty(&self) -> bool
    {
        self.len == 0
    }

    /// Purpose: This function says whether this program may write the region.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` for an allocated region, `false` for mapped weights.
    pub fn writable(&self) -> bool
    {
        self.writable
    }

    /// Purpose: This function gives the region's base address, for reading.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the base. An entry's bytes start at the base plus its offset.
    pub fn as_ptr(&self) -> *const u8
    {
        self.base.as_ptr()
    }

    /// Purpose: This function gives the region's base address, for writing.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the base, or `None` for weights, which are mapped read-only.
    pub fn as_mut_ptr(&self) -> Option<*mut u8>
    {
        self.writable.then_some(self.base.as_ptr())
    }

    /// Purpose: This function copies bytes out of the region.
    ///
    /// Inputs:
    /// - `offset`: where to start, in bytes from the base.
    /// - `len`: how many bytes.
    ///
    /// Outputs: the bytes, or `None` when the range runs past the region's end.
    pub fn read(
        &self,
        offset: usize,
        len: usize,
    ) -> Option<Vec<u8>>
    {
        let end = offset.checked_add(len).filter(|e| *e <= self.len)?;
        // Byte by byte, as relaxed atomics: another thread or process may write an
        // allocated region at the same time, and a plain copy racing a write is undefined
        // behavior. A relaxed byte load also works on read-only memory such as weights.
        let out = (offset..end)
            .map(|i| {
                // SAFETY: `i` lies inside the region, checked above, and the region stays
                // mapped while `self` lives. Every access these methods make is atomic.
                unsafe { AtomicU8::from_ptr(self.base.as_ptr().add(i)) }.load(Ordering::Relaxed)
            })
            .collect();
        Some(out)
    }

    /// Purpose: This function copies bytes into the region.
    ///
    /// Inputs:
    /// - `offset`: where to start, in bytes from the base.
    /// - `bytes`: the bytes to write.
    ///
    /// Outputs: `true` when written; `false` for a read-only region or a range that
    /// runs past the region's end.
    pub fn write(
        &self,
        offset: usize,
        bytes: &[u8],
    ) -> bool
    {
        let fits = offset.checked_add(bytes.len()).is_some_and(|e| e <= self.len);
        if !self.writable || !fits {
            return false;
        }
        for (i, b) in bytes.iter().enumerate() {
            // SAFETY: the byte lies inside the region, checked above, and the region is a
            // shared read-write mapping that stays mapped while `self` lives. As in
            // `read`, every access is a relaxed atomic.
            unsafe { AtomicU8::from_ptr(self.base.as_ptr().add(offset + i)) }.store(*b, Ordering::Relaxed);
        }
        true
    }
}

/// Purpose: This enum lists the errors starting, talking to or running the dovata
/// process can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostError {
    /// This build of dovata cannot run a process: the `host` feature is off, or the
    /// system is not Unix.
    NotBuilt,
    /// The name is not one a model can have.
    BadName(String),
    /// The dovata process is not running.
    NotRunning,
    /// The dovata process is already running, with this process id.
    AlreadyRunning(u32),
    /// The model cannot be fetched, opened, laid out or planned.
    Model(String),
    /// The plan needs memory dovata cannot hold yet, such as CUDA's VRAM.
    Unsupported(String),
    /// The dovata program could not be started.
    Start(String),
    /// The process answered with an error, such as a different table hash or a model
    /// that does not fit beside the others.
    Refused(String),
    /// A message was not in the expected form, or was of another protocol version.
    Protocol(String),
    /// A file, socket or memory call failed.
    Io(String),
}

impl fmt::Display for HostError {
    /// Purpose: This function formats errors from running or talking to the dovata
    /// process.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        match self {
            HostError::NotBuilt => write!(
                f,
                "this build of dovata cannot run its process: it was compiled without the `host` cargo feature, or not for Unix"
            ),
            HostError::BadName(m) => write!(f, "{m}"),
            HostError::NotRunning => write!(f, "dovata is not running; start it with `dovata start`"),
            HostError::AlreadyRunning(pid) => write!(f, "dovata is already running, pid {pid}"),
            HostError::Model(m) => write!(f, "{m}"),
            HostError::Unsupported(m) => write!(f, "{m}"),
            HostError::Start(m) => write!(f, "the dovata process did not start: {m}"),
            HostError::Refused(m) => write!(f, "{m}"),
            HostError::Protocol(m) => write!(f, "a message from the dovata process was not understood: {m}"),
            HostError::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for HostError {}

impl Process {
    /// Purpose: This function gives the process's socket.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the socket's path.
    pub fn socket(&self) -> &Path
    {
        &self.socket
    }

    /// Purpose: This function asks the process for its status and every model's.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the [`Status`], or a [`HostError`] when the process does not answer.
    pub fn status(&self) -> Result<Status, HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::status(self);
        #[cfg(not(all(feature = "host", unix)))]
        Err(HostError::NotBuilt)
    }

    /// Purpose: This function gets a model and adds it to the process, and waits until
    /// it is in memory.
    ///
    /// Inputs:
    /// - `model`: a GGUF file, or a Hugging Face link to a repo or to one file in it.
    /// - `opts`: the add options. A repo's model is the one `opts.want` names, the
    ///   unquantized one by default; nothing is quantized.
    ///
    /// Outputs: the model's status once it is in memory, or a [`HostError`], such as a
    /// name already in use or a model that does not fit beside the others.
    pub fn add(
        &self,
        model: &str,
        opts: &AddOptions,
    ) -> Result<ModelStatus, HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::add(self, model, opts);
        #[cfg(not(all(feature = "host", unix)))]
        {
            let _ = (model, opts);
            Err(HostError::NotBuilt)
        }
    }

    /// Purpose: This function adds a model already on disk to the process, and waits
    /// until it is in memory.
    ///
    /// Inputs:
    /// - `parts`: every part file of the model, part 1 first.
    /// - `opts`: the add options; `want` and `fetch` are not used.
    ///
    /// Outputs: the model's status once it is in memory, or a [`HostError`].
    pub fn add_parts(
        &self,
        parts: &[PathBuf],
        opts: &AddOptions,
    ) -> Result<ModelStatus, HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::add_parts(self, parts, opts);
        #[cfg(not(all(feature = "host", unix)))]
        {
            let _ = (parts, opts);
            Err(HostError::NotBuilt)
        }
    }

    /// Purpose: This function removes a model from the process. Programs that mapped it
    /// keep their mappings; its memory goes when the last of them unmaps it.
    ///
    /// Inputs:
    /// - `name`: the model's name in the process.
    ///
    /// Outputs: `Ok(())`, or a [`HostError`] when the process holds no such model.
    pub fn remove(
        &self,
        name: &str,
    ) -> Result<(), HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::remove(self, name);
        #[cfg(not(all(feature = "host", unix)))]
        {
            let _ = name;
            Err(HostError::NotBuilt)
        }
    }

    /// Purpose: This function asks the process for one model's whole plan, the JSON
    /// `dovata plan --json` prints.
    ///
    /// Inputs:
    /// - `name`: the model's name in the process.
    ///
    /// Outputs: the plan as JSON, or a [`HostError`].
    pub fn plan(
        &self,
        name: &str,
    ) -> Result<serde_json::Value, HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::plan(self, name);
        #[cfg(not(all(feature = "host", unix)))]
        {
            let _ = name;
            Err(HostError::NotBuilt)
        }
    }

    /// Purpose: This function maps one model's memory into this program, every region
    /// where the plan's offsets expect it.
    ///
    /// Inputs:
    /// - `name`: the model's name in the process.
    /// - `table_hash`: the table hash of the plan this program's offsets came from. The
    ///   process refuses a different one, since the offsets would then be wrong.
    ///
    /// Outputs: the [`Mapping`], or a [`HostError`] such as a different table hash or a
    /// model that is still loading.
    #[cfg(all(feature = "host", unix))]
    pub fn map(
        &self,
        name: &str,
        table_hash: &str,
    ) -> Result<Mapping, HostError>
    {
        client::map(self, name, table_hash)
    }

    /// Purpose: This function gives one model's tokenizer, built from the tokenizer its
    /// plan carries, to run in this program.
    ///
    /// Inputs:
    /// - `name`: the model's name in the process.
    /// - `table_hash`: the table hash of the plan this program was compiled against. The
    ///   process refuses a different one, since the tokenizer could then be another.
    ///
    /// Outputs: the [`crate::tokenize::Tokenizer`], or a [`HostError`].
    #[cfg(all(feature = "host", feature = "tokenize", unix))]
    pub fn tokenizer(
        &self,
        name: &str,
        table_hash: &str,
    ) -> Result<crate::tokenize::Tokenizer, HostError>
    {
        client::tokenizer(self, name, table_hash)
    }

    /// Purpose: This function stops the process, with every model in it, and waits until
    /// it has exited.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `Ok(())` once the process is gone, or a [`HostError`] when it does not
    /// answer. Memory a program has mapped stays mapped until that program unmaps it.
    pub fn kill(&self) -> Result<(), HostError>
    {
        #[cfg(all(feature = "host", unix))]
        return client::kill(self);
        #[cfg(not(all(feature = "host", unix)))]
        Err(HostError::NotBuilt)
    }
}

/// Purpose: This function returns `NotBuilt` in a build without the `host` feature.
///
/// Inputs:
/// - `opts`: the start options; unused.
///
/// Outputs: always [`HostError::NotBuilt`].
#[cfg(not(all(feature = "host", unix)))]
pub fn start(
    opts: &StartOptions,
) -> Result<Process, HostError>
{
    let _ = opts;
    Err(HostError::NotBuilt)
}

/// Purpose: This function returns `NotBuilt` in a build without the `host` feature.
///
/// Inputs: none.
///
/// Outputs: always [`HostError::NotBuilt`].
#[cfg(not(all(feature = "host", unix)))]
pub fn connect() -> Result<Process, HostError>
{
    Err(HostError::NotBuilt)
}

/// Purpose: This function returns `NotBuilt` in a build without the `host` feature.
///
/// Inputs:
/// - `ready`: what to call once the process answers; unused.
///
/// Outputs: always [`HostError::NotBuilt`].
#[cfg(not(all(feature = "host", unix)))]
pub fn serve(
    ready: impl FnOnce(&Ready) -> bool,
) -> Result<(), HostError>
{
    let _ = ready;
    Err(HostError::NotBuilt)
}

/// Purpose: This function builds what the dovata process needs to add a model.
///
/// Inputs:
/// - `parts`: every part file of the model, part 1 first.
/// - `opts`: the add options; their name, layout, tokenizer, step and CPU choice are
///   used, and their fetch options when the tokenizer has to be found on the Hub.
///
/// Outputs: the [`Spec`], with absolute paths, a name and a tokenizer, or a
/// [`HostError`] when a file cannot be found or the name is not allowed.
pub fn spec(
    parts: &[PathBuf],
    opts: &AddOptions,
) -> Result<Spec, HostError>
{
    // Absolute, not canonical: a Hugging Face cache path is a link into its `blobs`
    // directory, and the link's path is the one a person can read.
    let absolute = |p: &PathBuf| {
        std::fs::metadata(p).and_then(|_| std::path::absolute(p)).map_err(|e| HostError::Model(format!("{}: {e}", p.display())))
    };
    let parts = parts.iter().map(absolute).collect::<Result<Vec<_>, _>>()?;
    let first = parts.first().ok_or_else(|| HostError::Model("the model has no files".into()))?;
    let model = Model::from_parts(&parts).map_err(|e| HostError::Model(format!("{}: {e}", first.display())))?;
    let name = opts.name.clone().unwrap_or_else(|| name_for(model.meta().name(), first));
    check_name(&name)?;
    let layout = opts.layout.as_ref().map(absolute).transpose()?;
    let tokenizer = match &opts.tokenizer {
        Some(t) => absolute(t)?,
        None => crate::tokenize::find(first, model.meta(), None, &opts.fetch).map_err(|e| HostError::Model(e.to_string()))?,
    };
    Ok(Spec { name, parts, layout, tokenizer, step: opts.step, cpu: opts.cpu })
}

/// Purpose: This function gives the memory a target lets the plans on it use.
///
/// Inputs:
/// - `target`: the target.
///
/// Outputs: the budget in bytes: Metal's working set on Apple, host memory on the CPU,
/// VRAM on CUDA.
pub fn budget(
    target: &Target,
) -> u64
{
    match *target {
        Target::Apple { unified_bytes, .. } => unified_bytes,
        Target::Cpu { ram_bytes, .. } => ram_bytes,
        Target::Cuda { vram_bytes } => vram_bytes,
    }
}

/// Purpose: This function checks that one more model fits beside the models the process
/// already holds. The planner checks each model alone; this keeps the sum of them within
/// one machine's memory.
///
/// Inputs:
/// - `held`: the bytes the process's models hold already, files and allocations.
/// - `adding`: the bytes the new model needs, its files and allocations.
/// - `target`: the new model's target, whose budget the sum must fit.
///
/// Outputs: `Ok(())`, or a message saying how far over the budget the sum is.
pub fn fits(
    held: u64,
    adding: u64,
    target: &Target,
) -> Result<(), String>
{
    let budget = budget(target);
    let sum = held.saturating_add(adding);
    if held > 0 && sum > budget {
        return Err(format!(
            "the model needs {adding} bytes, and the models dovata holds already use {held} of the {budget} this machine allows, so it does not fit; remove one with `dovata remove`"
        ));
    }
    Ok(())
}

/// Purpose: This function derives a model's name from its metadata or its file.
///
/// Inputs:
/// - `general_name`: the model's `general.name`, when it has one.
/// - `file`: the model's first part file.
///
/// Outputs: a name of lowercase letters, digits, dots and dashes, such as
/// `qwen3-coder-30b-a3b-instruct`, no longer than [`MAX_NAME`].
pub fn name_for(
    general_name: Option<&str>,
    file: &Path,
) -> String
{
    let stem = file.file_name().and_then(|s| s.to_str()).unwrap_or_default();
    let stem = stem.strip_suffix(".gguf").unwrap_or(stem);
    // A split model's parts end in `-00001-of-00003`; the name is the model's, not the part's.
    let stem = match stem.rsplit_once("-of-") {
        Some((head, tail)) if tail.len() == 5 && tail.bytes().all(|b| b.is_ascii_digit()) => {
            head.rsplit_once('-').map_or(head, |(h, _)| h)
        }
        _ => stem,
    };
    let source = general_name.filter(|n| !n.trim().is_empty()).unwrap_or(stem);
    let mut name = String::new();
    for c in source.chars().flat_map(char::to_lowercase) {
        let c = if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' };
        if !(c == '-' && name.ends_with('-')) {
            name.push(c);
        }
    }
    let name: String = name.trim_matches(['-', '.']).chars().take(MAX_NAME).collect();
    let name = name.trim_end_matches(['-', '.']).to_owned();
    if name.is_empty() {
        "model".into()
    } else {
        name
    }
}

/// Purpose: This function checks that a name is one a model can have.
///
/// Inputs:
/// - `name`: the name.
///
/// Outputs: `Ok(())`, or [`HostError::BadName`] saying what is wrong.
pub fn check_name(
    name: &str,
) -> Result<(), HostError>
{
    let allowed = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '-' | '_');
    let starts = name.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit());
    if name.is_empty() || name.len() > MAX_NAME || !starts || !name.chars().all(allowed) {
        return Err(HostError::BadName(format!(
            "{name:?} is not a model name: use 1 to {MAX_NAME} lowercase letters, digits, dots, dashes and underscores, starting with a letter or digit"
        )));
    }
    Ok(())
}

/// Purpose: This function finds the directory that holds the dovata process's socket,
/// lock and log.
///
/// Inputs: none; it reads `DOVATA_RUN_DIR` and `HOME`.
///
/// Outputs: `DOVATA_RUN_DIR` when it is set, else `~/.dovata/run`; or a
/// [`HostError::Io`] when neither is set.
pub fn run_dir() -> Result<PathBuf, HostError>
{
    run_dir_from(std::env::var_os("DOVATA_RUN_DIR").map(PathBuf::from), std::env::var_os("HOME").map(PathBuf::from))
}

/// Purpose: This function picks the run directory from the two places it can come from.
///
/// Inputs:
/// - `over`: `DOVATA_RUN_DIR`, when set.
/// - `home`: `HOME`, when set.
///
/// Outputs: the directory, or a [`HostError::Io`] when neither is set.
fn run_dir_from(
    over: Option<PathBuf>,
    home: Option<PathBuf>,
) -> Result<PathBuf, HostError>
{
    match (over.filter(|p| !p.as_os_str().is_empty()), home.filter(|p| !p.as_os_str().is_empty())) {
        (Some(dir), _) => Ok(dir),
        (None, Some(home)) => Ok(home.join(".dovata").join("run")),
        (None, None) => Err(HostError::Io("neither DOVATA_RUN_DIR nor HOME is set, so dovata cannot find its run directory".into())),
    }
}

/// Purpose: This function gives the dovata process's socket path.
///
/// Inputs:
/// - `dir`: the run directory.
///
/// Outputs: the path, or a [`HostError::Io`] when it is longer than a socket path may
/// be.
pub fn socket_path(
    dir: &Path,
) -> Result<PathBuf, HostError>
{
    let path = dir.join("dovata.sock");
    if path.as_os_str().len() > MAX_SOCKET_PATH {
        return Err(HostError::Io(format!(
            "the socket path {} is longer than the {MAX_SOCKET_PATH} bytes a socket path may be; set DOVATA_RUN_DIR to a shorter directory",
            path.display()
        )));
    }
    Ok(path)
}

/// Purpose: This function gives the dovata process's log file.
///
/// Inputs:
/// - `dir`: the run directory.
///
/// Outputs: the path.
pub fn log_path(
    dir: &Path,
) -> PathBuf
{
    dir.join("dovata.log")
}

/// Purpose: This function lists what a region of a plan holds.
///
/// Inputs:
/// - `plan`: the plan.
/// - `region`: the region's id.
///
/// Outputs: each kind of thing in the region once, in the order the plan's entries name
/// them.
pub fn holds(
    plan: &Plan,
    region: u16,
) -> Vec<Holds>
{
    if plan.regions.get(usize::from(region)).is_some_and(|r| matches!(r.backing, Backing::File { .. })) {
        return vec![Holds::Weights];
    }
    let mut out = Vec::new();
    for e in plan.entries.iter().filter(|e| e.region == region) {
        let h = match e.role {
            Role::Weight => Holds::Weights,
            Role::Kv => Holds::KvCache,
            Role::State => Holds::State,
            Role::Input | Role::Scratch | Role::Output => Holds::Step,
            Role::View { .. } => continue,
        };
        if !out.contains(&h) {
            out.push(h);
        }
    }
    out
}

/// Purpose: This struct holds one request to a dovata process, with the protocol
/// version it was written for.
#[cfg_attr(not(all(feature = "host", unix)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Ask {
    /// [`PROTOCOL`] in the program that asks.
    pub(crate) protocol: u32,
    /// The request.
    #[serde(flatten)]
    pub(crate) request: Request,
}

/// Purpose: This enum lists the requests the dovata process answers.
#[cfg_attr(not(all(feature = "host", unix)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum Request {
    /// The process's [`Status`].
    Status,
    /// Add a model and start loading it.
    Add {
        /// The model.
        spec: Spec,
    },
    /// Remove a model.
    Remove {
        /// The model's name.
        name: String,
    },
    /// A model's whole plan, as JSON.
    Plan {
        /// The model's name.
        name: String,
    },
    /// A model's memory, as file descriptors to map.
    Map {
        /// The model's name.
        name: String,
        /// The table hash of the plan the asking program's offsets came from.
        table_hash: String,
    },
    /// Stop the process.
    Stop,
}

/// Purpose: This enum lists the answers the dovata process gives.
#[cfg_attr(not(all(feature = "host", unix)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Reply {
    /// The answer to [`Request::Status`], boxed since a status is much larger than the
    /// other replies.
    Status(Box<Status>),
    /// The answer to [`Request::Add`]: the model is planned, fits, and is loading.
    Added,
    /// The answer to [`Request::Remove`].
    Removed,
    /// The answer to [`Request::Plan`].
    Plan(serde_json::Value),
    /// The answer to [`Request::Map`]; the file descriptors come with it.
    Map(MapReply),
    /// The answer to [`Request::Stop`]: the process exits after sending it.
    Stopping,
    /// The request was refused, and why.
    Error(String),
}

/// Purpose: This struct holds how to map each region from the file descriptors sent
/// with it.
#[cfg_attr(not(all(feature = "host", unix)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct MapReply {
    /// The plan's table hash.
    pub(crate) table_hash: String,
    /// How many file descriptors come with the reply.
    pub(crate) objects: usize,
    /// One per region, in id order.
    pub(crate) regions: Vec<RegionMap>,
}

/// Purpose: This struct holds how to map one region.
#[cfg_attr(not(all(feature = "host", unix)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RegionMap {
    /// The region's id.
    pub(crate) id: u16,
    /// The kind of memory.
    pub(crate) kind: RegionKind,
    /// Which of the reply's file descriptors holds it, from 0.
    pub(crate) object: usize,
    /// Where the region starts in that file, in bytes.
    pub(crate) offset: u64,
    /// The region's size in bytes.
    pub(crate) bytes: u64,
    /// The alignment the region's base must have.
    pub(crate) align: u64,
    /// Whether the program may write it.
    pub(crate) writable: bool,
}

#[cfg(test)]
mod pure_tests {
    use super::*;

    #[test]
    fn names_come_from_the_model()
    {
        let f = Path::new("/m/Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(name_for(Some("Qwen3-Coder-30B-A3B-Instruct"), f), "qwen3-coder-30b-a3b-instruct");
        assert_eq!(name_for(Some("refback fixture"), f), "refback-fixture");
        assert_eq!(name_for(None, f), "qwen3-8b-q4-k-m");
        assert_eq!(name_for(Some("  "), Path::new("/m/Kimi-K2-Q2_K-00001-of-00009.gguf")), "kimi-k2-q2-k");
        assert_eq!(name_for(Some("--!!--"), f), "model");
        let long = name_for(Some(&"a-".repeat(60)), f);
        assert!(long.len() <= MAX_NAME && !long.ends_with('-'), "{long}");
        for n in [name_for(Some("Qwen3 (8B) / instruct"), f), name_for(None, Path::new("/x/.gguf"))] {
            assert!(check_name(&n).is_ok(), "{n}");
        }
    }

    #[test]
    fn bad_names_are_refused()
    {
        for good in ["qwen3-8b", "a", "m.v2_beta", &"x".repeat(MAX_NAME)] {
            assert!(check_name(good).is_ok(), "{good:?}");
        }
        for bad in ["", "-a", ".a", "Qwen", "a b", "a/b", "../x", &"x".repeat(MAX_NAME + 1)] {
            assert!(matches!(check_name(bad), Err(HostError::BadName(_))), "{bad:?}");
        }
    }

    #[test]
    fn the_run_directory_is_dovata_run_dir_else_home()
    {
        let p = PathBuf::from;
        assert_eq!(run_dir_from(Some(p("/r")), Some(p("/h"))), Ok(p("/r")));
        assert_eq!(run_dir_from(Some(p("")), Some(p("/h"))), Ok(p("/h/.dovata/run")));
        assert_eq!(run_dir_from(None, Some(p("/h"))), Ok(p("/h/.dovata/run")));
        assert!(run_dir_from(None, None).is_err());
    }

    #[test]
    fn a_socket_path_must_fit()
    {
        assert_eq!(socket_path(Path::new("/r")), Ok(PathBuf::from("/r/dovata.sock")));
        assert_eq!(log_path(Path::new("/r")), PathBuf::from("/r/dovata.log"));
        let deep = PathBuf::from(format!("/{}", "d".repeat(100)));
        assert!(matches!(socket_path(&deep), Err(HostError::Io(m)) if m.contains("DOVATA_RUN_DIR")));
    }

    #[test]
    fn a_model_must_fit_beside_the_others()
    {
        let apple = Target::Apple { unified_bytes: 100, max_buffer_bytes: 50, page_bytes: 16384 };
        assert_eq!(budget(&apple), 100);
        assert_eq!(budget(&Target::Cpu { ram_bytes: 7, page_bytes: 4096 }), 7);
        assert!(fits(60, 40, &apple).is_ok());
        let over = fits(61, 40, &apple).unwrap_err();
        assert!(over.contains("40 bytes") && over.contains("61 of the 100") && over.contains("dovata remove"), "{over}");
        // The first model is the planner's to judge, which counts what it holds the way
        // its target does; this check is only between models.
        assert!(fits(0, 1000, &apple).is_ok());
        assert!(fits(u64::MAX, 1, &apple).is_err());
    }

    #[test]
    fn messages_keep_their_form()
    {
        let ask = Ask { protocol: PROTOCOL, request: Request::Map { name: "m".into(), table_hash: "ab".into() } };
        let text = serde_json::to_string(&ask).unwrap();
        assert_eq!(text, r#"{"protocol":2,"op":"map","name":"m","table_hash":"ab"}"#);
        assert_eq!(serde_json::from_str::<Ask>(&text).unwrap(), ask);
        let stop = serde_json::to_string(&Ask { protocol: PROTOCOL, request: Request::Stop }).unwrap();
        assert_eq!(stop, r#"{"protocol":2,"op":"stop"}"#);
        let spec = Spec { name: "m".into(), parts: vec![PathBuf::from("/m.gguf")], layout: None, tokenizer: PathBuf::from("/tokenizer.json"), step: AddOptions::default().step, cpu: false };
        let add = Ask { protocol: PROTOCOL, request: Request::Add { spec } };
        assert_eq!(serde_json::from_str::<Ask>(&serde_json::to_string(&add).unwrap()).unwrap(), add);

        let model = ModelStatus {
            name: "m".into(),
            added_unix: 5,
            state: State::Loading { loaded_bytes: 1, total_bytes: 2 },
            model: ModelInfo { name: None, architecture: "llama".into(), file_type: Some("F32".into()) },
            target: Target::Cpu { ram_bytes: 1 << 30, page_bytes: 4096 },
            table_hash: "ab".into(),
            files: vec![FileUse { path: PathBuf::from("/m.gguf"), bytes: 100, resident_bytes: 90 }],
            regions: vec![
                RegionUse { id: 0, kind: RegionKind::Mmap, holds: vec![Holds::Weights], bytes: 60, resident_bytes: 50 },
                RegionUse { id: 1, kind: RegionKind::Mmap, holds: vec![Holds::Weights], bytes: 60, resident_bytes: 50 },
                RegionUse { id: 2, kind: RegionKind::Host, holds: vec![Holds::KvCache], bytes: 10, resident_bytes: 8 },
            ],
            clients: vec![Client { pid: 9, mapped_unix: 4, alive: true }],
            tokenizer: Some(TokenizerUse { kind: "gpt2".into(), tokens: 32, stop: vec![2], sha256: "cd".into() }),
        };
        // The two windows overlap in the file, so the file counts once.
        assert_eq!((model.bytes(), model.resident_bytes()), (110, 98));
        let other = ModelStatus { name: "n".into(), ..model.clone() };
        let status = Status { pid: 7, version: "3.1.0".into(), started_unix: 3, models: vec![model, other] };
        assert_eq!((status.bytes(), status.resident_bytes()), (220, 196));
        assert_eq!(status.model("n").map(|m| m.name.as_str()), Some("n"));
        assert!(status.model("x").is_none());
        for reply in [Reply::Status(Box::new(status.clone())), Reply::Added, Reply::Removed, Reply::Stopping, Reply::Error("no".into())] {
            let text = serde_json::to_string(&reply).unwrap();
            assert_eq!(serde_json::from_str::<Reply>(&text).unwrap(), reply, "{text}");
        }

        let line = StartLine::Ready(Ready { pid: 7, socket: PathBuf::from("/r/dovata.sock"), protocol: PROTOCOL, version: "3.1.0".into() });
        let text = serde_json::to_string(&line).unwrap();
        assert!(text.starts_with(r#"{"ready":{"#), "{text}");
        assert_eq!(serde_json::from_str::<StartLine>(&text).unwrap(), line);
    }

    #[test]
    fn a_spec_travels_as_json()
    {
        let spec = Spec {
            name: "m".into(),
            parts: vec![PathBuf::from("/m.gguf")],
            layout: None,
            tokenizer: PathBuf::from("/m/tokenizer.json"),
            step: BindOptions { n_ctx: 8192, n_tokens: 64, n_outputs: 1, kv_k: KvType::Q8_0, kv_v: KvType::BF16 },
            cpu: true,
        };
        let text = serde_json::to_string(&spec).unwrap();
        assert!(text.contains(r#""kv_k":"q8_0""#) && text.contains(r#""kv_v":"bf16""#), "{text}");
        assert_eq!(serde_json::from_str::<Spec>(&text).unwrap(), spec);
        let bad = text.replace("q8_0", "q2_k");
        assert!(serde_json::from_str::<Spec>(&bad).is_err());
    }
}
