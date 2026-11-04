//! Author: Joe Ricks
//! File: src/host/server.rs
//! Description: The dovata process: holds every model added to it, each planned for this
//! machine and loaded into memory, and answers status, add, remove, plan, map and stop
//! requests on a Unix socket.

use std::collections::BTreeMap;
use std::fs::File;
use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::layout::{self, Layout};
use crate::load::Model as GgufModel;
use crate::memplan::{Backing, Plan, Region};
use crate::quantize::Ftype;
use crate::target::{RegionKind, Target};

use super::memory::{self, Map};
use super::{
    check_name, fits, holds, run_dir, socket_path, wire, Ask, Client, FileUse, HostError, MapReply, ModelInfo, ModelStatus, Ready, RegionMap,
    RegionUse, Reply, Request, Spec, State, Status, PROTOCOL,
};

/// Purpose: This struct holds one part file of a model: open, and mapped whole.
struct Part {
    /// The file's path.
    path: PathBuf,
    /// The open file, whose descriptor programs get to map the weights.
    file: File,
    /// The whole file, mapped read-only.
    map: Map,
    /// The file's size in bytes.
    bytes: u64,
    /// When the file was last changed, as it was when the process mapped it.
    modified: Option<SystemTime>,
}

/// Purpose: This struct holds one allocated region: its shared memory object, mapped.
struct Alloc {
    /// The shared memory object, whose descriptor programs get to map it.
    fd: OwnedFd,
    /// The object, mapped read-write.
    map: Map,
}

/// Purpose: This struct holds a plan's memory as the process holds it.
struct Held {
    /// Each part file, part 1 first.
    parts: Vec<Part>,
    /// One per region, in id order: the allocation for an allocated region, `None` for
    /// a window of a part file.
    allocs: Vec<Option<Alloc>>,
}

/// Purpose: This struct holds one model the process holds: its plan, its memory, and
/// how far loading it has got.
struct Hosted {
    /// The model's name in the process.
    name: String,
    /// The plan the memory is laid out by.
    plan: Plan,
    /// Which model it is, for status.
    info: ModelInfo,
    /// The memory.
    held: Held,
    /// Bytes loaded so far.
    loaded: AtomicU64,
    /// Bytes to load in all, and the bytes the model counts against the machine.
    total: u64,
    /// Whether loading has finished.
    ready: AtomicBool,
    /// When it was added, in seconds since 1970.
    added: u64,
    /// Every program that mapped its memory.
    clients: Mutex<Vec<Client>>,
}

/// Purpose: This struct holds everything the process's request threads share.
struct Server {
    /// The process's socket, which a stop connects to so the accept loop wakes.
    socket: PathBuf,
    /// When the process started, in seconds since 1970.
    started: u64,
    /// Every model, by name.
    models: Mutex<BTreeMap<String, Arc<Hosted>>>,
    /// Set by a stop request; the accept loop ends when it sees it.
    stopping: AtomicBool,
}

/// Purpose: This struct holds the process's place in the run directory while it runs:
/// a lock on `dovata.lock`, and the socket file, which it removes when dropped, on a stop
/// and on an error alike.
struct Claim {
    /// The socket's path.
    path: PathBuf,
    /// The lock file, locked for as long as the process runs. The system releases the
    /// lock when the process exits, however it exits.
    _lock: File,
}

impl Drop for Claim {
    /// Purpose: This function removes the socket file.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Purpose: This function gives the time now.
///
/// Inputs: none.
///
/// Outputs: seconds since 1970.
fn now() -> u64
{
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Purpose: This function runs the dovata process: it takes the run directory's lock and
/// socket, and answers requests until it is stopped. It starts with no models; each
/// `add` request plans one and loads it.
///
/// Inputs:
/// - `ready`: called once the process answers on its socket. It returns `false` when
///   nobody is listening any more, such as a `dovata start` that was stopped with
///   Ctrl-C, and the process then stops.
///
/// Outputs: `Ok(())` after a stop request, or a [`HostError`], such as
/// [`HostError::AlreadyRunning`].
pub fn serve(
    ready: impl FnOnce(&Ready) -> bool,
) -> Result<(), HostError>
{
    serve_in(&run_dir()?, ready)
}

/// Purpose: This function runs the dovata process in a given run directory.
///
/// Inputs:
/// - `dir`: the run directory.
/// - `ready`: as for [`serve`].
///
/// Outputs: as for [`serve`].
pub(crate) fn serve_in(
    dir: &Path,
    ready: impl FnOnce(&Ready) -> bool,
) -> Result<(), HostError>
{
    let socket = socket_path(dir)?;
    let (listener, claim) = claim(dir, &socket)?;
    let server = Arc::new(Server { socket: socket.clone(), started: now(), models: Mutex::new(BTreeMap::new()), stopping: AtomicBool::new(false) });
    eprintln!("dovata: pid {}, socket {}", std::process::id(), socket.display());
    let r = Ready { pid: std::process::id(), socket, protocol: PROTOCOL, version: env!("CARGO_PKG_VERSION").into() };
    if !ready(&r) {
        eprintln!("dovata: nobody is waiting for the start any more, so it stops");
        return Ok(());
    }

    // Each request gets a thread of its own, so a model loading, or a program slow to
    // send its request, never holds up the others.
    for conn in listener.incoming() {
        if server.stopping.load(Ordering::Acquire) {
            break;
        }
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("dovata: accept failed: {e}");
                continue;
            }
        };
        let server = Arc::clone(&server);
        std::thread::spawn(move || {
            wire::quiet(&stream);
            // Read now: on macOS the peer's pid is gone once it closes the connection,
            // which a program does as soon as it has its reply.
            let peer = wire::peer_pid(&stream);
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
            if let Err(e) = answer(&stream, peer, &server) {
                eprintln!("dovata: a request failed: {e}");
            }
        });
    }
    drop(claim);
    eprintln!("dovata: stopped");
    Ok(())
}

/// Purpose: This function locks the model table, even after a request thread panicked
/// while holding it.
///
/// Inputs:
/// - `server`: the process's state.
///
/// Outputs: the locked table.
fn models(
    server: &Server,
) -> MutexGuard<'_, BTreeMap<String, Arc<Hosted>>>
{
    server.models.lock().unwrap_or_else(|e| e.into_inner())
}

/// Purpose: This function adds a model: it plans it for this machine, sets up its
/// memory, checks it fits beside the models already held, and starts loading it on a
/// thread of its own.
///
/// Inputs:
/// - `server`: the process's state.
/// - `spec`: the model.
///
/// Outputs: `Ok(())` once the model is loading, or a [`HostError`] saying why it cannot
/// be added.
fn add(
    server: &Server,
    spec: Spec,
) -> Result<(), HostError>
{
    check_name(&spec.name)?;
    let taken = |name: &str| HostError::Refused(format!("dovata already holds a model named {name}; remove it first, or add this one with another --name"));
    if models(server).contains_key(&spec.name) {
        return Err(taken(&spec.name));
    }
    // Planning and setting up memory happen outside the lock, so other requests are
    // answered meanwhile. Setting up only reserves address space; loading is what fills
    // memory, and it starts once the model is known to fit.
    let (plan, info) = plan(&spec)?;
    let held = Held::set_up(&spec.parts, &plan)?;
    let total = held.parts.iter().map(|p| p.bytes).sum::<u64>() + held.allocs.iter().flatten().map(|a| a.map_len()).sum::<u64>();
    let model = {
        let mut table = models(server);
        if table.contains_key(&spec.name) {
            return Err(taken(&spec.name));
        }
        let in_use = table.values().map(|m| m.total).sum::<u64>();
        fits(in_use, total, &plan.target).map_err(HostError::Refused)?;
        let model = Arc::new(Hosted {
            name: spec.name.clone(),
            plan,
            info,
            held,
            loaded: AtomicU64::new(0),
            total,
            ready: AtomicBool::new(false),
            added: now(),
            clients: Mutex::new(Vec::new()),
        });
        table.insert(spec.name.clone(), Arc::clone(&model));
        model
    };
    eprintln!("dovata: adding {}, {} bytes, from {}", model.name, total, spec.parts[0].display());
    std::thread::spawn(move || {
        model.held.load(|n| {
            model.loaded.fetch_add(n, Ordering::Relaxed);
        });
        model.ready.store(true, Ordering::Release);
        eprintln!("dovata: {} is ready", model.name);
    });
    Ok(())
}

/// Purpose: This function plans a model for this machine.
///
/// Inputs:
/// - `spec`: the model and how to plan it.
///
/// Outputs: the plan and the model's description, or a [`HostError`].
fn plan(
    spec: &Spec,
) -> Result<(Plan, ModelInfo), HostError>
{
    let first = spec.parts.first().ok_or_else(|| HostError::Model("the model has no files".into()))?;
    let bad = |e: &dyn std::fmt::Display| HostError::Model(format!("{}: {e}", first.display()));
    let model = GgufModel::from_parts(&spec.parts).map_err(|e| bad(&e))?;
    let caller = match &spec.layout {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| HostError::Model(format!("{}: {e}", path.display())))?;
            Some(Layout::from_toml(&text).map_err(|e| HostError::Model(format!("{}: {e}", path.display())))?)
        }
        None => None,
    };
    let layout = layout::for_model(&model, caller).map_err(|e| bad(&e))?;
    let machine = crate::machine::current().map_err(|e| HostError::Model(format!("cannot read this machine: {e}")))?;
    let target = machine.target(spec.cpu);
    if let Target::Cuda { .. } = target {
        return Err(HostError::Unsupported(
            "this machine's plan is for CUDA, and dovata holds Apple and CPU plans only so far; add the model with --cpu to hold it in host memory".into(),
        ));
    }
    let plan = crate::memplan::plan(&model, &layout, &spec.step, &target).map_err(|e| bad(&e))?;
    let meta = model.meta();
    let info = ModelInfo {
        name: meta.name().map(str::to_owned),
        architecture: model.architecture().to_owned(),
        file_type: meta.file_type().and_then(Ftype::from_file_type).map(|f| f.to_string()),
    };
    Ok((plan, info))
}

/// Purpose: This function takes the run directory for this process: its lock, then its
/// socket, removing a socket a killed process left behind.
///
/// Inputs:
/// - `dir`: the run directory, made if missing, readable by this user only.
/// - `socket`: the socket's path.
///
/// Outputs: the listener and the claim that holds the lock and removes the socket file,
/// or [`HostError::AlreadyRunning`] when another dovata process holds the lock.
fn claim(
    dir: &Path,
    socket: &Path,
) -> Result<(UnixListener, Claim), HostError>
{
    let io = |e: io::Error, what: &Path| HostError::Io(format!("{}: {e}", what.display()));
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| io(e, dir))?;
    // The lock makes taking the run directory one step. Without it, two starts after a
    // crash can both find the stale socket, and the second removes the first one's new
    // socket, leaving it running with nothing able to reach it.
    let lock_path = dir.join("dovata.lock");
    let mut lock = std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(&lock_path).map_err(|e| io(e, &lock_path))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            let pid = std::fs::read_to_string(&lock_path).ok().and_then(|t| t.trim().parse().ok()).unwrap_or(0);
            return Err(HostError::AlreadyRunning(pid));
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(io(e, &lock_path)),
    }
    // The lock file names its holder, so a refused start can say which process runs.
    {
        use std::io::{Seek, Write};
        let _ = lock.set_len(0).and_then(|()| lock.rewind()).and_then(|_| write!(lock, "{}", std::process::id())).and_then(|()| lock.flush());
    }
    if socket.exists() {
        std::fs::remove_file(socket).map_err(|e| io(e, socket))?;
    }
    let listener = UnixListener::bind(socket).map_err(|e| io(e, socket))?;
    let claim = Claim { path: socket.to_path_buf(), _lock: lock };
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600)).map_err(|e| io(e, socket))?;
    Ok((listener, claim))
}

/// Purpose: This function finds a model by name.
///
/// Inputs:
/// - `server`: the process's state.
/// - `name`: the model's name.
///
/// Outputs: the model, or [`HostError::Refused`] naming the models the process holds.
fn find(
    server: &Server,
    name: &str,
) -> Result<Arc<Hosted>, HostError>
{
    let table = models(server);
    table.get(name).cloned().ok_or_else(|| {
        let names: Vec<&str> = table.keys().map(String::as_str).collect();
        let held = if names.is_empty() { "no models".to_owned() } else { names.join(", ") };
        HostError::Refused(format!("dovata holds no model named {name}; it holds {held}"))
    })
}

/// Purpose: This function answers one request.
///
/// Inputs:
/// - `stream`: the connection, which carries one request.
/// - `peer`: the process id of the program at the other end, when the system says.
/// - `server`: the process's state.
///
/// Outputs: `Ok(())`, or the error that kept the request from being answered.
fn answer(
    stream: &UnixStream,
    peer: Option<u32>,
    server: &Server,
) -> io::Result<()>
{
    let (line, _) = wire::recv(stream)?;
    let reply = |r: &Reply, fds: &[RawFd]| {
        let text = serde_json::to_string(r).map_err(io::Error::other)?;
        wire::send(stream, format!("{text}\n").as_bytes(), fds)
    };
    let refuse = |e: HostError| reply(&Reply::Error(e.to_string()), &[]);
    let ask: Ask = match serde_json::from_slice(&line) {
        Ok(a) => a,
        Err(e) => return reply(&Reply::Error(format!("the request is not one dovata knows: {e}")), &[]),
    };
    if ask.protocol != PROTOCOL {
        return reply(&Reply::Error(format!("this process speaks protocol {PROTOCOL}, and the request is protocol {}; use the same dovata version", ask.protocol)), &[]);
    }
    match ask.request {
        Request::Status => reply(&Reply::Status(Box::new(status(server))), &[]),
        Request::Add { spec } => match add(server, spec) {
            Ok(()) => reply(&Reply::Added, &[]),
            Err(e) => refuse(e),
        },
        Request::Remove { name } => {
            // Taken in a statement of its own, so the table's lock is released before
            // `find` locks it again to name the models held.
            let removed = models(server).remove(&name);
            match removed {
                Some(_) => {
                    eprintln!("dovata: removed {name}");
                    reply(&Reply::Removed, &[])
                }
                None => refuse(find(server, &name).err().unwrap_or_else(|| HostError::Refused(format!("dovata holds no model named {name}")))),
            }
        }
        Request::Plan { name } => match find(server, &name) {
            Ok(m) => reply(&Reply::Plan(serde_json::to_value(&m.plan).map_err(io::Error::other)?), &[]),
            Err(e) => refuse(e),
        },
        Request::Map { name, table_hash } => {
            let m = match find(server, &name) {
                Ok(m) => m,
                Err(e) => return refuse(e),
            };
            if !m.ready.load(Ordering::Acquire) {
                return refuse(HostError::Refused(format!("{name} is still loading: {} of {} bytes", m.loaded.load(Ordering::Relaxed), m.total)));
            }
            if let Some(msg) = m.held.changed() {
                return refuse(HostError::Refused(msg));
            }
            if table_hash != m.plan.table_hash {
                return refuse(HostError::Refused(format!(
                    "dovata holds {name} with a plan whose table hash is {}, and the program's offsets come from a plan with table hash {table_hash}; add the model with the options the program was compiled with",
                    m.plan.table_hash
                )));
            }
            let (map, fds) = m.held.objects(&m.plan);
            reply(&Reply::Map(map), &fds)?;
            if let Some(pid) = peer {
                m.clients.lock().unwrap_or_else(|e| e.into_inner()).push(Client { pid, mapped_unix: now(), alive: true });
                eprintln!("dovata: {name} mapped by pid {pid}");
            }
            Ok(())
        }
        Request::Stop => {
            reply(&Reply::Stopping, &[])?;
            server.stopping.store(true, Ordering::Release);
            // The accept loop is waiting for a connection; this one wakes it, and it
            // sees the flag.
            let _ = UnixStream::connect(&server.socket);
            Ok(())
        }
    }
}

/// Purpose: This function reads the process's status, with each region's resident bytes
/// as they are right now.
///
/// Inputs:
/// - `server`: the process's state.
///
/// Outputs: the [`Status`].
fn status(
    server: &Server,
) -> Status
{
    let held: Vec<Arc<Hosted>> = models(server).values().cloned().collect();
    Status { pid: std::process::id(), version: env!("CARGO_PKG_VERSION").into(), started_unix: server.started, models: held.iter().map(|m| model_status(m)).collect() }
}

/// Purpose: This function reads one model's status.
///
/// Inputs:
/// - `m`: the model.
///
/// Outputs: the [`ModelStatus`].
fn model_status(
    m: &Hosted,
) -> ModelStatus
{
    let state = if m.ready.load(Ordering::Acquire) {
        State::Ready
    } else {
        State::Loading { loaded_bytes: m.loaded.load(Ordering::Relaxed), total_bytes: m.total }
    };
    let regions =
        m.plan.regions.iter().map(|r| RegionUse { id: r.id, kind: r.kind, holds: holds(&m.plan, r.id), bytes: r.bytes, resident_bytes: m.held.resident(r) }).collect();
    let files = m.held.parts.iter().map(|p| FileUse { path: p.path.clone(), bytes: p.bytes, resident_bytes: p.map.resident(0, p.map_len()).unwrap_or(0) }).collect();
    let mut clients = m.clients.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for c in &mut clients {
        c.alive = memory::alive(c.pid);
    }
    ModelStatus {
        name: m.name.clone(),
        added_unix: m.added,
        state,
        model: m.info.clone(),
        target: m.plan.target,
        table_hash: m.plan.table_hash.clone(),
        files,
        regions,
        clients,
    }
}

impl Part {
    /// Purpose: This function gives the length of the file's mapping.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the file's size as a mapping length.
    fn map_len(&self) -> usize
    {
        self.bytes as usize
    }
}

impl Alloc {
    /// Purpose: This function gives the size of the shared memory object.
    ///
    /// Inputs: none.
    ///
    /// Outputs: its size in bytes, a whole number of pages.
    fn map_len(&self) -> u64
    {
        self.map.len() as u64
    }
}

impl Held {
    /// Purpose: This function sets up a plan's memory: it maps each part file once,
    /// read-only, and makes a shared memory object for each allocated region.
    ///
    /// Inputs:
    /// - `paths`: the part files, part 1 first.
    /// - `plan`: the plan.
    ///
    /// Outputs: the [`Held`] memory, or a [`HostError`] for a file that cannot be mapped
    /// or a region dovata cannot hold.
    fn set_up(
        paths: &[PathBuf],
        plan: &Plan,
    ) -> Result<Held, HostError>
    {
        if paths.len() != plan.model.parts.len() {
            return Err(HostError::Model(format!("the plan has {} part files, and the model {}", plan.model.parts.len(), paths.len())));
        }
        let mut parts = Vec::new();
        for path in paths {
            let io = |e: io::Error| HostError::Io(format!("{}: {e}", path.display()));
            let file = File::open(path).map_err(io)?;
            let bytes = file.metadata().map_err(io)?.len();
            let len = usize::try_from(bytes).map_err(|_| HostError::Io(format!("{}: {bytes} bytes do not fit in memory", path.display())))?;
            let modified = file.metadata().map_err(io)?.modified().ok();
            let map = Map::new(file.as_fd(), 0, len, false).map_err(io)?;
            parts.push(Part { path: path.clone(), file, map, bytes, modified });
        }
        let page = memory::page_size() as u64;
        let mut allocs = Vec::new();
        for r in &plan.regions {
            let bad = |m: String| HostError::Model(format!("region {}: {m}", r.id));
            match (r.backing, r.kind) {
                (Backing::File { part, file_start }, RegionKind::Mmap) => {
                    let p = parts.get(usize::from(part)).ok_or_else(|| bad(format!("it maps part {}, and the model has {}", part + 1, parts.len())))?;
                    if file_start.checked_add(r.bytes).is_none_or(|end| end > p.bytes) {
                        return Err(bad(format!("its window runs past the end of {}", p.path.display())));
                    }
                    allocs.push(None);
                }
                (Backing::Alloc, RegionKind::Host | RegionKind::Unified) => {
                    if r.align > page {
                        return Err(bad(format!("it needs alignment {}, more than the {page}-byte pages shared memory starts on", r.align)));
                    }
                    let len = r.bytes.max(1).div_ceil(page) * page;
                    let io = |e: io::Error| HostError::Io(format!("shared memory of {len} bytes for region {}: {e}", r.id));
                    let fd = memory::shared_memory(len).map_err(io)?;
                    let map = Map::new(fd.as_fd(), 0, len as usize, true).map_err(io)?;
                    allocs.push(Some(Alloc { fd, map }));
                }
                (Backing::Alloc, RegionKind::Vram) => {
                    return Err(HostError::Unsupported(format!("region {} is VRAM, and a dovata process cannot hold CUDA memory yet", r.id)));
                }
                (backing, kind) => return Err(bad(format!("{kind:?} memory backed by {backing:?} is not a region dovata knows"))),
            }
        }
        let objects = parts.len() + allocs.iter().flatten().count();
        if objects > wire::MAX_FDS {
            return Err(HostError::Unsupported(format!(
                "the plan needs {objects} files and allocations, and a dovata process can hand out at most {}",
                wire::MAX_FDS
            )));
        }
        Ok(Held { parts, allocs })
    }

    /// Purpose: This function finds a part file that changed on disk after the process
    /// mapped it. A file rewritten in place changes the bytes under every mapping, and a
    /// shorter one makes a program that reads past its new end crash with SIGBUS.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a message naming the first changed file, or `None` when none changed.
    fn changed(&self) -> Option<String>
    {
        self.parts.iter().find_map(|p| {
            let now = p.file.metadata().ok();
            let same = now.as_ref().is_some_and(|m| m.len() == p.bytes && m.modified().ok() == p.modified);
            (!same).then(|| format!("{} changed on disk after the process loaded it; stop the process and start it again", p.path.display()))
        })
    }

    /// Purpose: This function brings the whole model into physical memory: every page of
    /// every part file, and every page of every allocated region.
    ///
    /// Inputs:
    /// - `progress`: called with the bytes done since its last call.
    ///
    /// Outputs: none.
    fn load(
        &self,
        mut progress: impl FnMut(u64),
    )
    {
        for p in &self.parts {
            p.map.touch(&mut progress);
        }
        for a in self.allocs.iter().flatten() {
            a.map.touch(&mut progress);
        }
    }

    /// Purpose: This function counts a region's bytes in physical memory.
    ///
    /// Inputs:
    /// - `r`: the region.
    ///
    /// Outputs: the resident bytes, or 0 when the system cannot say.
    fn resident(
        &self,
        r: &Region,
    ) -> u64
    {
        let len = r.bytes as usize;
        match (r.backing, self.allocs.get(usize::from(r.id))) {
            (Backing::File { part, file_start }, _) => {
                self.parts.get(usize::from(part)).and_then(|p| p.map.resident(file_start as usize, len).ok()).unwrap_or(0)
            }
            (Backing::Alloc, Some(Some(a))) => a.map.resident(0, len).unwrap_or(0),
            (Backing::Alloc, _) => 0,
        }
    }

    /// Purpose: This function lists what a program needs to map the memory: one file
    /// descriptor per part file and per allocated region, and how each region lies in
    /// them.
    ///
    /// Inputs:
    /// - `plan`: the plan.
    ///
    /// Outputs: the [`MapReply`] and the descriptors it numbers, part files first.
    fn objects(
        &self,
        plan: &Plan,
    ) -> (MapReply, Vec<RawFd>)
    {
        let mut fds: Vec<RawFd> = self.parts.iter().map(|p| p.file.as_raw_fd()).collect();
        let mut regions = Vec::new();
        for (r, alloc) in plan.regions.iter().zip(&self.allocs) {
            let (object, offset, writable) = match (r.backing, alloc) {
                (Backing::File { part, file_start }, _) => (usize::from(part), file_start, false),
                (Backing::Alloc, Some(a)) => {
                    fds.push(a.fd.as_raw_fd());
                    (fds.len() - 1, 0, true)
                }
                (Backing::Alloc, None) => continue,
            };
            regions.push(RegionMap { id: r.id, kind: r.kind, object, offset, bytes: r.bytes, align: r.align, writable });
        }
        (MapReply { table_hash: plan.table_hash.clone(), objects: fds.len(), regions }, fds)
    }
}
