//! Author: Joe Ricks
//! File: src/host/server.rs
//! Description: The dovata process: plans a model for this machine, sets up and loads
//! its memory, and answers status, plan, map and stop requests on a Unix socket.

use std::fs::File;
use std::io;
use std::os::fd::{AsFd, AsRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::layout::{self, Layout};
use crate::load::Model;
use crate::memplan::{Backing, Plan, Region};
use crate::quantize::Ftype;
use crate::target::{RegionKind, Target};

use super::memory::{self, Map};
use super::{
    check_name, holds, run_dir, socket_path, wire, Ask, Client, FileUse, HostError, MapReply, ModelInfo, Ready, RegionMap, RegionUse, Reply,
    Request, Spec, State, Status, PROTOCOL,
};

/// Purpose: This struct holds one part file of the model: open, and mapped whole.
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

/// Purpose: This struct holds a plan's memory as this process holds it.
struct Held {
    /// Each part file, part 1 first.
    parts: Vec<Part>,
    /// One per region, in id order: the allocation for an allocated region, `None` for
    /// a window of a part file.
    allocs: Vec<Option<Alloc>>,
}

/// Purpose: This struct holds everything the request loop and the loading thread share.
struct Shared {
    /// The process's name.
    name: String,
    /// The plan the memory is laid out by.
    plan: Plan,
    /// The model's description, for status.
    model: ModelInfo,
    /// The memory.
    held: Held,
    /// Bytes loaded so far.
    loaded: AtomicU64,
    /// Bytes to load in all.
    total: u64,
    /// Whether loading has finished.
    ready: AtomicBool,
    /// When the process started, in seconds since 1970.
    started: u64,
    /// Every program that mapped the memory.
    clients: Mutex<Vec<Client>>,
}

/// Purpose: This struct holds the process's name while it runs: a lock on
/// `<name>.lock`, and the socket file, which it removes when dropped, on a stop and on an
/// error alike.
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

/// Purpose: This function runs a dovata process: it plans the model for this machine,
/// sets up its memory, loads it, and answers requests until it is stopped.
///
/// Inputs:
/// - `spec`: the model and how to plan it.
/// - `ready`: called once the model is in memory. It returns `false` when nobody is
///   listening any more, such as a `dovata start` that was stopped with Ctrl-C, and the
///   process then stops.
///
/// Outputs: `Ok(())` after a stop request, or a [`HostError`] when the model cannot be
/// planned or held.
pub fn serve(
    spec: &Spec,
    ready: impl FnOnce(&Ready) -> bool + Send + 'static,
) -> Result<(), HostError>
{
    serve_in(spec, &run_dir()?, ready)
}

/// Purpose: This function runs a dovata process whose socket is in a given directory.
///
/// Inputs:
/// - `spec`: the model and how to plan it.
/// - `dir`: the run directory.
/// - `ready`: as for [`serve`].
///
/// Outputs: as for [`serve`].
pub(crate) fn serve_in(
    spec: &Spec,
    dir: &Path,
    ready: impl FnOnce(&Ready) -> bool + Send + 'static,
) -> Result<(), HostError>
{
    check_name(&spec.name)?;
    let (plan, model) = plan(spec)?;
    let socket = socket_path(dir, &spec.name)?;
    let (listener, claim) = claim(dir, &socket, &spec.name)?;
    let held = Held::set_up(&spec.parts, &plan)?;
    let total = held.parts.iter().map(|p| p.bytes).sum::<u64>() + held.allocs.iter().flatten().map(|a| a.map_len()).sum::<u64>();
    let started = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let shared = Arc::new(Shared {
        name: spec.name.clone(),
        plan,
        model,
        held,
        loaded: AtomicU64::new(0),
        total,
        ready: AtomicBool::new(false),
        started,
        clients: Mutex::new(Vec::new()),
    });
    eprintln!("dovata {}: pid {}, socket {}, loading {} bytes", spec.name, std::process::id(), socket.display(), total);

    // Loading runs on its own thread, so the process answers status and stop while it
    // loads. A stop during loading ends the process and this thread with it.
    let loader = Arc::clone(&shared);
    let socket_for_loader = socket.clone();
    std::thread::spawn(move || {
        loader.held.load(|n| {
            loader.loaded.fetch_add(n, Ordering::Relaxed);
        });
        loader.ready.store(true, Ordering::Release);
        eprintln!("dovata {}: ready", loader.name);
        let r = Ready {
            name: loader.name.clone(),
            pid: std::process::id(),
            socket: socket_for_loader.clone(),
            protocol: PROTOCOL,
            version: env!("CARGO_PKG_VERSION").into(),
        };
        if !ready(&r) {
            eprintln!("dovata {}: nobody is waiting for the start any more, so it stops", loader.name);
            stop_self(&socket_for_loader);
        }
    });

    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("dovata {}: accept failed: {e}", spec.name);
                continue;
            }
        };
        wire::quiet(&stream);
        // Read now: on macOS the peer's pid is gone once it closes the connection, which
        // a program does as soon as it has its reply.
        let peer = wire::peer_pid(&stream);
        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
        match answer(&stream, peer, &shared) {
            Ok(true) => break,
            Ok(false) => {}
            Err(e) => eprintln!("dovata {}: a request failed: {e}", spec.name),
        }
    }
    drop(claim);
    eprintln!("dovata {}: stopped", spec.name);
    Ok(())
}

/// Purpose: This function plans the model for this machine.
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
    let model = Model::from_parts(&spec.parts).map_err(|e| bad(&e))?;
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
            "this machine's plan is for CUDA, and a dovata process holds Apple and CPU plans only so far; start it with --cpu to hold the model in host memory".into(),
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

/// Purpose: This function takes the socket path for this process, removing a socket a
/// stopped process left behind, and listens on it.
///
/// Inputs:
/// - `dir`: the run directory, made if missing, readable by this user only.
/// - `socket`: the socket's path.
/// - `name`: the process's name.
///
/// Outputs: the listener and the claim that removes the socket file, or
/// [`HostError::AlreadyRunning`] when a process answers on it.
fn claim(
    dir: &Path,
    socket: &Path,
    name: &str,
) -> Result<(UnixListener, Claim), HostError>
{
    let io = |e: io::Error, what: &Path| HostError::Io(format!("{}: {e}", what.display()));
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir).map_err(|e| io(e, dir))?;
    // The lock makes taking the name one step. Without it, two starts after a crash can
    // both find the stale socket, and the second removes the first one's new socket,
    // leaving it running with nothing able to reach it.
    let lock_path = dir.join(format!("{name}.lock"));
    let lock = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(&lock_path).map_err(|e| io(e, &lock_path))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => return Err(HostError::AlreadyRunning(name.into())),
        Err(std::fs::TryLockError::Error(e)) => return Err(io(e, &lock_path)),
    }
    if socket.exists() {
        if UnixStream::connect(socket).is_ok() {
            return Err(HostError::AlreadyRunning(name.into()));
        }
        std::fs::remove_file(socket).map_err(|e| io(e, socket))?;
    }
    let listener = UnixListener::bind(socket).map_err(|e| match e.kind() {
        io::ErrorKind::AddrInUse => HostError::AlreadyRunning(name.into()),
        _ => io(e, socket),
    })?;
    let claim = Claim { path: socket.to_path_buf(), _lock: lock };
    std::fs::set_permissions(socket, std::fs::Permissions::from_mode(0o600)).map_err(|e| io(e, socket))?;
    Ok((listener, claim))
}

/// Purpose: This function sends this process a stop request, from its own loading
/// thread.
///
/// Inputs:
/// - `socket`: the process's socket.
///
/// Outputs: none.
fn stop_self(
    socket: &Path,
)
{
    if let Ok(s) = UnixStream::connect(socket) {
        wire::quiet(&s);
        let ask = serde_json::to_string(&Ask { protocol: PROTOCOL, request: Request::Stop }).unwrap_or_default();
        let _ = wire::send(&s, format!("{ask}\n").as_bytes(), &[]);
        let _ = wire::recv(&s);
    }
}

/// Purpose: This function answers one request.
///
/// Inputs:
/// - `stream`: the connection, which carries one request.
/// - `peer`: the process id of the program at the other end, when the system says.
/// - `shared`: the process's state.
///
/// Outputs: `true` when the request was a stop, `false` otherwise; or the error that
/// kept it from being answered.
fn answer(
    stream: &UnixStream,
    peer: Option<u32>,
    shared: &Shared,
) -> io::Result<bool>
{
    let (line, _) = wire::recv(stream)?;
    let reply = |r: &Reply, fds: &[RawFd]| {
        let text = serde_json::to_string(r).map_err(io::Error::other)?;
        wire::send(stream, format!("{text}\n").as_bytes(), fds)
    };
    let ask: Ask = match serde_json::from_slice(&line) {
        Ok(a) => a,
        Err(e) => {
            reply(&Reply::Error(format!("the request is not one dovata knows: {e}")), &[])?;
            return Ok(false);
        }
    };
    if ask.protocol != PROTOCOL {
        reply(&Reply::Error(format!("this process speaks protocol {PROTOCOL}, and the request is protocol {}; use the same dovata version", ask.protocol)), &[])?;
        return Ok(false);
    }
    match ask.request {
        Request::Status => reply(&Reply::Status(Box::new(status(shared))), &[])?,
        Request::Plan => reply(&Reply::Plan(serde_json::to_value(&shared.plan).map_err(io::Error::other)?), &[])?,
        Request::Map { table_hash } => {
            if !shared.ready.load(Ordering::Acquire) {
                let msg = format!("{} is still loading: {} of {} bytes", shared.name, shared.loaded.load(Ordering::Relaxed), shared.total);
                reply(&Reply::Error(msg), &[])?;
            } else if let Some(msg) = shared.held.changed() {
                reply(&Reply::Error(msg), &[])?;
            } else if table_hash != shared.plan.table_hash {
                let msg = format!(
                    "{} holds a plan with table hash {}, and the program's offsets come from a plan with table hash {table_hash}; plan the model with the same options, or start it with the program's",
                    shared.name, shared.plan.table_hash
                );
                reply(&Reply::Error(msg), &[])?;
            } else {
                let (map, fds) = shared.held.objects(&shared.plan);
                reply(&Reply::Map(map), &fds)?;
                if let Some(pid) = peer {
                    let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
                    shared.clients.lock().unwrap_or_else(|e| e.into_inner()).push(Client { pid, mapped_unix: now, alive: true });
                    eprintln!("dovata {}: mapped by pid {pid}", shared.name);
                }
            }
        }
        Request::Stop => {
            reply(&Reply::Stopping, &[])?;
            return Ok(true);
        }
    }
    Ok(false)
}

/// Purpose: This function reads the process's status, with each region's resident bytes
/// as they are right now.
///
/// Inputs:
/// - `shared`: the process's state.
///
/// Outputs: the [`Status`].
fn status(
    shared: &Shared,
) -> Status
{
    let state = if shared.ready.load(Ordering::Acquire) {
        State::Ready
    } else {
        State::Loading { loaded_bytes: shared.loaded.load(Ordering::Relaxed), total_bytes: shared.total }
    };
    let regions = shared
        .plan
        .regions
        .iter()
        .map(|r| RegionUse { id: r.id, kind: r.kind, holds: holds(&shared.plan, r.id), bytes: r.bytes, resident_bytes: shared.held.resident(r) })
        .collect();
    let files = shared.held.parts.iter().map(|p| FileUse { path: p.path.clone(), bytes: p.bytes, resident_bytes: p.map.resident(0, p.map_len()).unwrap_or(0) }).collect();
    let mut clients = shared.clients.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for c in &mut clients {
        c.alive = memory::alive(c.pid);
    }
    Status {
        name: shared.name.clone(),
        pid: std::process::id(),
        version: env!("CARGO_PKG_VERSION").into(),
        state,
        started_unix: shared.started,
        model: shared.model.clone(),
        target: shared.plan.target,
        table_hash: shared.plan.table_hash.clone(),
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
