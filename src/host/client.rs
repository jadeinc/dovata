//! Author: Joe Ricks
//! File: src/host/client.rs
//! Description: Starts a dovata process from another program, finds running ones, and
//! asks them for their status, their plan, their memory and to stop.

use std::io::{self, BufRead, BufReader};
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::hf_hub::{self, ModelSource};

use super::memory::{self, Map};
use super::{
    check_name, log_path, run_dir, socket_path, spec, wire, Ask, HostError, Hosted, MapReply, MappedRegion, Mapping, Reply, Request, StartLine,
    StartOptions, Status, PROTOCOL,
};

/// How long [`Hosted::kill`] waits for a stopped process to exit.
const STOP_WAIT: Duration = Duration::from_secs(10);

/// Purpose: This function gets a model and starts a dovata process that holds it.
///
/// Inputs:
/// - `model`: a GGUF file, or a Hugging Face link to a repo or to one file in it.
/// - `opts`: the start options. A repo's model is the one `opts.want` names, the
///   unquantized one by default; nothing is quantized.
///
/// Outputs: the [`Hosted`] process once its model is in memory, or a [`HostError`].
pub fn start(
    model: &str,
    opts: &StartOptions,
) -> Result<Hosted, HostError>
{
    let source = ModelSource::parse(model).map_err(|e| HostError::Model(e.to_string()))?;
    let fetched = hf_hub::fetch(&source, opts.want, &opts.fetch).map_err(|e| HostError::Model(e.to_string()))?;
    start_parts(&fetched.parts, opts)
}

/// Purpose: This function starts a dovata process that holds a model already on disk.
///
/// Inputs:
/// - `parts`: every part file of the model, part 1 first.
/// - `opts`: the start options; `want` and `fetch` are not used.
///
/// Outputs: the [`Hosted`] process once its model is in memory, or a [`HostError`],
/// such as [`HostError::AlreadyRunning`] or the process's own reason for stopping.
pub fn start_parts(
    parts: &[PathBuf],
    opts: &StartOptions,
) -> Result<Hosted, HostError>
{
    let spec = spec(parts, opts)?;
    let dir = run_dir()?;
    if attach_at(&spec.name, &socket_path(&dir, &spec.name)?).is_ok() {
        return Err(HostError::AlreadyRunning(spec.name));
    }
    let io = |e: io::Error, what: &Path| HostError::Io(format!("{}: {e}", what.display()));
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir).map_err(|e| io(e, &dir))?;
    let log = log_path(&dir, &spec.name);
    let log_file = std::fs::File::create(&log).map_err(|e| io(e, &log))?;

    let program = opts.program.clone().or_else(|| std::env::var_os("DOVATA_PROGRAM").map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("dovata"));
    // `dovata host` takes the spec as its argument, writes one line to standard output
    // when the model is loaded, and writes its log to standard error. An argument rather
    // than a pipe, so a program that does not ignore SIGPIPE cannot be stopped by a
    // child that exits before reading it. Its own process group keeps a Ctrl-C at this
    // program's terminal from reaching it.
    let text = serde_json::to_string(&spec).map_err(|e| HostError::Protocol(e.to_string()))?;
    let mut child = Command::new(&program)
        .arg("host")
        .arg(&text)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(log_file)
        .current_dir("/")
        .process_group(0)
        .spawn()
        .map_err(|e| HostError::Start(format!("cannot run {}: {e}. Install dovata, or set DOVATA_PROGRAM to its path", program.display())))?;
    let mut line = String::new();
    if let Some(stdout) = child.stdout.take() {
        let _ = BufReader::new(stdout).read_line(&mut line);
    }
    let started = serde_json::from_str::<StartLine>(line.trim());
    match started {
        Ok(StartLine::Ready(r)) if r.protocol == PROTOCOL => {
            // The process outlives this call. Waiting for it on a thread of its own
            // reaps it when it exits, so it does not linger as a zombie while this
            // program runs.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            Ok(Hosted { name: r.name, socket: r.socket })
        }
        Ok(StartLine::Ready(r)) => {
            let h = Hosted { name: r.name, socket: r.socket };
            let _ = kill(&h);
            let _ = child.wait();
            Err(HostError::Start(format!(
                "{} speaks protocol {} and dovata {}, and this program speaks protocol {PROTOCOL}; use the same dovata version",
                program.display(),
                r.protocol,
                r.version
            )))
        }
        Ok(StartLine::Error(m)) => {
            let _ = child.wait();
            Err(HostError::Start(m))
        }
        Err(_) => {
            let code = child.wait().map(|s| s.to_string()).unwrap_or_else(|e| e.to_string());
            let tail = std::fs::read_to_string(&log).unwrap_or_default();
            let tail: Vec<&str> = tail.lines().rev().take(5).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            Err(HostError::Start(format!("{} {code} before the model was loaded; its log {} ends:\n{}", program.display(), log.display(), tail.join("\n"))))
        }
    }
}

/// Purpose: This function finds the running dovata process with a given name.
///
/// Inputs:
/// - `name`: the process's name.
///
/// Outputs: the [`Hosted`] process, or [`HostError::NotRunning`] when none answers.
pub fn attach(
    name: &str,
) -> Result<Hosted, HostError>
{
    check_name(name)?;
    attach_at(name, &socket_path(&run_dir()?, name)?)
}

/// Purpose: This function finds the running dovata process on a given socket.
///
/// Inputs:
/// - `name`: the process's name.
/// - `socket`: its socket.
///
/// Outputs: the [`Hosted`] process, or [`HostError::NotRunning`] when none answers. A
/// socket file no process answers on, left by one that was killed, is removed.
pub(crate) fn attach_at(
    name: &str,
    socket: &Path,
) -> Result<Hosted, HostError>
{
    let h = Hosted { name: name.to_owned(), socket: socket.to_path_buf() };
    match status(&h) {
        Ok(_) => Ok(h),
        Err(HostError::NotRunning(n)) => {
            if socket.exists() {
                let _ = std::fs::remove_file(socket);
            }
            Err(HostError::NotRunning(n))
        }
        Err(e) => Err(e),
    }
}

/// Purpose: This function lists every running dovata process.
///
/// Inputs: none.
///
/// Outputs: the processes, sorted by name, or a [`HostError`] when the run directory
/// cannot be read. A missing run directory means none are running.
pub fn running() -> Result<Vec<Hosted>, HostError>
{
    let dir = run_dir()?;
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(HostError::Io(format!("{}: {e}", dir.display()))),
    };
    let mut names: Vec<String> =
        entries.flatten().filter_map(|e| e.file_name().to_str()?.strip_suffix(".sock").map(str::to_owned)).filter(|n| check_name(n).is_ok()).collect();
    names.sort();
    Ok(names.iter().filter_map(|n| attach(n).ok()).collect())
}

/// Purpose: This function sends one request and reads the reply.
///
/// Inputs:
/// - `h`: the process.
/// - `request`: the request.
///
/// Outputs: the reply and any file descriptors sent with it, or a [`HostError`]:
/// [`HostError::NotRunning`] when nothing answers on the socket, and
/// [`HostError::Refused`] when the process answers with an error.
fn ask(
    h: &Hosted,
    request: Request,
) -> Result<(Reply, Vec<OwnedFd>), HostError>
{
    let sock = match UnixStream::connect(&h.socket) {
        Ok(s) => s,
        Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused) => return Err(HostError::NotRunning(h.name.clone())),
        Err(e) => return Err(HostError::Io(format!("{}: {e}", h.socket.display()))),
    };
    wire::quiet(&sock);
    let _ = sock.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = sock.set_write_timeout(Some(Duration::from_secs(10)));
    let text = serde_json::to_string(&Ask { protocol: PROTOCOL, request }).map_err(|e| HostError::Protocol(e.to_string()))?;
    let io = |e: io::Error| HostError::Io(format!("{}: {e}", h.socket.display()));
    wire::send(&sock, format!("{text}\n").as_bytes(), &[]).map_err(io)?;
    let (line, fds) = wire::recv(&sock).map_err(io)?;
    match serde_json::from_slice::<Reply>(&line) {
        Ok(Reply::Error(m)) => Err(HostError::Refused(m)),
        Ok(r) => Ok((r, fds)),
        Err(e) => Err(HostError::Protocol(format!("{e}: {}", String::from_utf8_lossy(&line[..line.len().min(200)])))),
    }
}

/// Purpose: This function asks a process for its status.
///
/// Inputs:
/// - `h`: the process.
///
/// Outputs: the [`Status`], or a [`HostError`].
pub(crate) fn status(
    h: &Hosted,
) -> Result<Status, HostError>
{
    match ask(h, Request::Status)? {
        (Reply::Status(s), _) => Ok(*s),
        (other, _) => Err(unexpected("status", &other)),
    }
}

/// Purpose: This function asks a process for its whole plan.
///
/// Inputs:
/// - `h`: the process.
///
/// Outputs: the plan as JSON, or a [`HostError`].
pub(crate) fn plan(
    h: &Hosted,
) -> Result<serde_json::Value, HostError>
{
    match ask(h, Request::Plan)? {
        (Reply::Plan(p), _) => Ok(p),
        (other, _) => Err(unexpected("plan", &other)),
    }
}

/// Purpose: This function maps a process's memory into this program.
///
/// Inputs:
/// - `h`: the process.
/// - `table_hash`: the table hash of the plan this program's offsets came from.
///
/// Outputs: the [`Mapping`], or a [`HostError`].
pub(crate) fn map(
    h: &Hosted,
    table_hash: &str,
) -> Result<Mapping, HostError>
{
    let (reply, fds) = ask(h, Request::Map { table_hash: table_hash.to_owned() })?;
    let map = match reply {
        Reply::Map(m) => m,
        other => return Err(unexpected("map", &other)),
    };
    mapping(map, &fds)
}

/// Purpose: This function maps every region a map reply describes.
///
/// Inputs:
/// - `reply`: the reply.
/// - `fds`: the file descriptors sent with it.
///
/// Outputs: the [`Mapping`], or a [`HostError`] when the reply and descriptors do not
/// agree or a region cannot be mapped where the plan needs it.
fn mapping(
    reply: MapReply,
    fds: &[OwnedFd],
) -> Result<Mapping, HostError>
{
    if fds.len() != reply.objects {
        return Err(HostError::Protocol(format!("the reply names {} file descriptors and {} came with it", reply.objects, fds.len())));
    }
    let mut regions = Vec::new();
    for (i, r) in reply.regions.iter().enumerate() {
        if usize::from(r.id) != i {
            return Err(HostError::Protocol(format!("region {} is listed at index {i}", r.id)));
        }
        let fd = fds.get(r.object).ok_or_else(|| HostError::Protocol(format!("region {} names file descriptor {}, of {}", r.id, r.object, fds.len())))?;
        let len = usize::try_from(r.bytes).map_err(|_| HostError::Io(format!("region {}: {} bytes do not fit in memory", r.id, r.bytes)))?;
        // A region past the end of its file would map, and then crash this program with
        // SIGBUS on its first read there, as after the file was rewritten shorter.
        let size = memory::object_size(fd.as_fd()).map_err(|e| HostError::Io(format!("region {}: {e}", r.id)))?;
        if r.offset.checked_add(r.bytes).is_none_or(|end| end > size) {
            return Err(HostError::Refused(format!("region {} runs past the end of its {size}-byte file; the model changed on disk", r.id)));
        }
        let (map, delta) = Map::range(fd.as_fd(), r.offset, len, r.writable).map_err(|e| HostError::Io(format!("mapping region {}: {e}", r.id)))?;
        // SAFETY: `delta` is less than a page and the mapping is `delta + len` bytes
        // long, so the base lies inside it.
        let base = unsafe { std::ptr::NonNull::new_unchecked(map.as_ptr().as_ptr().add(delta)) };
        if r.align > 1 && !(base.as_ptr() as usize as u64).is_multiple_of(r.align) {
            return Err(HostError::Io(format!("region {} was mapped at {:p}, which is not aligned to {}", r.id, base.as_ptr(), r.align)));
        }
        regions.push(MappedRegion { id: r.id, kind: r.kind, base, len, writable: r.writable, _map: map });
    }
    Ok(Mapping { table_hash: reply.table_hash, regions })
}

/// Purpose: This function stops a process and waits until it has gone.
///
/// Inputs:
/// - `h`: the process.
///
/// Outputs: `Ok(())` once nothing answers on its socket, or a [`HostError`].
pub(crate) fn kill(
    h: &Hosted,
) -> Result<(), HostError>
{
    match ask(h, Request::Stop)? {
        (Reply::Stopping, _) => {}
        (other, _) => return Err(unexpected("stop", &other)),
    }
    let start = Instant::now();
    while start.elapsed() < STOP_WAIT {
        if UnixStream::connect(&h.socket).is_err() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(HostError::Io(format!("{} said it would stop, and still answers after {} s", h.name, STOP_WAIT.as_secs())))
}

/// Purpose: This function describes a reply of the wrong kind.
///
/// Inputs:
/// - `asked`: what was asked for.
/// - `got`: the reply.
///
/// Outputs: a [`HostError::Protocol`].
fn unexpected(
    asked: &str,
    got: &Reply,
) -> HostError
{
    let kind = match got {
        Reply::Status(_) => "a status",
        Reply::Plan(_) => "a plan",
        Reply::Map(_) => "a map",
        Reply::Stopping => "a stop",
        Reply::Error(_) => "an error",
    };
    HostError::Protocol(format!("asked for {asked}, and got {kind}"))
}
