//! Author: Joe Ricks
//! File: src/host/client.rs
//! Description: Starts the dovata process from another program, finds it when it runs,
//! and asks it to add and remove models, for its status, for a model's plan and memory,
//! and to stop.

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
    log_path, run_dir, socket_path, spec, wire, AddOptions, Ask, HostError, MapReply, MappedRegion, Mapping, ModelStatus, Process, Reply, Request,
    StartLine, StartOptions, State, Status, PROTOCOL,
};

/// How long [`Process::kill`] waits for a stopped process to exit.
const STOP_WAIT: Duration = Duration::from_secs(10);

/// How often [`Process::add`] asks whether the model it added is in memory yet.
const LOAD_POLL: Duration = Duration::from_millis(50);

/// Purpose: This function starts the dovata process, with no models in it.
///
/// Inputs:
/// - `opts`: which `dovata` program to run.
///
/// Outputs: the [`Process`] once it answers on its socket, or a [`HostError`], such as
/// [`HostError::AlreadyRunning`] or the process's own reason for stopping.
pub fn start(
    opts: &StartOptions,
) -> Result<Process, HostError>
{
    let dir = run_dir()?;
    let socket = socket_path(&dir)?;
    if let Ok(s) = status(&Process { socket: socket.clone() }) {
        return Err(HostError::AlreadyRunning(s.pid));
    }
    let io = |e: io::Error, what: &Path| HostError::Io(format!("{}: {e}", what.display()));
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir).map_err(|e| io(e, &dir))?;
    let log = log_path(&dir);
    let log_file = std::fs::OpenOptions::new().create(true).append(true).open(&log).map_err(|e| io(e, &log))?;

    let program = opts.program.clone().or_else(|| std::env::var_os("DOVATA_PROGRAM").map(PathBuf::from)).unwrap_or_else(|| PathBuf::from("dovata"));
    // `dovata host` writes one line to standard output once it answers on its socket,
    // and its log to standard error. Its own process group keeps a Ctrl-C at this
    // program's terminal from reaching it.
    let mut child = Command::new(&program)
        .arg("host")
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
    match serde_json::from_str::<StartLine>(line.trim()) {
        Ok(StartLine::Ready(r)) if r.protocol == PROTOCOL => {
            // The process outlives this call. Waiting for it on a thread of its own
            // reaps it when it exits, so it does not linger as a zombie while this
            // program runs.
            std::thread::spawn(move || {
                let _ = child.wait();
            });
            Ok(Process { socket: r.socket })
        }
        Ok(StartLine::Ready(r)) => {
            let _ = kill(&Process { socket: r.socket });
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
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            let tail: Vec<&str> = text.lines().rev().take(5).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            Err(HostError::Start(format!("{} {code}; its log {} ends:\n{}", program.display(), log.display(), tail.join("\n"))))
        }
    }
}

/// Purpose: This function finds the running dovata process.
///
/// Inputs: none.
///
/// Outputs: the [`Process`], or [`HostError::NotRunning`] when none answers.
pub fn connect() -> Result<Process, HostError>
{
    let p = Process { socket: socket_path(&run_dir()?)? };
    status(&p)?;
    Ok(p)
}

/// Purpose: This function sends one request and reads the reply.
///
/// Inputs:
/// - `p`: the process.
/// - `request`: the request.
///
/// Outputs: the reply and any file descriptors sent with it, or a [`HostError`]:
/// [`HostError::NotRunning`] when nothing answers on the socket, and
/// [`HostError::Refused`] when the process answers with an error.
fn ask(
    p: &Process,
    request: Request,
) -> Result<(Reply, Vec<OwnedFd>), HostError>
{
    let sock = match UnixStream::connect(&p.socket) {
        Ok(s) => s,
        Err(e) if matches!(e.kind(), io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused) => return Err(HostError::NotRunning),
        Err(e) => return Err(HostError::Io(format!("{}: {e}", p.socket.display()))),
    };
    wire::quiet(&sock);
    // An add plans the model before it answers, which takes a few seconds for a large one.
    let _ = sock.set_read_timeout(Some(Duration::from_secs(300)));
    let _ = sock.set_write_timeout(Some(Duration::from_secs(10)));
    let text = serde_json::to_string(&Ask { protocol: PROTOCOL, request }).map_err(|e| HostError::Protocol(e.to_string()))?;
    let io = |e: io::Error| HostError::Io(format!("{}: {e}", p.socket.display()));
    wire::send(&sock, format!("{text}\n").as_bytes(), &[]).map_err(io)?;
    let (line, fds) = wire::recv(&sock).map_err(io)?;
    match serde_json::from_slice::<Reply>(&line) {
        Ok(Reply::Error(m)) => Err(HostError::Refused(m)),
        Ok(r) => Ok((r, fds)),
        Err(e) => Err(HostError::Protocol(format!("{e}: {}", String::from_utf8_lossy(&line[..line.len().min(200)])))),
    }
}

/// Purpose: This function asks the process for its status.
///
/// Inputs:
/// - `p`: the process.
///
/// Outputs: the [`Status`], or a [`HostError`].
pub(crate) fn status(
    p: &Process,
) -> Result<Status, HostError>
{
    match ask(p, Request::Status)? {
        (Reply::Status(s), _) => Ok(*s),
        (other, _) => Err(unexpected("status", &other)),
    }
}

/// Purpose: This function gets a model and adds it to the process.
///
/// Inputs:
/// - `p`: the process.
/// - `model`: a GGUF file, or a Hugging Face link to a repo or to one file in it.
/// - `opts`: the add options.
///
/// Outputs: the model's status once it is in memory, or a [`HostError`].
pub(crate) fn add(
    p: &Process,
    model: &str,
    opts: &AddOptions,
) -> Result<ModelStatus, HostError>
{
    // Asking first means a stopped process is reported before anything is downloaded.
    status(p)?;
    let source = ModelSource::parse(model).map_err(|e| HostError::Model(e.to_string()))?;
    let fetched = hf_hub::fetch(&source, opts.want, &opts.fetch).map_err(|e| HostError::Model(e.to_string()))?;
    // A model from the Hub finds its tokenizer through its repo too, which a later step
    // given only the files cannot.
    let tokenizer = match &opts.tokenizer {
        Some(t) => t.clone(),
        None => {
            let gguf = crate::load::Model::from_parts(&fetched.parts).map_err(|e| HostError::Model(e.to_string()))?;
            let origin = source.origin(fetched.revision.as_deref());
            crate::tokenize::find(&fetched.parts[0], gguf.meta(), origin.as_ref(), &opts.fetch).map_err(|e| HostError::Model(e.to_string()))?
        }
    };
    add_parts(p, &fetched.parts, &AddOptions { tokenizer: Some(tokenizer), ..opts.clone() })
}

/// Purpose: This function adds a model already on disk to the process, and waits until
/// it is in memory.
///
/// Inputs:
/// - `p`: the process.
/// - `parts`: every part file of the model, part 1 first.
/// - `opts`: the add options.
///
/// Outputs: the model's status once it is in memory, or a [`HostError`].
pub(crate) fn add_parts(
    p: &Process,
    parts: &[PathBuf],
    opts: &AddOptions,
) -> Result<ModelStatus, HostError>
{
    let spec = spec(parts, opts)?;
    let name = spec.name.clone();
    match ask(p, Request::Add { spec })? {
        (Reply::Added, _) => {}
        (other, _) => return Err(unexpected("add", &other)),
    }
    loop {
        let s = status(p)?;
        match s.model(&name) {
            Some(m) if m.state == State::Ready => return Ok(m.clone()),
            Some(_) => std::thread::sleep(LOAD_POLL),
            None => return Err(HostError::Refused(format!("{name} was removed before it finished loading"))),
        }
    }
}

/// Purpose: This function removes a model from the process.
///
/// Inputs:
/// - `p`: the process.
/// - `name`: the model's name.
///
/// Outputs: `Ok(())`, or a [`HostError`].
pub(crate) fn remove(
    p: &Process,
    name: &str,
) -> Result<(), HostError>
{
    match ask(p, Request::Remove { name: name.to_owned() })? {
        (Reply::Removed, _) => Ok(()),
        (other, _) => Err(unexpected("remove", &other)),
    }
}

/// Purpose: This function asks the process for one model's whole plan.
///
/// Inputs:
/// - `p`: the process.
/// - `name`: the model's name.
///
/// Outputs: the plan as JSON, or a [`HostError`].
pub(crate) fn plan(
    p: &Process,
    name: &str,
) -> Result<serde_json::Value, HostError>
{
    match ask(p, Request::Plan { name: name.to_owned() })? {
        (Reply::Plan(v), _) => Ok(v),
        (other, _) => Err(unexpected("plan", &other)),
    }
}

/// Purpose: This function builds one model's tokenizer from its plan in the process.
///
/// Inputs:
/// - `p`: the process.
/// - `name`: the model's name.
/// - `table_hash`: the table hash of the plan this program was compiled against.
///
/// Outputs: the tokenizer, or a [`HostError`].
#[cfg(feature = "tokenize")]
pub(crate) fn tokenizer(
    p: &Process,
    name: &str,
    table_hash: &str,
) -> Result<crate::tokenize::Tokenizer, HostError>
{
    let plan = plan(p, name)?;
    let held = plan["table_hash"].as_str().unwrap_or_default();
    if held != table_hash {
        return Err(HostError::Refused(format!(
            "dovata holds {name} with a plan whose table hash is {held}, and the program comes from a plan with table hash {table_hash}"
        )));
    }
    let t: crate::tokenize::TokenizerPlan = serde_json::from_value(plan["tokenizer"].clone())
        .map_err(|e| HostError::Protocol(format!("{name}'s plan has no tokenizer dovata can read: {e}")))?;
    crate::tokenize::Tokenizer::from_plan(&t).map_err(|e| HostError::Model(e.to_string()))
}

/// Purpose: This function maps one model's memory into this program.
///
/// Inputs:
/// - `p`: the process.
/// - `name`: the model's name.
/// - `table_hash`: the table hash of the plan this program's offsets came from.
///
/// Outputs: the [`Mapping`], or a [`HostError`].
pub(crate) fn map(
    p: &Process,
    name: &str,
    table_hash: &str,
) -> Result<Mapping, HostError>
{
    let (reply, fds) = ask(p, Request::Map { name: name.to_owned(), table_hash: table_hash.to_owned() })?;
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

/// Purpose: This function stops the process and waits until it has gone.
///
/// Inputs:
/// - `p`: the process.
///
/// Outputs: `Ok(())` once nothing answers on its socket, or a [`HostError`].
pub(crate) fn kill(
    p: &Process,
) -> Result<(), HostError>
{
    match ask(p, Request::Stop)? {
        (Reply::Stopping, _) => {}
        (other, _) => return Err(unexpected("stop", &other)),
    }
    let start = Instant::now();
    while start.elapsed() < STOP_WAIT {
        if UnixStream::connect(&p.socket).is_err() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    Err(HostError::Io(format!("dovata said it would stop, and still answers after {} s", STOP_WAIT.as_secs())))
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
        Reply::Added => "an add",
        Reply::Removed => "a remove",
        Reply::Plan(_) => "a plan",
        Reply::Map(_) => "a map",
        Reply::Stopping => "a stop",
        Reply::Error(_) => "an error",
    };
    HostError::Protocol(format!("asked for {asked}, and got {kind}"))
}
