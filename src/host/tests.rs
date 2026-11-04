//! Author: Joe Ricks
//! File: src/host/tests.rs
//! Description: Runs the dovata process on a thread of the test, adds tiny models to it,
//! and checks every request against the model file and the plan.

use std::io::Write as _;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::gguf::write::Writer;
use crate::gguf::Value;
use crate::layout::BindOptions;
use crate::load::KvType;
use crate::target::RegionKind;

use super::server::serve_in;
use super::{AddOptions, HostError, Process, Reply, State};

#[path = "fixture.rs"]
mod fixture;

/// Purpose: This struct holds a dovata process running on a thread of the test.
struct Running {
    /// The process, as a program talks to it.
    process: Process,
    /// The thread; it returns what `serve_in` returned.
    thread: JoinHandle<Result<(), HostError>>,
    /// The fixture model's file.
    model: PathBuf,
    /// The run directory, which also holds the model; removed at the end.
    dir: fixture::TempDir,
}

/// Purpose: This function gives the add options the tests use: a small step, so the
/// fixture's regions are small.
///
/// Inputs:
/// - `name`: the model's name.
///
/// Outputs: the options.
fn opts(
    name: &str,
) -> AddOptions
{
    let step = BindOptions { n_ctx: 64, n_tokens: 8, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 };
    AddOptions { name: Some(name.into()), step, ..AddOptions::default() }
}

/// Purpose: This function starts the dovata process on a thread, with a fixture model
/// written beside it but not yet added, and waits until it answers.
///
/// Inputs: none.
///
/// Outputs: the [`Running`] process.
fn run() -> Running
{
    let dir = fixture::TempDir::new();
    let model = dir.path.join("host-fixture.gguf");
    fixture::write_llama(&model);
    let (tx, rx) = mpsc::channel();
    let run_dir = dir.path.clone();
    let thread = std::thread::spawn(move || serve_in(&run_dir, move |r| tx.send(r.clone()).is_ok()));
    let ready = match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(r) => r,
        Err(_) => panic!("not ready: {:?}", thread.join()),
    };
    assert_eq!(ready.pid, std::process::id());
    Running { process: Process { socket: ready.socket }, thread, model, dir }
}

/// Purpose: This function stops a process the test started and checks it stopped
/// cleanly.
///
/// Inputs:
/// - `r`: the process.
///
/// Outputs: none.
fn stop(
    r: Running,
)
{
    r.process.kill().unwrap();
    assert_eq!(r.thread.join().unwrap(), Ok(()));
    assert!(!r.process.socket().exists(), "the socket file is removed");
    assert_eq!(r.process.status(), Err(HostError::NotRunning));
}

/// Purpose: This function reads a JSON number as a usize.
///
/// Inputs:
/// - `v`: the value.
///
/// Outputs: the number.
fn n(
    v: &serde_json::Value,
) -> usize
{
    v.as_u64().unwrap_or_else(|| panic!("{v} is not a number")) as usize
}

/// Purpose: This function reads part of a file.
///
/// Inputs:
/// - `path`: the file.
/// - `offset`: where to start.
/// - `len`: how many bytes.
///
/// Outputs: the bytes.
fn file_bytes(
    path: &Path,
    offset: usize,
    len: usize,
) -> Vec<u8>
{
    std::fs::read(path).unwrap()[offset..offset + len].to_vec()
}

#[test]
fn the_process_starts_empty_and_a_program_maps_a_model_added_to_it()
{
    let r = run();
    let p = &r.process;
    let s = p.status().unwrap();
    assert_eq!(s.pid, std::process::id());
    assert!(s.models.is_empty());

    let m = p.add_parts(std::slice::from_ref(&r.model), &opts("fixture")).unwrap();
    assert_eq!((m.name.as_str(), m.state), ("fixture", State::Ready));
    assert_eq!(m.model.architecture, "llama");
    let size = std::fs::metadata(&r.model).unwrap().len();
    assert_eq!(m.files.len(), 1);
    assert_eq!(m.files[0].bytes, size);
    assert!(m.files[0].resident_bytes > 0, "the model was read into memory: {:?}", m.files[0]);
    assert!(m.regions.iter().any(|x| x.kind == RegionKind::Mmap) && m.allocated().count() >= 1, "{:?}", m.regions);
    for a in m.allocated() {
        assert_eq!(a.resident_bytes, a.bytes, "an allocated region is touched while loading: {a:?}");
    }
    assert!(m.clients.is_empty());

    let plan = p.plan("fixture").unwrap();
    assert_eq!(plan["table_hash"].as_str(), Some(m.table_hash.as_str()));
    // The process plans a model exactly as the compiler does for this machine, so a
    // program compiled against `dovata plan` maps it.
    let gguf = crate::load::Model::open(&r.model).unwrap();
    let layout = crate::layout::for_model(&gguf, None).unwrap();
    let target = crate::machine::current().unwrap().target(false);
    let direct = crate::memplan::plan(&gguf, &layout, &opts("x").step, &target).unwrap();
    assert_eq!(direct.table_hash, m.table_hash);
    assert!(matches!(p.plan("nothing"), Err(HostError::Refused(e)) if e.contains("no model named nothing") && e.contains("fixture")));

    // A program whose offsets came from another plan is refused.
    let wrong = p.map("fixture", &"0".repeat(64));
    assert!(matches!(&wrong, Err(HostError::Refused(e)) if e.contains("table hash")), "{wrong:?}");

    // Every weight is at its planned address, with the file's bytes.
    let mem = p.map("fixture", &m.table_hash).unwrap();
    assert_eq!(mem.table_hash(), m.table_hash);
    assert_eq!(mem.regions().len(), m.regions.len());
    let mut weights = 0;
    for e in plan["entries"].as_array().unwrap() {
        if e["role"]["kind"] != "weight" {
            continue;
        }
        let region = mem.region(n(&e["region"]) as u16).unwrap();
        let got = region.read(n(&e["offset"]), n(&e["bytes"])).unwrap();
        assert_eq!(got, file_bytes(&r.model, n(&e["source"]["file_offset"]), n(&e["bytes"])), "{}", e["tensor"]);
        weights += 1;
    }
    assert!(weights > 10, "{weights} weights");

    // Weights are read-only; allocated regions are shared between every mapping.
    let w = mem.regions().iter().find(|x| x.kind() == RegionKind::Mmap).unwrap();
    assert!(!w.writable() && w.as_mut_ptr().is_none() && !w.write(0, &[1]));
    let id = mem.regions().iter().find(|x| x.writable()).unwrap().id();
    let other = p.map("fixture", &m.table_hash).unwrap();
    let pattern: Vec<u8> = (0..=255u8).collect();
    assert!(other.region(id).unwrap().write(100, &pattern));
    assert_eq!(mem.region(id).unwrap().read(100, 256).unwrap(), pattern, "a write through one mapping shows in the other");
    let a = mem.region(id).unwrap();
    assert!(a.read(a.len() - 1, 2).is_none() && !a.write(a.len(), &[0]), "reads and writes stay inside the region");

    let s = p.status().unwrap();
    let clients = &s.model("fixture").unwrap().clients;
    assert_eq!(clients.len(), 2);
    assert!(clients.iter().all(|c| c.pid == std::process::id() && c.alive), "{clients:?}");

    stop(r);
    // A mapping outlives the process.
    assert_eq!(mem.region(id).unwrap().read(100, 4).unwrap(), &pattern[..4]);
}

#[test]
fn one_process_holds_several_models_and_each_name_once()
{
    let r = run();
    let p = &r.process;
    let a = p.add_parts(std::slice::from_ref(&r.model), &opts("a")).unwrap();
    let b = p.add_parts(std::slice::from_ref(&r.model), &opts("b")).unwrap();
    let s = p.status().unwrap();
    let names: Vec<&str> = s.models.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(names, ["a", "b"]);
    assert_eq!(s.bytes(), a.bytes() + b.bytes());

    let again = p.add_parts(std::slice::from_ref(&r.model), &opts("a"));
    assert!(matches!(&again, Err(HostError::Refused(e)) if e.contains("already holds a model named a")), "{again:?}");

    // Each model has memory of its own: a write to one's allocation is not in the other's.
    let ma = p.map("a", &a.table_hash).unwrap();
    let mb = p.map("b", &b.table_hash).unwrap();
    let id = ma.regions().iter().find(|x| x.writable()).unwrap().id();
    assert!(ma.region(id).unwrap().write(0, &[7, 7, 7]));
    assert_eq!(mb.region(id).unwrap().read(0, 3).unwrap(), [0, 0, 0]);

    // Removing a model takes it out of the process; a program's mapping stays.
    p.remove("a").unwrap();
    let s = p.status().unwrap();
    assert_eq!(s.models.len(), 1);
    assert!(s.model("a").is_none());
    assert!(matches!(p.map("a", &a.table_hash), Err(HostError::Refused(_))));
    assert!(matches!(p.remove("a"), Err(HostError::Refused(e)) if e.contains("no model named a")));
    assert_eq!(ma.region(id).unwrap().read(0, 3).unwrap(), [7, 7, 7]);
    // Its name is free again.
    p.add_parts(std::slice::from_ref(&r.model), &opts("a")).unwrap();
    stop(r);
}

#[test]
fn a_second_process_is_refused_and_a_stale_socket_is_replaced()
{
    let r = run();
    let second = serve_in(&r.dir.path, |_| true);
    assert_eq!(second, Err(HostError::AlreadyRunning(std::process::id())));
    stop(r);

    // A process that was killed leaves its socket file behind. Nothing answers on it,
    // and the next process replaces it.
    let dir = fixture::TempDir::new();
    let socket = dir.path.join("dovata.sock");
    drop(UnixListener::bind(&socket).unwrap());
    assert!(socket.exists());
    assert_eq!(Process { socket: socket.clone() }.status(), Err(HostError::NotRunning));
    let (tx, rx) = mpsc::channel();
    let run_dir = dir.path.clone();
    let thread = std::thread::spawn(move || serve_in(&run_dir, move |r| tx.send(r.clone()).is_ok()));
    let ready = rx.recv_timeout(Duration::from_secs(30)).unwrap();
    assert_eq!(ready.socket, socket);
    stop(Running { process: Process { socket }, thread, model: PathBuf::new(), dir });
}

#[test]
fn a_request_of_another_protocol_is_refused()
{
    let r = run();
    let mut s = UnixStream::connect(r.process.socket()).unwrap();
    s.write_all(b"{\"protocol\":99,\"op\":\"status\"}\n").unwrap();
    let (line, _) = super::wire::recv(&s).unwrap();
    let reply: Reply = serde_json::from_slice(&line).unwrap();
    assert!(matches!(&reply, Reply::Error(m) if m.contains("protocol 2") && m.contains("protocol 99")), "{reply:?}");
    let mut s = UnixStream::connect(r.process.socket()).unwrap();
    s.write_all(b"{\"protocol\":2,\"op\":\"launch\"}\n").unwrap();
    let (line, _) = super::wire::recv(&s).unwrap();
    assert!(matches!(serde_json::from_slice::<Reply>(&line).unwrap(), Reply::Error(_)));
    stop(r);
}

#[test]
fn a_start_nobody_waits_for_stops_the_process()
{
    let dir = fixture::TempDir::new();
    // The ready callback says nobody is listening, as when `dovata start` was stopped
    // with Ctrl-C before the process answered.
    assert_eq!(serve_in(&dir.path, |_| false), Ok(()));
    assert!(!dir.path.join("dovata.sock").exists());
}

#[test]
fn a_model_that_cannot_be_planned_is_refused_and_the_process_goes_on()
{
    let r = run();
    let layout = r.dir.path.join("bad.toml");
    std::fs::write(&layout, b"not a layout [").unwrap();
    let bad = AddOptions { layout: Some(layout), ..opts("bad") };
    let e = r.process.add_parts(std::slice::from_ref(&r.model), &bad);
    assert!(matches!(&e, Err(HostError::Refused(m)) if m.contains("bad.toml")), "{e:?}");
    assert!(r.process.status().unwrap().models.is_empty());
    r.process.add_parts(std::slice::from_ref(&r.model), &opts("good")).unwrap();
    stop(r);
}

/// Purpose: This function sends a line with many copies of one file descriptor
/// attached, more than `wire::send` allows, the way a hostile or broken program might.
///
/// Inputs:
/// - `sock`: the socket.
/// - `line`: the line.
/// - `copies`: how many descriptors to attach.
///
/// Outputs: none; it panics when the send fails.
fn send_many_fds(
    sock: &UnixStream,
    line: &[u8],
    copies: usize,
)
{
    use std::os::fd::AsRawFd;
    let fd = sock.as_raw_fd();
    let fds = vec![fd; copies];
    let data_len = std::mem::size_of_val(fds.as_slice()) as libc::c_uint;
    // SAFETY: CMSG_SPACE only computes a size.
    let space = unsafe { libc::CMSG_SPACE(data_len) } as usize;
    let mut control = vec![0u64; space.div_ceil(8)];
    let mut iov = libc::iovec { iov_base: line.as_ptr().cast_mut().cast(), iov_len: line.len() };
    // SAFETY: a zeroed msghdr is a valid empty one; the header and its data fit in
    // `control`, which is `space` bytes or more and aligned for a header.
    unsafe {
        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = space as _;
        let c = libc::CMSG_FIRSTHDR(&msg);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(data_len) as _;
        std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(c).cast::<libc::c_int>(), fds.len());
        assert!(libc::sendmsg(fd, &msg, 0) > 0, "{}", std::io::Error::last_os_error());
    }
}

#[test]
fn a_request_with_too_many_descriptors_does_not_harm_the_process()
{
    // More descriptors than the receiving buffer holds. macOS reports a truncated header
    // whose length still claims all of them; the process must read only those that
    // arrived, and go on answering.
    let r = run();
    let s = UnixStream::connect(r.process.socket()).unwrap();
    send_many_fds(&s, b"{\"protocol\":2,\"op\":\"status\"}\n", 100);
    let _ = super::wire::recv(&s);
    drop(s);
    assert!(r.process.status().is_ok());
    stop(r);
}

#[test]
fn a_model_changed_on_disk_is_not_handed_out()
{
    let r = run();
    let m = r.process.add_parts(std::slice::from_ref(&r.model), &opts("changed")).unwrap();
    drop(r.process.map("changed", &m.table_hash).unwrap());
    // Appending changes the size and the time; the bytes already mapped stay valid.
    std::fs::OpenOptions::new().append(true).open(&r.model).unwrap().write_all(b"more").unwrap();
    let refused = r.process.map("changed", &m.table_hash);
    assert!(matches!(&refused, Err(HostError::Refused(e)) if e.contains("changed on disk")), "{refused:?}");
    stop(r);
}
