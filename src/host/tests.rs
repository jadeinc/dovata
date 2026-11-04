//! Author: Joe Ricks
//! File: src/host/tests.rs
//! Description: Runs a dovata process on a thread of the test, on a tiny model, and
//! checks every request against the model file and the plan.

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

use super::client::attach_at;
use super::server::serve_in;
use super::{HostError, Hosted, Reply, Spec, State};

#[path = "fixture.rs"]
mod fixture;

/// Purpose: This struct holds a dovata process running on a thread of the test.
struct Running {
    /// The process, as a program talks to it.
    hosted: Hosted,
    /// The thread; it returns what `serve_in` returned.
    thread: JoinHandle<Result<(), HostError>>,
    /// The model file.
    model: PathBuf,
    /// The run directory and the model, removed at the end.
    _dir: fixture::TempDir,
}

/// Purpose: This function gives the step the tests plan for: small, so the fixture's
/// regions are small.
///
/// Inputs: none.
///
/// Outputs: the step.
fn step() -> BindOptions
{
    BindOptions { n_ctx: 64, n_tokens: 8, n_outputs: 1, kv_k: KvType::F16, kv_v: KvType::F16 }
}

/// Purpose: This function starts a dovata process on a thread, on the fixture model, and
/// waits until it is ready.
///
/// Inputs:
/// - `name`: the process's name.
///
/// Outputs: the [`Running`] process.
fn run(
    name: &str,
) -> Running
{
    let dir = fixture::TempDir::new();
    let model = dir.path.join("host-fixture.gguf");
    fixture::write_llama(&model);
    let spec = Spec { name: name.into(), parts: vec![model.clone()], layout: None, step: step(), cpu: false };
    let (tx, rx) = mpsc::channel();
    let run_dir = dir.path.clone();
    let thread = std::thread::spawn(move || serve_in(&spec, &run_dir, move |r| tx.send(r.clone()).is_ok()));
    let ready = match rx.recv_timeout(Duration::from_secs(60)) {
        Ok(r) => r,
        Err(_) => panic!("not ready: {:?}", thread.join()),
    };
    assert_eq!((ready.name.as_str(), ready.pid), (name, std::process::id()));
    Running { hosted: Hosted { name: ready.name, socket: ready.socket }, thread, model, _dir: dir }
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
    r.hosted.kill().unwrap();
    assert_eq!(r.thread.join().unwrap(), Ok(()));
    assert!(!r.hosted.socket().exists(), "the socket file is removed");
    assert!(matches!(attach_at(r.hosted.name(), r.hosted.socket()), Err(HostError::NotRunning(_))));
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
fn a_process_holds_the_model_and_a_program_maps_it()
{
    let r = run("fixture");
    let h = &r.hosted;

    let s = h.status().unwrap();
    assert_eq!(s.state, State::Ready);
    assert_eq!(s.model.architecture, "llama");
    let size = std::fs::metadata(&r.model).unwrap().len();
    assert_eq!(s.files.len(), 1);
    assert_eq!(s.files[0].bytes, size);
    assert!(s.files[0].resident_bytes > 0, "the model was read into memory: {:?}", s.files[0]);
    assert!(s.regions.iter().any(|x| x.kind == RegionKind::Mmap) && s.allocated().count() >= 1, "{:?}", s.regions);
    for a in s.allocated() {
        assert_eq!(a.resident_bytes, a.bytes, "an allocated region is touched while loading: {a:?}");
    }
    assert!(s.clients.is_empty());

    let plan = h.plan().unwrap();
    assert_eq!(plan["table_hash"].as_str(), Some(s.table_hash.as_str()));

    // A program whose offsets came from another plan is refused.
    let wrong = h.map(&"0".repeat(64));
    assert!(matches!(&wrong, Err(HostError::Refused(m)) if m.contains("table hash")), "{wrong:?}");

    // Every weight is at its planned address, with the file's bytes.
    let m = h.map(&s.table_hash).unwrap();
    assert_eq!(m.table_hash(), s.table_hash);
    assert_eq!(m.regions().len(), s.regions.len());
    let mut weights = 0;
    for e in plan["entries"].as_array().unwrap() {
        if e["role"]["kind"] != "weight" {
            continue;
        }
        let region = m.region(n(&e["region"]) as u16).unwrap();
        let got = region.read(n(&e["offset"]), n(&e["bytes"])).unwrap();
        assert_eq!(got, file_bytes(&r.model, n(&e["source"]["file_offset"]), n(&e["bytes"])), "{}", e["tensor"]);
        weights += 1;
    }
    assert!(weights > 10, "{weights} weights");

    // Weights are read-only; allocated regions are shared between every mapping.
    let w = m.regions().iter().find(|x| x.kind() == RegionKind::Mmap).unwrap();
    assert!(!w.writable() && w.as_mut_ptr().is_none() && !w.write(0, &[1]));
    let id = m.regions().iter().find(|x| x.writable()).unwrap().id();
    let other = h.map(&s.table_hash).unwrap();
    let pattern: Vec<u8> = (0..=255u8).collect();
    assert!(other.region(id).unwrap().write(100, &pattern));
    assert_eq!(m.region(id).unwrap().read(100, 256).unwrap(), pattern, "a write through one mapping shows in the other");
    let a = m.region(id).unwrap();
    assert!(a.read(a.len() - 1, 2).is_none() && !a.write(a.len(), &[0]), "reads and writes stay inside the region");

    let s = h.status().unwrap();
    assert_eq!(s.clients.len(), 2);
    assert!(s.clients.iter().all(|c| c.pid == std::process::id() && c.alive), "{:?}", s.clients);

    stop(r);
    // A mapping outlives the process.
    assert_eq!(m.region(id).unwrap().read(100, 4).unwrap(), &pattern[..4]);
}

#[test]
fn a_name_holds_one_process_and_a_stale_socket_is_replaced()
{
    let r = run("only-one");
    let spec = Spec { name: "only-one".into(), parts: vec![r.model.clone()], layout: None, step: step(), cpu: false };
    let dir = r.hosted.socket().parent().unwrap().to_path_buf();
    assert_eq!(serve_in(&spec, &dir, |_| true), Err(HostError::AlreadyRunning("only-one".into())));
    stop(r);

    // A process that was killed leaves its socket file behind. Nothing answers on it,
    // so a program finds no process there, and a new process takes the name.
    let dir = fixture::TempDir::new();
    let socket = dir.path.join("stale.sock");
    drop(UnixListener::bind(&socket).unwrap());
    assert!(socket.exists());
    assert!(matches!(attach_at("stale", &socket), Err(HostError::NotRunning(_))));
    assert!(!socket.exists(), "the stale socket file is removed");
    drop(UnixListener::bind(&socket).unwrap());
    let model = dir.path.join("m.gguf");
    fixture::write_llama(&model);
    let spec = Spec { name: "stale".into(), parts: vec![model], layout: None, step: step(), cpu: false };
    let (tx, rx) = mpsc::channel();
    let run_dir = dir.path.clone();
    let thread = std::thread::spawn(move || serve_in(&spec, &run_dir, move |r| tx.send(r.clone()).is_ok()));
    let ready = rx.recv_timeout(Duration::from_secs(60)).unwrap();
    stop(Running { hosted: Hosted { name: ready.name, socket: ready.socket }, thread, model: PathBuf::new(), _dir: dir });
}

#[test]
fn a_request_of_another_protocol_is_refused()
{
    let r = run("protocol");
    let mut s = UnixStream::connect(r.hosted.socket()).unwrap();
    s.write_all(b"{\"protocol\":99,\"op\":\"status\"}\n").unwrap();
    let (line, _) = super::wire::recv(&s).unwrap();
    let reply: Reply = serde_json::from_slice(&line).unwrap();
    assert!(matches!(&reply, Reply::Error(m) if m.contains("protocol 1") && m.contains("protocol 99")), "{reply:?}");
    let mut s = UnixStream::connect(r.hosted.socket()).unwrap();
    s.write_all(b"{\"protocol\":1,\"op\":\"launch\"}\n").unwrap();
    let (line, _) = super::wire::recv(&s).unwrap();
    assert!(matches!(serde_json::from_slice::<Reply>(&line).unwrap(), Reply::Error(_)));
    stop(r);
}

#[test]
fn a_start_nobody_waits_for_stops_the_process()
{
    let dir = fixture::TempDir::new();
    let model = dir.path.join("m.gguf");
    fixture::write_llama(&model);
    let spec = Spec { name: "abandoned".into(), parts: vec![model], layout: None, step: step(), cpu: false };
    // The ready callback says nobody is listening, as when `dovata start` was stopped
    // with Ctrl-C while the model loaded.
    assert_eq!(serve_in(&spec, &dir.path, |_| false), Ok(()));
    assert!(!dir.path.join("abandoned.sock").exists());
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
    let r = run("many-fds");
    let s = UnixStream::connect(r.hosted.socket()).unwrap();
    send_many_fds(&s, b"{\"protocol\":1,\"op\":\"status\"}\n", 100);
    let _ = super::wire::recv(&s);
    drop(s);
    assert_eq!(r.hosted.status().unwrap().state, State::Ready);
    stop(r);
}

#[test]
fn a_model_changed_on_disk_is_not_handed_out()
{
    let r = run("changed");
    let hash = r.hosted.status().unwrap().table_hash;
    drop(r.hosted.map(&hash).unwrap());
    // Appending changes the size and the time; the bytes already mapped stay valid.
    std::fs::OpenOptions::new().append(true).open(&r.model).unwrap().write_all(b"more").unwrap();
    let refused = r.hosted.map(&hash);
    assert!(matches!(&refused, Err(HostError::Refused(m)) if m.contains("changed on disk")), "{refused:?}");
    stop(r);
}
