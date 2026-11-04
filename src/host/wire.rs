//! Author: Joe Ricks
//! File: src/host/wire.rs
//! Description: Sends and receives one line of JSON over a Unix socket, with file
//! descriptors attached when a reply carries memory, and finds the process at the other
//! end.

use std::io;
use std::mem::{size_of, zeroed};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::UnixStream;

/// The longest line either side reads. A plan's JSON for a large model is a few MiB.
pub(crate) const MAX_LINE: usize = 256 << 20;

/// The most file descriptors one reply carries: one per part file and one per
/// allocated region.
pub(crate) const MAX_FDS: usize = 64;

/// Purpose: This function gives the flags that keep a write to a closed socket from
/// stopping the program with SIGPIPE.
///
/// Inputs: none.
///
/// Outputs: `MSG_NOSIGNAL` on Linux. macOS has no such flag; [`quiet`] sets the socket
/// option instead.
fn send_flags() -> libc::c_int
{
    #[cfg(target_os = "linux")]
    return libc::MSG_NOSIGNAL;
    #[cfg(not(target_os = "linux"))]
    0
}

/// Purpose: This function stops writes to a closed socket from raising SIGPIPE on
/// systems with no per-write flag for it. A program that links dovata may not ignore
/// SIGPIPE the way Rust programs do.
///
/// Inputs:
/// - `sock`: the socket.
///
/// Outputs: none.
pub(crate) fn quiet(
    sock: &UnixStream,
)
{
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
    {
        let one: libc::c_int = 1;
        // SAFETY: the option value is a c_int that lives through the call.
        unsafe {
            libc::setsockopt(sock.as_raw_fd(), libc::SOL_SOCKET, libc::SO_NOSIGPIPE, (&one as *const libc::c_int).cast(), size_of::<libc::c_int>() as libc::socklen_t)
        };
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "freebsd")))]
    let _ = sock;
}

/// Purpose: This function sends bytes, with file descriptors attached to the first of
/// them.
///
/// Inputs:
/// - `sock`: the socket.
/// - `bytes`: the bytes, usually one JSON line.
/// - `fds`: the descriptors to send; the other side gets its own copies.
///
/// Outputs: `Ok(())` once every byte is sent, or the system's error.
pub(crate) fn send(
    sock: &UnixStream,
    bytes: &[u8],
    fds: &[RawFd],
) -> io::Result<()>
{
    if fds.len() > MAX_FDS {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{} file descriptors are more than {MAX_FDS}", fds.len())));
    }
    let mut sent = 0usize;
    let mut first = true;
    while sent < bytes.len() || first {
        let rest = &bytes[sent..];
        let mut iov = libc::iovec { iov_base: rest.as_ptr().cast_mut().cast(), iov_len: rest.len() };
        // A u64 buffer, so the control message header is aligned.
        let mut control = [0u64; 1 + (MAX_FDS * size_of::<libc::c_int>() + 64) / 8];
        // SAFETY: a zeroed msghdr is a valid empty one.
        let mut msg: libc::msghdr = unsafe { zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        if first && !fds.is_empty() {
            let data_len = std::mem::size_of_val(fds) as libc::c_uint;
            // SAFETY: CMSG_SPACE and CMSG_LEN only compute sizes.
            let space = unsafe { libc::CMSG_SPACE(data_len) } as usize;
            msg.msg_control = control.as_mut_ptr().cast();
            msg.msg_controllen = space as _;
            // SAFETY: the control buffer holds `space` bytes, room for one header and
            // `fds`, so the first header and its data lie inside it.
            unsafe {
                let c = libc::CMSG_FIRSTHDR(&msg);
                (*c).cmsg_level = libc::SOL_SOCKET;
                (*c).cmsg_type = libc::SCM_RIGHTS;
                (*c).cmsg_len = libc::CMSG_LEN(data_len) as _;
                std::ptr::copy_nonoverlapping(fds.as_ptr(), libc::CMSG_DATA(c).cast::<libc::c_int>(), fds.len());
            }
        }
        // SAFETY: `msg` points at `iov` and `control`, which live through the call.
        let n = unsafe { libc::sendmsg(sock.as_raw_fd(), &msg, send_flags()) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        first = false;
        sent += n as usize;
    }
    Ok(())
}

/// Purpose: This function reads one line, and every file descriptor sent with it.
///
/// Inputs:
/// - `sock`: the socket. The other side sends one line and then closes, or waits.
///
/// Outputs: the line without its newline, and the descriptors; or the system's error, or
/// an error when the line is longer than [`MAX_LINE`] or the other side closed first.
pub(crate) fn recv(
    sock: &UnixStream,
) -> io::Result<(Vec<u8>, Vec<OwnedFd>)>
{
    let mut line = Vec::new();
    let mut fds = Vec::new();
    let mut buf = vec![0u8; 64 << 10];
    loop {
        let mut iov = libc::iovec { iov_base: buf.as_mut_ptr().cast(), iov_len: buf.len() };
        let mut control = [0u64; 1 + (MAX_FDS * size_of::<libc::c_int>() + 64) / 8];
        // SAFETY: a zeroed msghdr is a valid empty one.
        let mut msg: libc::msghdr = unsafe { zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.as_mut_ptr().cast();
        msg.msg_controllen = std::mem::size_of_val(&control) as _;
        #[cfg(target_os = "linux")]
        let flags = libc::MSG_CMSG_CLOEXEC;
        #[cfg(not(target_os = "linux"))]
        let flags = 0;
        // SAFETY: `msg` points at `iov`, `buf` and `control`, which live through the call.
        let n = unsafe { libc::recvmsg(sock.as_raw_fd(), &mut msg, flags) };
        if n < 0 {
            let e = io::Error::last_os_error();
            if e.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(e);
        }
        take_fds(&msg, &mut fds)?;
        if msg.msg_flags & libc::MSG_CTRUNC != 0 {
            return Err(io::Error::other("the file descriptors sent did not fit in the buffer"));
        }
        let n = n as usize;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "the other side closed before it finished a line"));
        }
        line.extend_from_slice(&buf[..n]);
        if let Some(end) = line.iter().position(|b| *b == b'\n') {
            line.truncate(end);
            return Ok((line, fds));
        }
        if line.len() > MAX_LINE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("a line longer than {MAX_LINE} bytes")));
        }
    }
}

/// Purpose: This function takes ownership of every file descriptor in a received
/// message's control data.
///
/// Inputs:
/// - `msg`: the message recvmsg filled in.
/// - `fds`: where to put the descriptors.
///
/// Outputs: `Ok(())`, or the system's error setting close-on-exec.
fn take_fds(
    msg: &libc::msghdr,
    fds: &mut Vec<OwnedFd>,
) -> io::Result<()>
{
    // The end of the control bytes the kernel wrote. macOS keeps a truncated header's
    // full `cmsg_len`, which can claim more descriptors than arrived, so the count is
    // bounded by this end too.
    let end = (msg.msg_control as usize).saturating_add(msg.msg_controllen as usize);
    // SAFETY: `msg` was filled in by recvmsg, so its first header is the system's own.
    // Each descriptor read lies before `end`, inside the control buffer, and CMSG_NXTHDR
    // returns null rather than a header past `end`.
    unsafe {
        let mut c = libc::CMSG_FIRSTHDR(msg);
        while !c.is_null() {
            if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                let data = libc::CMSG_DATA(c);
                let claimed = ((*c).cmsg_len as usize).saturating_sub(libc::CMSG_LEN(0) as usize);
                let present = end.saturating_sub(data as usize);
                let n = claimed.min(present) / size_of::<libc::c_int>();
                for i in 0..n {
                    let fd: libc::c_int = std::ptr::read_unaligned(data.cast::<libc::c_int>().add(i));
                    fds.push(OwnedFd::from_raw_fd(fd));
                }
            }
            c = libc::CMSG_NXTHDR(msg, c);
        }
    }
    // Linux sets close-on-exec as it receives them; elsewhere it is set here, so a
    // program this one starts does not inherit them.
    #[cfg(not(target_os = "linux"))]
    for fd in fds.iter() {
        // SAFETY: `fd` is a descriptor this function owns.
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Purpose: This function finds the process at the other end of a socket.
///
/// Inputs:
/// - `sock`: a connected socket.
///
/// Outputs: the process id, or `None` when the system cannot say.
pub(crate) fn peer_pid(
    sock: &UnixStream,
) -> Option<u32>
{
    #[cfg(target_os = "macos")]
    {
        let mut pid: libc::pid_t = 0;
        let mut len = size_of::<libc::pid_t>() as libc::socklen_t;
        // SAFETY: the option value is a pid_t, and `len` says so.
        let rc = unsafe { libc::getsockopt(sock.as_raw_fd(), libc::SOL_LOCAL, libc::LOCAL_PEERPID, (&mut pid as *mut libc::pid_t).cast(), &mut len) };
        (rc == 0).then(|| u32::try_from(pid).ok()).flatten()
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: a zeroed ucred is a valid one.
        let mut cred: libc::ucred = unsafe { zeroed() };
        let mut len = size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY: the option value is a ucred, and `len` says so.
        let rc = unsafe { libc::getsockopt(sock.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&mut cred as *mut libc::ucred).cast(), &mut len) };
        (rc == 0).then(|| u32::try_from(cred.pid).ok()).flatten()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = sock;
        None
    }
}
