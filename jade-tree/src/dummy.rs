use jade_protocol::{Frame, InferenceRequest};
use std::io::{Read, Write};
use std::os::unix::net::UnixListener;

pub fn run(sock_path: String) {
    eprintln!("jade-tree v{} [dummy — Unix socket, canned responses]", env!("CARGO_PKG_VERSION"));

    std::fs::create_dir_all(std::path::Path::new(&sock_path).parent().unwrap())
        .expect("create socket dir");
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("jade-tree: could not bind {sock_path}: {e}");
        std::process::exit(1);
    });

    eprintln!("jade-tree: listening on {sock_path} [dummy]");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream) {
                    eprintln!("jade-tree: connection error: {e}");
                }
            }
            Err(e) => eprintln!("jade-tree: accept error: {e}"),
        }
    }
}

fn handle_connection(stream: &mut std::os::unix::net::UnixStream) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];

    let req = loop {
        match InferenceRequest::decode(&buf) {
            Ok(req) => {
                let payload_len =
                    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                buf.drain(..4 + payload_len);
                break req;
            }
            Err(jade_protocol::RequestDecodeError::Incomplete) => {
                let n = stream.read(&mut tmp)?;
                if n == 0 {
                    return Ok(());
                }
                buf.extend_from_slice(&tmp[..n]);
            }
            Err(e) => {
                eprintln!("jade-tree: malformed request: {e:?}");
                return Ok(());
            }
        }
    };

    let preview = req.prompt.chars().take(60).collect::<String>();
    eprintln!("jade-tree: request — prompt={preview:?}");

    let response = respond(&req.prompt);
    stream.write_all(&Frame::Token(response.to_owned()).encode())?;
    stream.write_all(&Frame::Done { tokens_used: 1 }.encode())?;
    Ok(())
}

fn respond(prompt: &str) -> &'static str {
    let s = prompt.to_lowercase();
    if s.contains("hello") || s.contains("hi") {
        "Hello! I am JADE OS [dummy mode — no model loaded]."
    } else if s.contains("status") || s.contains("health") {
        "nominal"
    } else if s.contains("list") || s.contains("show") {
        "(empty)"
    } else if s.contains("run") || s.contains("exec") {
        "started"
    } else if s.contains("stop") || s.contains("kill") {
        "stopped"
    } else {
        "ok [dummy]"
    }
}
