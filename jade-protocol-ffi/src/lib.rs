//! `jade-protocol.so` — a C-ABI streaming client for the jade inference daemon.
//!
//! This is the single, shared implementation of "talk to the daemon" that the
//! native `use llm` package (and any C client) links against, instead of each
//! hand-rolling the length-prefixed request, the binary frame loop, and the
//! `<tool>…</tool>` parsing. The wire codec lives in the pure `jade-protocol`
//! crate (re-exported here as `protocol`); this crate adds the Unix-socket
//! transport, the tool-span parser ([`stream`]), and the C ABI.
//!
//! Lifecycle from C (see the generated `include/jade_protocol.h` for exact types):
//! ```c
//! JadeConn* c = jade_connect(NULL);                    // $HOME/.jade/llm.sock
//! JadeRequest* r = jade_request_messages();
//! jade_request_add_msg(r, "user", "weather in SF?");
//! jade_request_enable_tools(r, tools_gbnf);            // anchors + keep_anchors
//! jade_infer(c, r, &callbacks, userdata);              // streams events
//! jade_request_free(r);
//! jade_disconnect(c);
//! ```

// These are C-ABI entry points: callers pass raw pointers across the FFI boundary
// and are responsible for their validity (documented per function). Marking each
// `unsafe` would not change the generated C header, so we allow the lint crate-wide
// and keep the pointer-safety contract in the doc comments.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString};
use std::io::{Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::os::unix::net::UnixStream;

use protocol::{anchors, Frame, FrameError, InferenceRequest};

mod stream;
use stream::{Event, ToolStreamParser};

// ── Status codes ────────────────────────────────────────────────────────────

/// Inference completed and a Done frame was received.
pub const JADE_OK: c_int = 0;
/// A null/invalid argument was passed.
pub const JADE_ERR_ARGS: c_int = -1;
/// Socket I/O, encode, or frame-decode failure.
pub const JADE_ERR_IO: c_int = -2;
/// The daemon returned an Error frame (forwarded to `on_error`).
pub const JADE_ERR_DAEMON: c_int = -3;

// ── Callbacks ───────────────────────────────────────────────────────────────

/// Event callbacks for [`jade_infer`]. Any field may be null to ignore that event.
/// All pointers passed to callbacks are valid only for the duration of the call —
/// copy anything you need to retain. `on_token` text is NOT null-terminated; use `len`.
#[repr(C)]
pub struct JadeCallbacks {
    pub on_meta: Option<extern "C" fn(model: *const c_char, userdata: *mut c_void)>,
    pub on_token: Option<extern "C" fn(text: *const c_char, len: usize, userdata: *mut c_void)>,
    pub on_tool_call:
        Option<extern "C" fn(name: *const c_char, args: *const c_char, userdata: *mut c_void)>,
    pub on_done: Option<extern "C" fn(tokens_used: u64, userdata: *mut c_void)>,
    pub on_error: Option<extern "C" fn(msg: *const c_char, userdata: *mut c_void)>,
}

// ── Opaque handles ──────────────────────────────────────────────────────────

/// Opaque connection handle.
pub struct JadeConn {
    stream: UnixStream,
}

/// Opaque request builder. Either a plain single-prompt request or a
/// `_jade_protocol: messages` envelope, depending on the constructor used.
pub struct JadeRequest {
    inner: InferenceRequest,
    is_messages: bool,
    system: String,
    messages: Vec<(String, String)>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Borrow a C string as `&str`; `None` if null or not UTF-8.
unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

fn default_sock_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_owned());
    format!("{home}/.jade/llm.sock")
}

fn fresh_request() -> InferenceRequest {
    InferenceRequest {
        prompt: String::new(),
        model: String::new(),
        max_tokens: 512,
        grammar: None,
        anchor: None,
        stop_anchor: None,
        count_only: false,
        stats_only: false,
        keep_anchors: false,
        trust: protocol::request::trust::TRUSTED,
    }
}

/// Render the messages-envelope prompt string the daemon's `extract_jade_protocol` expects.
fn build_messages_prompt(system: &str, messages: &[(String, String)]) -> String {
    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
        .collect();
    serde_json::json!({
        "_jade_protocol": "messages",
        "system": system,
        "messages": msgs,
    })
    .to_string()
}

// ── Connection ──────────────────────────────────────────────────────────────

/// Connect to the daemon. `path` null → `$HOME/.jade/llm.sock`. Returns null on failure.
#[no_mangle]
pub extern "C" fn jade_connect(path: *const c_char) -> *mut JadeConn {
    let sock = match unsafe { cstr(path) } {
        Some(s) => s.to_owned(),
        None if path.is_null() => default_sock_path(),
        None => return std::ptr::null_mut(),
    };
    match UnixStream::connect(&sock) {
        Ok(stream) => Box::into_raw(Box::new(JadeConn { stream })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Close a connection and free it. Safe to call with null.
#[no_mangle]
pub extern "C" fn jade_disconnect(conn: *mut JadeConn) {
    if !conn.is_null() {
        drop(unsafe { Box::from_raw(conn) });
    }
}

// ── Request builders ────────────────────────────────────────────────────────

/// New plain single-prompt request (set the prompt with `jade_request_set_prompt`).
#[no_mangle]
pub extern "C" fn jade_request_new() -> *mut JadeRequest {
    Box::into_raw(Box::new(JadeRequest {
        inner: fresh_request(),
        is_messages: false,
        system: String::new(),
        messages: Vec::new(),
    }))
}

/// New multi-turn request using the `_jade_protocol: messages` envelope.
#[no_mangle]
pub extern "C" fn jade_request_messages() -> *mut JadeRequest {
    Box::into_raw(Box::new(JadeRequest {
        inner: fresh_request(),
        is_messages: true,
        system: String::new(),
        messages: Vec::new(),
    }))
}

/// Free a request builder. Safe to call with null.
#[no_mangle]
pub extern "C" fn jade_request_free(req: *mut JadeRequest) {
    if !req.is_null() {
        drop(unsafe { Box::from_raw(req) });
    }
}

/// Set the prompt of a plain request. No-op on a messages request.
#[no_mangle]
pub extern "C" fn jade_request_set_prompt(req: *mut JadeRequest, prompt: *const c_char) -> c_int {
    let (Some(r), Some(p)) = (unsafe { req.as_mut() }, unsafe { cstr(prompt) }) else {
        return JADE_ERR_ARGS;
    };
    r.inner.prompt = p.to_owned();
    JADE_OK
}

/// Set the model name.
#[no_mangle]
pub extern "C" fn jade_request_set_model(req: *mut JadeRequest, model: *const c_char) -> c_int {
    let (Some(r), Some(m)) = (unsafe { req.as_mut() }, unsafe { cstr(model) }) else {
        return JADE_ERR_ARGS;
    };
    r.inner.model = m.to_owned();
    JADE_OK
}

/// Set the system prompt (messages requests only).
#[no_mangle]
pub extern "C" fn jade_request_set_system(req: *mut JadeRequest, system: *const c_char) -> c_int {
    let (Some(r), Some(s)) = (unsafe { req.as_mut() }, unsafe { cstr(system) }) else {
        return JADE_ERR_ARGS;
    };
    r.system = s.to_owned();
    JADE_OK
}

/// Append a message (`role`, `content`) to a messages request.
#[no_mangle]
pub extern "C" fn jade_request_add_msg(
    req: *mut JadeRequest,
    role: *const c_char,
    content: *const c_char,
) -> c_int {
    let (Some(r), Some(role), Some(content)) =
        (unsafe { req.as_mut() }, unsafe { cstr(role) }, unsafe { cstr(content) })
    else {
        return JADE_ERR_ARGS;
    };
    r.messages.push((role.to_owned(), content.to_owned()));
    JADE_OK
}

/// Set the maximum number of tokens to generate.
#[no_mangle]
pub extern "C" fn jade_request_set_max_tokens(req: *mut JadeRequest, max_tokens: u32) -> c_int {
    match unsafe { req.as_mut() } {
        Some(r) => {
            r.inner.max_tokens = max_tokens;
            JADE_OK
        }
        None => JADE_ERR_ARGS,
    }
}

/// Mark the prompt as TAINTED (LLM/network/other untrusted provenance). Default TRUSTED.
#[no_mangle]
pub extern "C" fn jade_request_set_tainted(req: *mut JadeRequest, tainted: bool) -> c_int {
    match unsafe { req.as_mut() } {
        Some(r) => {
            r.inner.trust =
                if tainted { protocol::request::trust::TAINTED } else { protocol::request::trust::TRUSTED };
            JADE_OK
        }
        None => JADE_ERR_ARGS,
    }
}

/// Enable tool calls: constrain the tool body with `gbnf`, set the `<tool>`/`</tool>`
/// anchors, and request the observable in-band boundary (`keep_anchors`) so
/// [`jade_infer`] can surface complete tool calls via `on_tool_call`.
#[no_mangle]
pub extern "C" fn jade_request_enable_tools(req: *mut JadeRequest, gbnf: *const c_char) -> c_int {
    let (Some(r), Some(g)) = (unsafe { req.as_mut() }, unsafe { cstr(gbnf) }) else {
        return JADE_ERR_ARGS;
    };
    r.inner.grammar = Some(g.to_owned());
    r.inner.anchor = Some(anchors::TOOL_OPEN.to_owned());
    r.inner.stop_anchor = Some(anchors::TOOL_CLOSE.to_owned());
    r.inner.keep_anchors = true;
    JADE_OK
}

// ── Inference ───────────────────────────────────────────────────────────────

/// Send `req` over `conn` and stream the response through `callbacks`. Blocks until
/// the daemon sends Done or Error. Returns a `JADE_*` status code.
#[no_mangle]
pub extern "C" fn jade_infer(
    conn: *mut JadeConn,
    req: *const JadeRequest,
    callbacks: *const JadeCallbacks,
    userdata: *mut c_void,
) -> c_int {
    let (Some(conn), Some(req), Some(cb)) =
        (unsafe { conn.as_mut() }, unsafe { req.as_ref() }, unsafe { callbacks.as_ref() })
    else {
        return JADE_ERR_ARGS;
    };

    // Materialize the wire request (assembling the messages envelope if needed).
    let mut wire = req.inner.clone();
    if req.is_messages {
        wire.prompt = build_messages_prompt(&req.system, &req.messages);
    }

    run_infer(&mut conn.stream, &wire, cb, userdata)
}

fn run_infer(
    stream: &mut UnixStream,
    req: &InferenceRequest,
    cb: &JadeCallbacks,
    userdata: *mut c_void,
) -> c_int {
    let payload = match req.encode() {
        Ok(p) => p,
        Err(_) => return JADE_ERR_IO,
    };
    if stream.write_all(&payload).is_err() {
        return JADE_ERR_IO;
    }

    // Tool-span detection is on only when the request opted into the observable
    // boundary AND defined both anchors.
    let mut parser = match (req.keep_anchors, req.anchor.clone(), req.stop_anchor.clone()) {
        (true, Some(open), Some(close)) => ToolStreamParser::new(open, close),
        _ => ToolStreamParser::disabled(),
    };

    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        loop {
            match Frame::decode(&buf) {
                Ok((frame, consumed)) => {
                    buf.drain(..consumed);
                    match frame {
                        Frame::Meta { model } => emit_meta(cb, &model, userdata),
                        Frame::Token(text) => {
                            for ev in parser.push_token(&text) {
                                emit_event(cb, ev, userdata);
                            }
                        }
                        Frame::Done { tokens_used } => {
                            for ev in parser.finish() {
                                emit_event(cb, ev, userdata);
                            }
                            if let Some(f) = cb.on_done {
                                f(tokens_used, userdata);
                            }
                            return JADE_OK;
                        }
                        Frame::Error(msg) => {
                            emit_error(cb, &msg, userdata);
                            return JADE_ERR_DAEMON;
                        }
                    }
                }
                Err(FrameError::Incomplete) => break,
                Err(_) => return JADE_ERR_IO,
            }
        }
        match stream.read(&mut tmp) {
            Ok(0) => return JADE_ERR_IO, // EOF before Done
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(_) => return JADE_ERR_IO,
        }
    }
}

fn emit_event(cb: &JadeCallbacks, ev: Event, userdata: *mut c_void) {
    match ev {
        Event::Content(text) => {
            if let Some(f) = cb.on_token {
                f(text.as_ptr() as *const c_char, text.len(), userdata);
            }
        }
        Event::ToolCall { name, args } => {
            if let Some(f) = cb.on_tool_call {
                let name = CString::new(name).unwrap_or_default();
                let args = CString::new(args).unwrap_or_default();
                f(name.as_ptr(), args.as_ptr(), userdata);
            }
        }
    }
}

fn emit_meta(cb: &JadeCallbacks, model: &str, userdata: *mut c_void) {
    if let Some(f) = cb.on_meta {
        let m = CString::new(model).unwrap_or_default();
        f(m.as_ptr(), userdata);
    }
}

fn emit_error(cb: &JadeCallbacks, msg: &str, userdata: *mut c_void) {
    if let Some(f) = cb.on_error {
        let m = CString::new(msg).unwrap_or_default();
        f(m.as_ptr(), userdata);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_prompt_is_valid_jade_protocol() {
        let s = build_messages_prompt(
            "be brief",
            &[("user".into(), "hi".into()), ("assistant".into(), "yo".into())],
        );
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["_jade_protocol"], "messages");
        assert_eq!(v["system"], "be brief");
        assert_eq!(v["messages"][0]["role"], "user");
        assert_eq!(v["messages"][1]["content"], "yo");
    }

    #[test]
    fn enable_tools_sets_anchors_and_flag() {
        let mut r = JadeRequest {
            inner: fresh_request(),
            is_messages: false,
            system: String::new(),
            messages: Vec::new(),
        };
        let gbnf = CString::new(r#"root ::= "{" [^}]* "}""#).unwrap();
        let rc = jade_request_enable_tools(&mut r as *mut _, gbnf.as_ptr());
        assert_eq!(rc, JADE_OK);
        assert_eq!(r.inner.anchor.as_deref(), Some("<tool>"));
        assert_eq!(r.inner.stop_anchor.as_deref(), Some("</tool>"));
        assert!(r.inner.keep_anchors);
        assert!(r.inner.grammar.is_some());
    }
}
