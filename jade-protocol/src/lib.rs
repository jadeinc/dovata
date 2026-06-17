//! jade-protocol — wire protocol for jade ↔ jade-tree LLM inference
//!
//! ## Transport
//!
//! Communication flows through a Unix domain socket at `/run/jade/llm.sock`.
//! The client (typically the AOT-compiled program's C runtime) opens one
//! connection at first use and reuses it for the program's lifetime, sending
//! multiple requests sequentially. jade-tree loops on each connection,
//! processing one request at a time, until the client closes (EOF).
//!
//! ```text
//!   jade (compiler/tool)
//!     connect → /run/jade/llm.sock     (once per program)
//!     for each ?p in the program:
//!       write() → InferenceRequest    (length-prefixed JSON)
//!       read()  ← Frame stream         (binary frames until DONE or ERROR)
//!     close
//!   jade-tree (inference daemon)
//!     accept() ← connection
//!     loop:
//!       read()  ← InferenceRequest    (EOF → close connection)
//!       write() → Frame stream
//! ```
//!
//! ## Request encoding
//!
//! ```text
//! [4 bytes LE: json_len][json_len bytes: JSON InferenceRequest]
//! ```
//!
//! ## Response frame encoding
//!
//! ```text
//! [1 byte: type][2 bytes LE: payload_len][payload_len bytes: payload]
//!
//! 0x01  TOKEN   UTF-8 token text
//! 0x02  DONE    8-byte LE u64 total_tokens_used
//! 0x03  ERROR   UTF-8 error message
//! 0x04  META    UTF-8 model name (sent first)
//! ```
//!
//! ## Request prompt format — the two-layer message contract
//!
//! `InferenceRequest.prompt` is interpreted by the daemon
//! (`jaded::prompt::extract_jade_protocol`, the authoritative parser) one of two ways:
//!
//! * **Plain string** — wrapped as a single `user` message.
//! * **A "messages" envelope** — a JSON object with `"_jade_protocol": "messages"`,
//!   rendered through the model's own GGUF chat template:
//!
//!   ```json
//!   {
//!     "_jade_protocol": "messages",
//!     "system": "optional system prompt",
//!     "messages": [
//!       {"role": "user",      "content": "What's the weather in SF?"},
//!       {"role": "assistant", "content": "<tool>{\"tool_name\":\"get_weather\",\"city\":\"SF\"}</tool>"},
//!       {"role": "tool",      "content": "26°C and sunny"}
//!     ]
//!   }
//!   ```
//!
//!   A `tool`-role message is **load-bearing**: its content is wrapped in
//!   `<tool_response>…</tool_response>` and the role is kept as `tool` (NOT remapped to
//!   `user`) — Qwen3's embedded template only associates a tool result with the preceding
//!   tool call when it sees `<|im_start|>tool`.
//!
//! ## Anchored tool-call spans
//!
//! A tool call is requested by setting `anchor` = [`anchors::TOOL_OPEN`], `stop_anchor` =
//! [`anchors::TOOL_CLOSE`], and a `grammar` constraining the JSON body. With
//! `keep_anchors = true`, the daemon makes the closing boundary observable in-band so a
//! client can delimit the span by pure string parsing (see [`InferenceRequest::keep_anchors`]).

pub mod request;
pub mod response;

pub use request::{InferenceRequest, RequestDecodeError};
pub use response::{Frame, FrameError};

/// Canonical anchor strings for tool-call spans — the single source of truth shared by
/// the daemon, the client library, and (eventually) the language compiler. A tool call is
/// the text span `<tool>{…json…}</tool>`.
pub mod anchors {
    /// Opening delimiter of a tool-call span (the request `anchor`).
    pub const TOOL_OPEN: &str = "<tool>";
    /// Closing delimiter of a tool-call span (the request `stop_anchor`).
    pub const TOOL_CLOSE: &str = "</tool>";
}
