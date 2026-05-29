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
//! ```

pub mod request;
pub mod response;

pub use request::{InferenceRequest, RequestDecodeError};
pub use response::{Frame, FrameError};
