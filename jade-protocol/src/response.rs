const TYPE_TOKEN: u8 = 0x01;
const TYPE_DONE: u8 = 0x02;
const TYPE_ERROR: u8 = 0x03;
const TYPE_META: u8 = 0x04;
const TYPE_JSON: u8 = 0x05;

/// A single streaming response frame from jade-tree.
#[derive(Debug, Clone)]
pub enum Frame {
    /// One emitted token of text.
    Token(String),
    /// Inference complete. `tokens_used` is the total token count for the request.
    Done { tokens_used: u64 },
    /// Inference failed. The daemon will send no further frames for this request.
    Error(String),
    /// Metadata about the serving model. Sent as the first frame before any tokens.
    Meta { model: String },
    /// A structured JSON payload (UTF-8). Used for out-of-band, non-token responses
    /// such as the health report (see [`crate::Health`]). The payload is an opaque
    /// JSON string; the schema is defined by the request that elicited it.
    Json(String),
}

impl Frame {
    /// Encode to wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Frame::Token(text) => encode_frame(TYPE_TOKEN, text.as_bytes()),
            Frame::Done { tokens_used } => encode_frame(TYPE_DONE, &tokens_used.to_le_bytes()),
            Frame::Error(msg) => encode_frame(TYPE_ERROR, msg.as_bytes()),
            Frame::Meta { model } => encode_frame(TYPE_META, model.as_bytes()),
            Frame::Json(json) => encode_frame(TYPE_JSON, json.as_bytes()),
        }
    }

    /// Decode one frame from `buf`.
    ///
    /// Returns `(frame, bytes_consumed)` on success. Call repeatedly on a
    /// sliding buffer to drain a stream of frames.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize), FrameError> {
        if buf.len() < 3 {
            return Err(FrameError::Incomplete);
        }
        let frame_type = buf[0];
        let payload_len = u16::from_le_bytes([buf[1], buf[2]]) as usize;
        if buf.len() < 3 + payload_len {
            return Err(FrameError::Incomplete);
        }
        let payload = &buf[3..3 + payload_len];

        let frame = match frame_type {
            TYPE_TOKEN => Frame::Token(utf8(payload)?),
            TYPE_DONE => {
                if payload_len != 8 {
                    return Err(FrameError::Malformed("DONE payload must be 8 bytes"));
                }
                Frame::Done {
                    tokens_used: u64::from_le_bytes(payload.try_into().unwrap()),
                }
            }
            TYPE_ERROR => Frame::Error(utf8(payload)?),
            TYPE_META => Frame::Meta { model: utf8(payload)? },
            TYPE_JSON => Frame::Json(utf8(payload)?),
            other => return Err(FrameError::UnknownType(other)),
        };
        Ok((frame, 3 + payload_len))
    }
}

fn encode_frame(frame_type: u8, payload: &[u8]) -> Vec<u8> {
    let len = payload.len() as u16;
    let mut buf = Vec::with_capacity(3 + payload.len());
    buf.push(frame_type);
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

fn utf8(bytes: &[u8]) -> Result<String, FrameError> {
    std::str::from_utf8(bytes)
        .map(|s| s.to_owned())
        .map_err(|_| FrameError::InvalidUtf8)
}

#[derive(Debug)]
pub enum FrameError {
    Incomplete,
    InvalidUtf8,
    UnknownType(u8),
    Malformed(&'static str),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Incomplete => write!(f, "incomplete frame buffer"),
            Self::InvalidUtf8 => write!(f, "frame payload is not valid UTF-8"),
            Self::UnknownType(t) => write!(f, "unknown frame type: {t:#04x}"),
            Self::Malformed(msg) => write!(f, "malformed frame: {msg}"),
        }
    }
}

impl std::error::Error for FrameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_roundtrip() {
        let frame = Frame::Token("hello".into());
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(matches!(decoded, Frame::Token(ref t) if t == "hello"));
    }

    #[test]
    fn done_roundtrip() {
        let frame = Frame::Done { tokens_used: 42 };
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(matches!(decoded, Frame::Done { tokens_used: 42 }));
    }

    #[test]
    fn error_roundtrip() {
        let frame = Frame::Error("model not loaded".into());
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(matches!(decoded, Frame::Error(ref e) if e == "model not loaded"));
    }

    #[test]
    fn json_roundtrip() {
        let frame = Frame::Json(r#"{"status":"ok"}"#.into());
        let encoded = frame.encode();
        let (decoded, consumed) = Frame::decode(&encoded).unwrap();
        assert_eq!(consumed, encoded.len());
        assert!(matches!(decoded, Frame::Json(ref j) if j == r#"{"status":"ok"}"#));
    }

    #[test]
    fn incomplete_returns_error() {
        assert!(matches!(Frame::decode(&[0x01, 0x03]), Err(FrameError::Incomplete)));
    }

    #[test]
    fn multi_frame_stream() {
        let frames = vec![
            Frame::Token("foo".into()),
            Frame::Token(" bar".into()),
            Frame::Done { tokens_used: 10 },
        ];
        let mut stream: Vec<u8> = frames.iter().flat_map(|f| f.encode()).collect();

        let mut decoded = Vec::new();
        while !stream.is_empty() {
            let (frame, consumed) = Frame::decode(&stream).unwrap();
            decoded.push(frame);
            stream.drain(..consumed);
        }
        assert_eq!(decoded.len(), 3);
    }
}
