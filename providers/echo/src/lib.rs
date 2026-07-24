//! Minimal reference provider: echoes the request prompt back as one token, then
//! completes. It exists as the smallest possible `impl Provider` — useful as a
//! copy-paste starting point and as the host loader's round-trip test fixture.
//! It needs no config and no network.

use ovata_infer_protocol::{export_provider, Frame, FrameSink, InferenceRequest, Provider};

pub struct Echo;

impl Provider for Echo {
    const NAME: &'static str = "echo";

    fn from_config(_config: &[u8]) -> Result<Self, String> {
        Ok(Echo)
    }

    fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) {
        sink.emit_text(&req.prompt);
        sink.emit(Frame::Done { tokens_used: 0 });
    }
}

export_provider!(Echo);
