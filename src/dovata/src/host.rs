//! The daemon host: owns the Unix socket, frames requests off the wire, answers
//! daemon-wide `health`/`stats`, and routes each request to a provider — emitting the
//! provider's name in the `Meta` frame. The **local** llama.cpp engine
//! ([`dovata::engine`]) is compiled in and served in-process; **remote** providers are
//! `.so` packages loaded via [`dovata::providers`](crate::providers). The local engine
//! is `#[cfg(target_os = "linux")]`, so on macOS the host builds without it (remote
//! providers still load).
//!
//! ## Configuration (environment)
//!
//! * `JADE_LOCAL_ENGINE` — set to `0`/`off`/`false` to disable the in-process local
//!   engine (Linux only; on by default). Its model/config come from the `JADE_*` env
//!   read by `Config::from_env` (e.g. `JADE_MODEL_PATH`).
//! * `JADE_PROVIDERS` — optional comma-separated remote provider libraries to load,
//!   each `path/to/lib.so` or `path/to/lib.so@config.json` (the JSON file is passed to
//!   the provider's `from_config`; absent → `{}`).
//! * `JADE_DEFAULT_PROVIDER` — name of the provider that serves unrouted requests
//!   (defaults to `dovata` if loaded, else the first).
//! * `JADE_MODEL_ROUTES` — optional `prefix=provider,...` rules; a request whose
//!   `model` starts with `prefix` goes to that provider (first match wins).

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ovata_infer_protocol::{
    Frame, FrameSink, Health, InferenceRequest, RequestDecodeError, PROTOCOL_VERSION,
};

use dovata::providers::{LoadedPlugin, ProviderHandle};

/// Daemon-wide cumulative generated-token counter (served by `stats_only`).
static GLOBAL_TOKENS: AtomicU64 = AtomicU64::new(0);

/// A [`FrameSink`] that writes each frame to the client socket and records the last
/// `Done`'s token count. A write error is latched and surfaced after `infer` returns
/// (the trait method can't return one).
struct SocketSink<'a> {
    stream: &'a mut UnixStream,
    err: Option<std::io::Error>,
    tokens: u64,
}

impl FrameSink for SocketSink<'_> {
    fn emit(&mut self, frame: Frame) {
        if self.err.is_some() {
            return; // a prior write already failed; stop touching the socket
        }
        if let Frame::Done { tokens_used } = &frame {
            self.tokens = *tokens_used;
        }
        if let Err(e) = self.stream.write_all(&frame.encode()) {
            self.err = Some(e);
        }
    }
}

/// Pure route selection: the first prefix rule whose prefix is a prefix of `model`,
/// else `default_idx`. Split out so it can be unit-tested without loading providers.
fn select_provider_index(model: &str, routes: &[(String, usize)], default_idx: usize) -> usize {
    for (prefix, idx) in routes {
        if !prefix.is_empty() && model.starts_with(prefix.as_str()) {
            return *idx;
        }
    }
    default_idx
}

/// Read the local model's GGUF header, pick its profile, and publish the choice for the
/// engine — or refuse the boot with a message that names the fix.
///
/// Only the engine's own `Config` knows where the model is, so this reads it from the
/// same place. `JADE_ALLOW_UNPROFILED=1` turns the no-profile refusal into a warning;
/// everything else here is a hard failure, because every one of them is a failure
/// llama.cpp would hit anyway, later and with a worse message.
///
/// Not gated to Linux even though its only caller is: nothing in here touches the
/// engine, so leaving it compiled everywhere means the macOS build type-checks it
/// instead of letting it rot behind a `cfg` until a GPU box finds the mistake.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn preflight_local_model() -> Result<(), String> {
    let cfg = dovata::config::Config::from_env();
    let allow_unprofiled = std::env::var("JADE_ALLOW_UNPROFILED")
        .map(|v| matches!(v.trim(), "1" | "on" | "true"))
        .unwrap_or(false);

    let pf = dovata::preflight::Preflight::read(
        &cfg.model_path,
        &dovata::profile::Registry::builtin(),
        allow_unprofiled,
    )
    .map_err(|e| format!("model preflight failed\n  {e}"))?;

    for line in pf.summary() {
        eprintln!("{line}");
    }
    if pf.ctx_exceeds_native(cfg.ctx_size) {
        eprintln!(
            "dovata: WARNING JADE_CTX_SIZE={} exceeds the model's native window of {}",
            cfg.ctx_size,
            pf.context_length.unwrap_or(0)
        );
    }

    // Hand the decision to the engine so it serves the same profile the boot approved.
    dovata::preflight::publish(&pf);
    Ok(())
}

/// The loaded providers plus routing rules.
pub struct Router {
    providers: Vec<ProviderHandle>,
    routes: Vec<(String, usize)>,
    default_idx: usize,
}

impl Router {
    /// Build from the environment (see the module docs).
    pub fn from_env() -> Result<Self, String> {
        let mut providers = Vec::new();
        let mut names = Vec::new();

        // The local llama.cpp engine is compiled in and served in-process. On Linux it
        // loads by default (dovata is a local inference daemon); set JADE_LOCAL_ENGINE=0
        // for a remote-only daemon. Compiled out entirely on non-Linux targets.
        #[cfg(target_os = "linux")]
        {
            let local_on = std::env::var("JADE_LOCAL_ENGINE")
                .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false"))
                .unwrap_or(true);
            if local_on {
                use ovata_infer_protocol::Provider;

                // Vet the model file before llama.cpp spends minutes loading it. This
                // reads the GGUF header only, so a bad path, an unusable file, or a
                // model with no profile costs a second instead of a full load. See
                // dovata::preflight for the policy.
                preflight_local_model()?;

                let engine = dovata::engine::Dovata::from_config(b"{}")
                    .map_err(|e| format!("loading local engine: {e}"))?;
                let name = <dovata::engine::Dovata as Provider>::NAME;
                eprintln!("dovata: local engine ready (provider '{name}')");
                names.push(name.to_string());
                providers.push(ProviderHandle::Local(engine));
            }
        }

        // Remote provider plugins (.so), optional: comma-separated `path.so` entries,
        // each optionally `path.so@config.json` (the JSON is passed to from_config).
        if let Ok(spec) = std::env::var("JADE_PROVIDERS") {
            for entry in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                let (so_path, cfg_path) = match entry.split_once('@') {
                    Some((p, c)) => (p.trim(), Some(c.trim())),
                    None => (entry, None),
                };
                let config: Vec<u8> = match cfg_path {
                    Some(c) => std::fs::read(c).map_err(|e| format!("reading provider config {c}: {e}"))?,
                    None => b"{}".to_vec(),
                };
                // SAFETY: operator-provided provider library built with export_provider!.
                let p = unsafe { LoadedPlugin::load(Path::new(so_path), &config) }
                    .map_err(|e| format!("loading provider {so_path}: {e}"))?;
                eprintln!("dovata: loaded provider '{}' from {so_path}", p.name());
                names.push(p.name().to_string());
                providers.push(ProviderHandle::Plugin(p));
            }
        }
        if providers.is_empty() {
            return Err("no providers loaded: enable the local engine (Linux, default on) \
                        and/or set JADE_PROVIDERS to remote provider .so paths"
                .into());
        }

        let mut routes = Vec::new();
        if let Ok(r) = std::env::var("JADE_MODEL_ROUTES") {
            for rule in r.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                if let Some((prefix, name)) = rule.split_once('=') {
                    match names.iter().position(|n| n == name.trim()) {
                        Some(idx) => routes.push((prefix.trim().to_string(), idx)),
                        None => eprintln!(
                            "dovata: route rule '{rule}' names unknown provider '{}' — ignored",
                            name.trim()
                        ),
                    }
                }
            }
        }

        let default_idx = match std::env::var("JADE_DEFAULT_PROVIDER") {
            Ok(name) => names
                .iter()
                .position(|n| n == name.trim())
                .ok_or_else(|| format!("JADE_DEFAULT_PROVIDER '{name}' is not a loaded provider"))?,
            Err(_) => names.iter().position(|n| n == "dovata").unwrap_or(0),
        };
        eprintln!("dovata: default provider = '{}'", names[default_idx]);

        Ok(Router { providers, routes, default_idx })
    }

    /// The provider that serves `model`.
    fn route(&self, model: &str) -> &ProviderHandle {
        &self.providers[select_provider_index(model, &self.routes, self.default_idx)]
    }

    /// The default provider (for daemon-wide reporting).
    fn default_provider(&self) -> &ProviderHandle {
        &self.providers[self.default_idx]
    }
}

/// Bind the socket and serve forever.
pub fn run(sock_path: String) {
    let start = Instant::now();
    eprintln!("JADE OS — dovata v{} (host)", env!("CARGO_PKG_VERSION"));

    let router = Router::from_env().unwrap_or_else(|e| {
        eprintln!("dovata: {e}");
        std::process::exit(1);
    });

    if let Some(parent) = Path::new(&sock_path).parent() {
        std::fs::create_dir_all(parent).expect("create socket dir");
    }
    let _ = std::fs::remove_file(&sock_path);
    let listener = UnixListener::bind(&sock_path).unwrap_or_else(|e| {
        eprintln!("dovata: could not bind {sock_path}: {e}");
        std::process::exit(1);
    });
    eprintln!("dovata: listening on {sock_path}");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(e) = handle_connection(&mut stream, &router, &start) {
                    eprintln!("dovata: connection error: {e}");
                }
            }
            Err(e) => eprintln!("dovata: accept error: {e}"),
        }
    }
}

fn handle_connection(
    stream: &mut UnixStream,
    router: &Router,
    start: &Instant,
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::new();
    let mut tmp = [0u8; 4096];
    let mut request_count: u64 = 0;

    loop {
        // Read until a full [4-byte len][JSON] request frame is buffered.
        let req = loop {
            match InferenceRequest::decode(&buf) {
                Ok(req) => {
                    let payload_len = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                    buf.drain(..4 + payload_len);
                    break req;
                }
                Err(RequestDecodeError::Incomplete) => {
                    let n = stream.read(&mut tmp)?;
                    if n == 0 {
                        eprintln!("dovata: connection closed after {request_count} requests");
                        return Ok(());
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                Err(e) => {
                    eprintln!("dovata: malformed request: {e}");
                    return Ok(());
                }
            }
        };
        request_count += 1;

        // Daemon-wide probes are answered by the host, not a provider.
        if req.health_only {
            let health = Health {
                status: "ok".to_owned(),
                model: router.default_provider().name().to_owned(),
                model_loaded: true,
                uptime_secs: start.elapsed().as_secs(),
                protocol_version: PROTOCOL_VERSION,
            };
            let json = serde_json::to_string(&health).unwrap_or_else(|_| "{}".to_owned());
            stream.write_all(&Frame::Json(json).encode())?;
            stream.write_all(&Frame::Done { tokens_used: 0 }.encode())?;
            continue;
        }
        if req.stats_only {
            let total = GLOBAL_TOKENS.load(Ordering::Relaxed);
            stream.write_all(&Frame::Done { tokens_used: total }.encode())?;
            continue;
        }

        // Route, then send Meta (provider name) before the provider streams its frames.
        let provider = router.route(&req.model);
        stream.write_all(&Frame::Meta { provider: provider.name().to_owned() }.encode())?;

        // Run the provider, forwarding its frames to the socket via the sink. Scope
        // the sink so its &mut borrow of `stream` ends before we touch `stream` again.
        let (infer_result, sink_err, tokens) = {
            let mut sink = SocketSink { stream: &mut *stream, err: None, tokens: 0 };
            let r = provider.infer(&req, &mut sink);
            (r, sink.err.take(), sink.tokens)
        };
        if let Some(e) = sink_err {
            return Err(e); // the socket write failed — drop the connection
        }
        match infer_result {
            Ok(()) => {
                GLOBAL_TOKENS.fetch_add(tokens, Ordering::Relaxed);
            }
            Err(msg) => {
                // Provider-level failure with no frames emitted — surface an Error.
                stream.write_all(&Frame::Error(msg).encode())?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::select_provider_index;

    #[test]
    fn routing_prefers_first_matching_prefix_then_default() {
        let routes = vec![("claude".to_string(), 1), ("gpt".to_string(), 2)];
        assert_eq!(select_provider_index("claude-sonnet-5", &routes, 0), 1);
        assert_eq!(select_provider_index("gpt-4o", &routes, 0), 2);
        // no rule matches → default
        assert_eq!(select_provider_index("qwen3-coder", &routes, 0), 0);
        // empty model → default
        assert_eq!(select_provider_index("", &routes, 0), 0);
    }

    #[test]
    fn empty_prefix_rule_never_matches() {
        let routes = vec![("".to_string(), 1)];
        assert_eq!(select_provider_index("anything", &routes, 0), 0);
    }
}
