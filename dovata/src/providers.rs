//! Host-side provider loader — `dlopen`s a provider `.so`/`.dylib` and drives its
//! C-ABI (the daemon half of the plugin boundary; the provider half is
//! `ovata_infer_protocol::export_provider!`). Pure and cross-platform (no
//! llama.cpp), so it builds and unit-tests on macOS.

use std::ffi::c_void;
use std::path::Path;

use libloading::{Library, Symbol};
use ovata_infer_protocol::{Frame, FrameSink, InferenceRequest, PROVIDER_ABI_VERSION};

// The exported C-ABI signatures — must match what `export_provider!` emits.
type AbiVersionFn = unsafe extern "C" fn() -> u32;
type NamePtrFn = unsafe extern "C" fn() -> *const u8;
type NameLenFn = unsafe extern "C" fn() -> usize;
type NewFn = unsafe extern "C" fn(*const u8, usize) -> *mut c_void;
type FrameCb = extern "C" fn(*mut c_void, *const u8, usize);
type InferFn = unsafe extern "C" fn(*mut c_void, *const u8, usize, FrameCb, *mut c_void) -> i32;
type FreeFn = unsafe extern "C" fn(*mut c_void);

/// Why loading a provider `.so` failed.
#[derive(Debug)]
pub enum LoadError {
    /// The library could not be opened (missing file, bad format, unresolved deps).
    Open(String),
    /// A required exported symbol was absent.
    MissingSymbol(String),
    /// The provider was built against an incompatible provider ABI.
    AbiMismatch { provider: u32, host: u32 },
    /// `ovata_provider_new` returned null (the provider rejected its config).
    Init,
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open(e) => write!(f, "could not open provider library: {e}"),
            Self::MissingSymbol(s) => write!(f, "provider missing symbol: {s}"),
            Self::AbiMismatch { provider, host } => {
                write!(f, "provider ABI {provider} != host ABI {host}")
            }
            Self::Init => write!(f, "provider failed to initialize (bad config?)"),
        }
    }
}

impl std::error::Error for LoadError {}

/// A loaded, initialized provider plugin. Owns the dynamic library and the
/// provider handle; frees the handle and unloads the library on drop.
pub struct LoadedPlugin {
    handle: *mut c_void,
    name: String,
    infer: InferFn,
    free: FreeFn,
    // Declared last so it drops last: our Drop frees the handle first (while the
    // library is still mapped), then this field unloads the library.
    _lib: Library,
}

impl LoadedPlugin {
    /// Load and initialize the provider at `path` with opaque `config` bytes
    /// (typically JSON from the daemon config).
    ///
    /// # Safety
    /// `path` must be a provider library built with `export_provider!` against a
    /// compatible protocol crate. The ABI-version check rejects the common
    /// mismatch, but a library exporting the right symbol *names* with wrong
    /// signatures is still undefined behavior.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self, LoadError> {
        let lib = Library::new(path).map_err(|e| LoadError::Open(e.to_string()))?;

        let sym = |n: &[u8]| -> Result<*const (), LoadError> {
            // Helper for the presence check only; typed gets are below.
            unsafe { lib.get::<*const ()>(n) }
                .map(|s| *s)
                .map_err(|e| LoadError::MissingSymbol(e.to_string()))
        };
        // Fail fast with a clear error if any symbol is absent.
        for n in [
            b"ovata_provider_abi_version\0".as_slice(),
            b"ovata_provider_name_ptr\0",
            b"ovata_provider_name_len\0",
            b"ovata_provider_new\0",
            b"ovata_provider_infer\0",
            b"ovata_provider_free\0",
        ] {
            sym(n)?;
        }

        let abi: Symbol<AbiVersionFn> = lib
            .get(b"ovata_provider_abi_version\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
        let provider_abi = abi();
        if provider_abi != PROVIDER_ABI_VERSION {
            return Err(LoadError::AbiMismatch { provider: provider_abi, host: PROVIDER_ABI_VERSION });
        }

        let name_ptr: Symbol<NamePtrFn> = lib
            .get(b"ovata_provider_name_ptr\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
        let name_len: Symbol<NameLenFn> = lib
            .get(b"ovata_provider_name_len\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
        let name = {
            let p = name_ptr();
            let l = name_len();
            // SAFETY: the name symbols return a static &str's ptr/len.
            String::from_utf8_lossy(std::slice::from_raw_parts(p, l)).into_owned()
        };

        let new_fn: Symbol<NewFn> = lib
            .get(b"ovata_provider_new\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
        let handle = new_fn(config.as_ptr(), config.len());
        if handle.is_null() {
            return Err(LoadError::Init);
        }

        // Detach the two hot-path fn pointers from their Symbol borrow of `lib`.
        // Sound because `lib` is moved into the struct and outlives every call.
        let infer: InferFn = *lib
            .get(b"ovata_provider_infer\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;
        let free: FreeFn = *lib
            .get(b"ovata_provider_free\0")
            .map_err(|e| LoadError::MissingSymbol(e.to_string()))?;

        Ok(LoadedPlugin { handle, name, infer, free, _lib: lib })
    }

    /// The provider's name — what the daemon puts in the `Meta` frame.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Serve one request, forwarding every frame the provider emits to `sink`.
    pub fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) -> Result<(), String> {
        let body = req.encode_body().map_err(|e| e.to_string())?;

        // Trampoline: decode the provider's frame bytes and push to the Rust sink.
        // `ctx` points at the `&mut dyn FrameSink` (a fat pointer) on our stack.
        extern "C" fn trampoline(ctx: *mut c_void, ptr: *const u8, len: usize) {
            // SAFETY: ctx is the `&mut &mut dyn FrameSink` set up below; ptr/len
            // are valid for this call only (the provider owns the buffer).
            let sink = unsafe { &mut *(ctx as *mut &mut dyn FrameSink) };
            let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
            if let Ok((frame, _)) = Frame::decode(bytes) {
                sink.emit(frame);
            }
        }

        let mut sink_ref: &mut dyn FrameSink = sink;
        let ctx = &mut sink_ref as *mut &mut dyn FrameSink as *mut c_void;
        // SAFETY: live handle; body valid for its len; trampoline matches FrameCb.
        let rc = unsafe { (self.infer)(self.handle, body.as_ptr(), body.len(), trampoline, ctx) };
        if rc == 0 {
            Ok(())
        } else {
            Err(format!("provider '{}' infer failed (rc={rc})", self.name))
        }
    }
}

impl Drop for LoadedPlugin {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle came from ovata_provider_new and is freed exactly once.
            unsafe { (self.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
        // `_lib` unloads after this returns.
    }
}

/// A routable provider the host can dispatch a request to: either the **local**
/// llama.cpp engine served in-process (compiled in on Linux, no FFI hop) or a
/// **remote** provider `.so` loaded via [`LoadedPlugin`]. The [`Router`](crate::host)
/// holds a `Vec<ProviderHandle>` and picks one per request.
pub enum ProviderHandle {
    /// The in-process local engine. Linux-only (it links the vendored infer-engine).
    #[cfg(target_os = "linux")]
    Local(crate::engine::Dovata),
    /// A dlopened remote-API provider (anthropic/openai).
    Plugin(LoadedPlugin),
}

impl ProviderHandle {
    /// The provider's name — what the daemon puts in the `Meta` frame.
    pub fn name(&self) -> &str {
        match self {
            #[cfg(target_os = "linux")]
            Self::Local(_) => <crate::engine::Dovata as ovata_infer_protocol::Provider>::NAME,
            Self::Plugin(p) => p.name(),
        }
    }

    /// Serve one request, forwarding every frame the provider emits to `sink`. The
    /// in-process engine surfaces its own errors as `Error` frames (so it always
    /// returns `Ok`); a plugin can also fail at the C-ABI boundary.
    pub fn infer(&self, req: &InferenceRequest, sink: &mut dyn FrameSink) -> Result<(), String> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Local(engine) => {
                use ovata_infer_protocol::Provider;
                engine.infer(req, sink);
                Ok(())
            }
            Self::Plugin(p) => p.infer(req, sink),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct VecSink(Vec<Frame>);
    impl FrameSink for VecSink {
        fn emit(&mut self, f: Frame) {
            self.0.push(f);
        }
    }

    // The echo fixture built by `cargo build -p ovata-provider-echo`.
    fn echo_fixture_path() -> std::path::PathBuf {
        let filename = libloading::library_filename("ovata_provider_echo");
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop(); // repo root (manifest dir is <repo>/dovata)
        p.push("target");
        p.push("debug");
        p.push(filename);
        p
    }

    #[test]
    fn loads_echo_fixture_and_round_trips_frames() {
        let path = echo_fixture_path();
        assert!(
            path.exists(),
            "echo fixture not built at {path:?} — run `cargo build -p ovata-provider-echo` first"
        );

        // SAFETY: the echo fixture is built from export_provider! against this
        // same protocol crate, so its ABI matches.
        let provider = unsafe { LoadedPlugin::load(&path, b"{}") }.expect("load echo provider");
        assert_eq!(provider.name(), "echo");

        let req = InferenceRequest { prompt: "ping".into(), ..Default::default() };
        let mut sink = VecSink(Vec::new());
        provider.infer(&req, &mut sink).expect("infer succeeds");

        assert!(matches!(sink.0.first(), Some(Frame::Token(t)) if t == "ping"));
        assert!(matches!(sink.0.last(), Some(Frame::Done { .. })));
    }

    #[test]
    fn loading_a_nonexistent_library_errors() {
        let path = Path::new("/nonexistent/libnope.so");
        let err = unsafe { LoadedPlugin::load(path, b"{}") };
        assert!(matches!(err, Err(LoadError::Open(_))));
    }
}
