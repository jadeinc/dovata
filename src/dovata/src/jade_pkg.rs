//! The entry point the Jade compiler actually looks for.
//!
//! `export_provider!` gives this crate the `ovata_provider_*` C ABI, and that is the
//! right surface for any host that speaks plain C. It is not the surface the Jade
//! compiler uses. The compiler loads a provider through its own native-package
//! machinery: `dlopen`, then `jade_pkg_init` to collect name-to-function-pointer
//! bindings, then a binding called `infer`. It dropped its dependency on
//! `ovata-infer-protocol`'s Rust half in jade v1.1.30 and kept only `jade/infer.jde`,
//! which declares the *messages* and says nothing about the entry point.
//!
//! So the two sides agreed on every message and disagreed on how to start the
//! conversation, and the protocol's own docs asserted the C-ABI answer the compiler
//! had stopped honouring. Loading the engine from a Jade program failed with
//! "missing `jade_pkg_init` symbol". See ovata-infer-protocol v0.5.2, which now names
//! the compiler's loader as the authority.
//!
//! This module is the conforming half. Both entry points live in one library over one
//! `Provider` impl — nothing about the engine changes, and a non-Jade host keeps the
//! ABI it already links.
//!
//! ## The ABI, and why every byte of it is copied from the compiler
//!
//! The layouts below mirror `src/native/mod.rs` in the jade repo exactly, and its
//! sibling `src/runtime_aot/native.c`. They are duplicated rather than imported
//! because the compiler does not publish them as a crate. A drift here is not a
//! compile error on either side — it is a corrupted value at runtime, which is why
//! `jade_pkg_abi_version` exists and why this file states the version it was written
//! against.
//!
//! ## Who owns which allocation
//!
//! The rule that matters: the compiler calls `ffi_free` on the value this module
//! writes into `out`, and `ffi_free` recurses through arrays, dicts and structs
//! freeing every string inside them with libc `free`. So every byte of an returned
//! tree must come from libc `malloc`/`strdup`, never from Rust's allocator — the
//! process holds two allocators and freeing across them corrupts the heap.
//!
//! A *top-level* string return is the exception: `ffi_free` only recurses for
//! array/dict/struct tags, so a bare `JADE_TAG_STR` would leak. This module never
//! returns one; `infer` always returns an array.

// On a non-Linux build the engine is compiled out, so the exported entry points are
// too and the marshalling below is reachable only from tests. It is still compiled and
// tested there on purpose: it is the platform-independent half of this boundary, and
// the field names it pins are the part most likely to drift.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::ffi::{c_char, CStr};
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

use ovata_infer_protocol::Frame;
#[cfg(target_os = "linux")]
use ovata_infer_protocol::{FrameSink, InferenceRequest, Provider};

// ── Value tags — must match src/native/mod.rs in the jade repo ────────────────

const JADE_TAG_NIL: u8 = 0;
const JADE_TAG_INT: u8 = 1;
const JADE_TAG_BOOL: u8 = 3;
const JADE_TAG_STR: u8 = 4;
const JADE_TAG_ERROR: u8 = 5;
const JADE_TAG_ARRAY: u8 = 6;
const JADE_TAG_STRUCT: u8 = 8;

/// The compiler's value ABI this module is written against. It went 1 -> 2 in jade
/// v1.1.31, when structs began crossing the boundary — which this module depends on,
/// since both the request and every frame cross as a struct. The compiler refuses a
/// package whose version it cannot talk to, so this number is load-bearing.
const JADE_VALUE_ABI: u32 = 2;

#[repr(C)]
union JadeValData {
    as_int: i64,
    as_float: f64,
    as_bool: u8,
    as_str: *const u8,
    as_nil: u64,
    as_arr: *mut JadeArr,
    as_dict: *mut JadeMap,
    as_struct: *mut JadeStruct,
}

#[repr(C)]
struct JadeVal {
    tag: u8,
    _pad: [u8; 7],
    data: JadeValData,
}

#[repr(C)]
struct JadeArr {
    items: *mut JadeVal,
    len: usize,
}

#[repr(C)]
struct JadeMap {
    keys: *mut *const u8,
    vals: *mut JadeVal,
    len: usize,
}

#[repr(C)]
struct JadeStruct {
    type_name: *const u8,
    keys: *mut *const u8,
    vals: *mut JadeVal,
    len: usize,
}

type JadeNativeFnPtr =
    unsafe extern "C" fn(argc: usize, argv: *const JadeVal, out: *mut JadeVal) -> i32;

#[repr(C)]
struct JadeBinding {
    name: *const c_char,
    func: JadeNativeFnPtr,
}

#[repr(C)]
struct JadeNativePkg {
    name: *const c_char,
    bindings: *const JadeBinding,
    binding_count: usize,
}

// ── libc-heap construction ───────────────────────────────────────────────────
//
// Everything below allocates with libc so the compiler's `ffi_free` can release it.

/// Copy `s` into a libc-owned, null-terminated buffer.
fn dup_str(s: &str) -> *const u8 {
    // SAFETY: `len + 1` bytes are allocated and every one is written before use.
    unsafe {
        let len = s.len();
        let p = libc::malloc(len + 1).cast::<u8>();
        if p.is_null() {
            return std::ptr::null();
        }
        std::ptr::copy_nonoverlapping(s.as_ptr(), p, len);
        *p.add(len) = 0;
        p
    }
}

/// Copy a slice into a libc-owned array, returning its base pointer.
fn dup_slice<T>(items: Vec<T>) -> *mut T {
    unsafe {
        let n = items.len();
        if n == 0 {
            return std::ptr::null_mut();
        }
        let p = libc::malloc(std::mem::size_of::<T>() * n).cast::<T>();
        if p.is_null() {
            return std::ptr::null_mut();
        }
        for (i, item) in items.into_iter().enumerate() {
            std::ptr::write(p.add(i), item);
        }
        p
    }
}

fn val_nil() -> JadeVal {
    JadeVal { tag: JADE_TAG_NIL, _pad: [0; 7], data: JadeValData { as_nil: 0 } }
}

fn val_str(s: &str) -> JadeVal {
    JadeVal { tag: JADE_TAG_STR, _pad: [0; 7], data: JadeValData { as_str: dup_str(s) } }
}

fn val_int(n: i64) -> JadeVal {
    JadeVal { tag: JADE_TAG_INT, _pad: [0; 7], data: JadeValData { as_int: n } }
}

/// A struct value carrying its type name — the tag the compiler reads to tell one
/// frame from another. A dict with the wrong keys would read as a bag of nils and
/// fail silently; a named struct the receiver can refuse.
fn val_struct(type_name: &str, fields: Vec<(&str, JadeVal)>) -> JadeVal {
    let n = fields.len();
    let mut keys = Vec::with_capacity(n);
    let mut vals = Vec::with_capacity(n);
    for (k, v) in fields {
        keys.push(dup_str(k));
        vals.push(v);
    }
    let s = JadeStruct {
        type_name: dup_str(type_name),
        keys: dup_slice(keys),
        vals: dup_slice(vals),
        len: n,
    };
    let boxed = unsafe {
        let p = libc::malloc(std::mem::size_of::<JadeStruct>()).cast::<JadeStruct>();
        std::ptr::write(p, s);
        p
    };
    JadeVal { tag: JADE_TAG_STRUCT, _pad: [0; 7], data: JadeValData { as_struct: boxed } }
}

fn val_array(items: Vec<JadeVal>) -> JadeVal {
    let n = items.len();
    let arr = JadeArr { items: dup_slice(items), len: n };
    let boxed = unsafe {
        let p = libc::malloc(std::mem::size_of::<JadeArr>()).cast::<JadeArr>();
        std::ptr::write(p, arr);
        p
    };
    JadeVal { tag: JADE_TAG_ARRAY, _pad: [0; 7], data: JadeValData { as_arr: boxed } }
}

// ── Reading the request the compiler hands us ────────────────────────────────

/// Read a null-terminated pointer as a `String`, or `None` if null/not UTF-8.
unsafe fn read_cstr(p: *const u8) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p.cast::<c_char>()).to_str().ok().map(str::to_owned) }
}

/// One field of the incoming `InferRequest`, as a `String`. Nil fields (the common
/// case for `grammar`/`anchor`/`stop_anchor`) come back as `None`.
unsafe fn field_str(v: &JadeVal, name: &str) -> Option<String> {
    if v.tag != JADE_TAG_STRUCT {
        return None;
    }
    let s = unsafe { v.data.as_struct };
    if s.is_null() {
        return None;
    }
    let (keys, vals, len) = unsafe { ((*s).keys, (*s).vals, (*s).len) };
    for i in 0..len {
        let key = unsafe { read_cstr(*keys.add(i)) };
        if key.as_deref() == Some(name) {
            let val = unsafe { &*vals.add(i) };
            return match val.tag {
                JADE_TAG_STR | JADE_TAG_ERROR => unsafe { read_cstr(val.data.as_str) },
                _ => None,
            };
        }
    }
    None
}

// ── The engine, loaded once ──────────────────────────────────────────────────

/// The model is loaded on the first `infer` and kept for the life of the process.
/// `ovata_provider_new` does the same job for the C ABI; a Jade program never calls
/// it, so the package path needs its own handle. Loading is the expensive step —
/// tens of seconds and most of a GPU's memory — so it must not happen per request.
#[cfg(target_os = "linux")]
static ENGINE: OnceLock<Result<crate::engine::Dovata, String>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn engine() -> Result<&'static crate::engine::Dovata, String> {
    match ENGINE.get_or_init(|| crate::engine::Dovata::from_config(b"")) {
        Ok(e) => Ok(e),
        Err(e) => Err(e.clone()),
    }
}

/// One protocol frame as the struct the compiler reads.
///
/// Pure: no engine, no llama.cpp, so it is unit-tested on any machine. The field
/// names come from the protocol's own `jade/infer.jde`, which is the half the
/// compiler kept — get one wrong and the frame decodes as a bag of nils.
fn frame_to_val(frame: Frame) -> JadeVal {
    match frame {
        Frame::Token(text) => val_struct("Token", vec![("text", val_str(&text))]),
        Frame::Done { tokens_used } => val_struct(
            "Done",
            // Saturating rather than wrapping: a token count past i64::MAX is not
            // reachable, and a silently negative one would be worse than a clamp.
            vec![("tokens_used", val_int(i64::try_from(tokens_used).unwrap_or(i64::MAX)))],
        ),
        Frame::Error(message) => val_struct("Error", vec![("message", val_str(&message))]),
        Frame::Meta { provider } => val_struct("Meta", vec![("provider", val_str(&provider))]),
        Frame::Json(json) => val_struct("Json", vec![("json", val_str(&json))]),
    }
}

/// Collects frames into the array the compiler expects, rather than streaming them.
///
/// The C ABI streams frames through a callback as they are produced. This boundary
/// has no callback — the compiler's ABI carries no function values, deliberately —
/// so `infer` returns the whole reply at once. The engine still generates
/// incrementally; only the delivery is batched.
#[cfg(target_os = "linux")]
struct VecSink {
    frames: Vec<JadeVal>,
}

#[cfg(target_os = "linux")]
impl FrameSink for VecSink {
    fn emit(&mut self, frame: Frame) {
        self.frames.push(frame_to_val(frame));
    }
}

// ── Bindings ─────────────────────────────────────────────────────────────────

/// `infer(request) -> [Frame]`.
///
/// A non-zero return makes the compiler raise, reading the message out of `out` when
/// it is a string. An `Error` frame inside a successful array raises too, as a
/// catchable inference error — which is the right shape for a model that failed to
/// answer, as opposed to a package that could not be called at all.
#[cfg(target_os = "linux")]
unsafe extern "C" fn jade_infer(argc: usize, argv: *const JadeVal, out: *mut JadeVal) -> i32 {
    let result = std::panic::catch_unwind(|| {
        if argc < 1 || argv.is_null() {
            return Err("infer expects one InferRequest argument".to_owned());
        }
        let arg = unsafe { &*argv };
        let input = unsafe { field_str(arg, "input") }
            .ok_or_else(|| "InferRequest.input is missing or not a string".to_owned())?;

        let engine = engine()?;
        let req = InferenceRequest {
            prompt: input,
            grammar: unsafe { field_str(arg, "grammar") },
            anchor: unsafe { field_str(arg, "anchor") },
            stop_anchor: unsafe { field_str(arg, "stop_anchor") },
            ..Default::default()
        };

        let mut sink = VecSink { frames: Vec::new() };
        engine.infer(&req, &mut sink);
        Ok(sink.frames)
    });

    match result {
        Ok(Ok(frames)) => {
            unsafe { std::ptr::write(out, val_array(frames)) };
            0
        }
        Ok(Err(message)) => {
            unsafe { std::ptr::write(out, val_str(&message)) };
            1
        }
        // A panic must never cross the C boundary; it is undefined behaviour.
        Err(_) => {
            unsafe { std::ptr::write(out, val_str("dovata: panicked during inference")) };
            1
        }
    }
}

/// `configure(opts)` — accepted and ignored.
///
/// The remote providers take an API key here. The local engine has no credential to
/// store: it reads its model path and tuning from the environment at load. The
/// binding exists because the compiler calls `configure` when a config blob is
/// present, and a package that does not export it simply skips the step — declaring
/// it makes the no-op deliberate rather than incidental.
#[cfg(target_os = "linux")]
unsafe extern "C" fn jade_configure(_argc: usize, _argv: *const JadeVal, out: *mut JadeVal) -> i32 {
    unsafe { std::ptr::write(out, val_nil()) };
    0
}

// ── Package entry point ──────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
static PKG_NAME: &[u8] = b"dovata\0";
#[cfg(target_os = "linux")]
static INFER_NAME: &[u8] = b"infer\0";
#[cfg(target_os = "linux")]
static CONFIGURE_NAME: &[u8] = b"configure\0";

/// The value ABI this package was built against. The compiler compares it with its
/// own and refuses a mismatch at load, with a message naming the fix.
#[cfg(target_os = "linux")]
#[no_mangle]
pub extern "C" fn jade_pkg_abi_version() -> u32 {
    JADE_VALUE_ABI
}

/// The symbol the compiler looks for. Writes the binding table into `out`.
///
/// The table is `static`, so it outlives the call and needs no allocation — the
/// compiler copies the name strings it keeps.
///
/// # Safety
/// `out` must point to a writable `JadeNativePkg`.
#[cfg(target_os = "linux")]
#[no_mangle]
pub unsafe extern "C" fn jade_pkg_init(out: *mut JadeNativePkg) -> i32 {
    static BINDINGS: [JadeBinding; 2] = [
        JadeBinding { name: INFER_NAME.as_ptr().cast::<c_char>(), func: jade_infer },
        JadeBinding { name: CONFIGURE_NAME.as_ptr().cast::<c_char>(), func: jade_configure },
    ];
    if out.is_null() {
        return 1;
    }
    unsafe {
        std::ptr::write(
            out,
            JadeNativePkg {
                name: PKG_NAME.as_ptr().cast::<c_char>(),
                bindings: BINDINGS.as_ptr(),
                binding_count: BINDINGS.len(),
            },
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read a struct value back out of the FFI representation, the way the compiler
    /// does. Mirrors `ffi_to_vm`'s struct arm closely enough to catch a wrong tag,
    /// a wrong type name, or a field written under the wrong key.
    unsafe fn read_struct(v: &JadeVal) -> (String, Vec<(String, u8)>) {
        assert_eq!(v.tag, JADE_TAG_STRUCT, "frame must cross as a struct, not a dict");
        let s = unsafe { &*v.data.as_struct };
        let name = unsafe { read_cstr(s.type_name) }.expect("type name");
        let mut fields = Vec::new();
        for i in 0..s.len {
            let k = unsafe { read_cstr(*s.keys.add(i)) }.expect("key");
            let t = unsafe { (*s.vals.add(i)).tag };
            fields.push((k, t));
        }
        (name, fields)
    }

    #[test]
    fn token_frame_carries_its_text_under_the_protocol_field_name() {
        let v = frame_to_val(Frame::Token("hi".into()));
        let (name, fields) = unsafe { read_struct(&v) };
        assert_eq!(name, "Token");
        assert_eq!(fields, vec![("text".to_owned(), JADE_TAG_STR)]);
        let s = unsafe { &*v.data.as_struct };
        let text = unsafe { read_cstr((*s.vals).data.as_str) };
        assert_eq!(text.as_deref(), Some("hi"));
    }

    #[test]
    fn every_frame_type_uses_the_name_the_compiler_reads() {
        // These five names and their field names are the protocol's `jade/infer.jde`.
        // A rename on either side decodes as a bag of nils rather than failing loudly,
        // which is exactly the silent-empty-reply failure the strict decoder exists to
        // prevent — so pin them.
        let cases = vec![
            (Frame::Token(String::new()), "Token", "text"),
            (Frame::Done { tokens_used: 0 }, "Done", "tokens_used"),
            (Frame::Error(String::new()), "Error", "message"),
            (Frame::Meta { provider: String::new() }, "Meta", "provider"),
            (Frame::Json(String::new()), "Json", "json"),
        ];
        for (frame, want_name, want_field) in cases {
            let v = frame_to_val(frame);
            let (name, fields) = unsafe { read_struct(&v) };
            assert_eq!(name, want_name);
            assert_eq!(fields[0].0, want_field, "field name for {want_name}");
        }
    }

    #[test]
    fn done_saturates_rather_than_going_negative() {
        // tokens_used is u64 and crosses as i64. Wrapping would report a negative
        // token count, which reads as corruption rather than as a clamp.
        let v = frame_to_val(Frame::Done { tokens_used: u64::MAX });
        let s = unsafe { &*v.data.as_struct };
        let n = unsafe { (*s.vals).data.as_int };
        assert_eq!(n, i64::MAX);
    }

    #[test]
    fn request_fields_are_read_by_name_and_nil_stays_none() {
        // Built the way the compiler builds it: type InferRequest, four fields, the
        // three optional ones nil.
        let req = val_struct(
            "InferRequest",
            vec![
                ("input", val_str("say hi")),
                ("grammar", val_nil()),
                ("anchor", val_nil()),
                ("stop_anchor", val_str("</tool>")),
            ],
        );
        assert_eq!(unsafe { field_str(&req, "input") }.as_deref(), Some("say hi"));
        assert_eq!(unsafe { field_str(&req, "grammar") }, None);
        assert_eq!(unsafe { field_str(&req, "anchor") }, None);
        assert_eq!(unsafe { field_str(&req, "stop_anchor") }.as_deref(), Some("</tool>"));
        // A field the request does not carry must read as absent, not panic.
        assert_eq!(unsafe { field_str(&req, "nonexistent") }, None);
    }

    #[test]
    fn an_array_of_frames_reports_its_length() {
        let arr = val_array(vec![
            frame_to_val(Frame::Token("a".into())),
            frame_to_val(Frame::Done { tokens_used: 1 }),
        ]);
        assert_eq!(arr.tag, JADE_TAG_ARRAY);
        assert_eq!(unsafe { (*arr.data.as_arr).len }, 2);
    }

    #[test]
    fn value_abi_matches_what_the_compiler_requires() {
        // The compiler refuses a package whose value ABI it cannot talk to. 2 is the
        // version structs cross under, introduced in jade v1.1.31.
        assert_eq!(JADE_VALUE_ABI, 2);
    }
}
