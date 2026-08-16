//! The Rust half of the MoE expert-offload FFI.
//!
//! `moe_driver_init` is a C symbol compiled into libllama by the driver's CMake
//! wiring (`src/infer-engine/moe-driver/`). This module owns the `#[repr(C)]`
//! mirror of `moe_driver_config` and the one `unsafe` call, so the rest of the
//! crate hands over a validated [`ExpertOrder`] and a [`MoeOffload`] and never
//! touches a raw pointer.
//!
//! It used to live inline in `engine/mod.rs`, next to the model and context
//! lifecycle. That put the only hand-written `extern "C"` block in the crate in
//! the file least about FFI, and it meant the MoE capability was spread across
//! three places — the pure loader here, the declaration there, and the C++ in a
//! sibling directory. All of it belongs together.
//!
//! Linux-only and feature-gated: the symbol exists only in a build that linked
//! llama.cpp with the driver's seam compiled in.

use super::ExpertOrder;
use crate::config::MoeOffload;

/// Mirrors `moe_driver_config` in `moe_driver.h`. Field order and types are the
/// contract — changing either side alone corrupts the driver's config silently,
/// which is why the C header and this struct are named in each other's comments.
#[repr(C)]
struct MoeDriverConfig {
    enabled: bool,
    pin: i32,
    cache: i32,
    order_path: *const std::os::raw::c_char,
}

extern "C" {
    /// Returns 0 on success; non-zero leaves the driver inactive (stock path).
    fn moe_driver_init(
        cfg: *const MoeDriverConfig,
        n_layers: i32,
        n_experts: i32,
        n_expert_used: i32,
        expert_order_flat: *const i32,
    ) -> i32;

    /// Returns 1 in a build whose llama.cpp carries the `build_moe_ffn` seam.
    ///
    /// Defined by the capability bundle's first patch, in `llama-graph.cpp`, right
    /// beside the seam it reports on — so the symbol exists exactly when the seam
    /// does. Deliberately *not* declared in `moe_driver.h`: the non-CUDA build
    /// compiles `moe_stub.cpp`, which implements that whole header, and a stub
    /// definition would satisfy this while proving nothing.
    ///
    /// This is the last of three guards against the quietest failure in the build —
    /// llama.cpp compiled without the seam. `build.rs` panics if a patch fails to
    /// apply; a missing symbol here fails the link or the `dlopen`; and this call
    /// catches the remainder. All three exist because a seam that is inert when its
    /// driver is off looks exactly like a seam that was never compiled in.
    fn moe_driver_seam_present() -> i32;
}

/// Experts selected per token by the router. Qwen3-Coder-30B-A3B uses 8; the value
/// is passed to the driver so it can size its per-token working set.
const N_EXPERT_USED: i32 = 8;

/// Activate the driver for a loaded model.
///
/// `Ok(())` means the driver reported itself active and the llama graph seam will
/// route through it. `Err` means it declined — the caller decides whether that is
/// fatal. It is: a driver that quietly does not engage looks identical to one that
/// does, except for three times the VRAM, so the load fails rather than running
/// stock under a config that asked for offload.
pub(crate) fn init(cfg: &MoeOffload, order: &ExpertOrder) -> Result<(), String> {
    // SAFETY: a zero-argument C function returning int, defined by the bundle patch.
    if unsafe { moe_driver_seam_present() } != 1 {
        return Err(
            "the llama.cpp this engine links has no build_moe_ffn seam, so the driver \
             would be initialised and never called. The `moe` capability bundle did not \
             make it into the build — run `python3 tools/scripts/check-bundles.py`."
                .to_string(),
        );
    }

    let mc = MoeDriverConfig {
        enabled: true,
        pin: cfg.pin as i32,
        cache: cfg.cache as i32,
        // The driver reads the order from the flat array below, not from a path —
        // the JSON is parsed and validated on this side so the C++ never sees it.
        order_path: std::ptr::null(),
    };

    // SAFETY: `mc` and `order.flat()` both outlive the call, the flat slice is
    // exactly `n_layers * n_experts` i32s (checked by `ExpertOrder::load`), and the
    // driver copies what it needs before returning.
    let rc = unsafe {
        moe_driver_init(
            &mc,
            order.n_layers as i32,
            order.n_experts as i32,
            N_EXPERT_USED,
            order.flat().as_ptr(),
        )
    };

    if rc == 0 {
        Ok(())
    } else {
        Err(format!("moe_driver_init returned {rc}"))
    }
}
