//! The machine a plan is for, and the kinds of memory a plan can put tensors in.
//!
//! dovata does not probe the build machine. The Jade compiler may build on a laptop for
//! a GPU box, so the caller says what the target has, and the plan is only as right as
//! what it was told.

use serde::Serialize;

/// One machine with at most one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    /// Apple Silicon. CPU and GPU share one memory pool, so nothing is copied to a
    /// device: weights stay in the mapped GGUF and the GPU reads them there.
    Apple {
        /// Memory the plan may use, in bytes. The caller decides how much of the
        /// machine's RAM that is; Metal's `recommendedMaxWorkingSetSize` is the usual
        /// answer.
        unified_bytes: u64,
        /// The largest single GPU buffer: `MTLDevice.maxBufferLength`. Weights are
        /// bound through windows no larger than this.
        max_buffer_bytes: u64,
        /// The VM page size, which every window's start must be a multiple of. 16 KiB
        /// on Apple Silicon.
        page_bytes: u64,
    },
    /// CPU only. Weights stay in the mapped GGUF; the KV cache and scratch go in host
    /// memory.
    Cpu {
        /// Host memory the plan may allocate, in bytes, not counting the mapped file,
        /// which the page cache holds.
        ram_bytes: u64,
        /// The VM page size: 4 KiB on x86-64, and 4, 16 or 64 KiB on aarch64 Linux.
        page_bytes: u64,
    },
    /// One NVIDIA GPU. Everything goes in VRAM. A model that does not fit is refused
    /// for now: placing part of it in host memory, and paging MoE experts, is deferred.
    Cuda {
        /// VRAM the plan may use, in bytes, after the CUDA context and anything else
        /// on the card.
        vram_bytes: u64,
    },
}

impl Target {
    /// An Apple target with the usual page size.
    pub fn apple(unified_bytes: u64, max_buffer_bytes: u64) -> Target {
        Target::Apple { unified_bytes, max_buffer_bytes, page_bytes: 16 * 1024 }
    }

    /// A CPU target with 4 KiB pages.
    pub fn cpu(ram_bytes: u64) -> Target {
        Target::Cpu { ram_bytes, page_bytes: 4096 }
    }

    /// A CUDA target.
    pub fn cuda(vram_bytes: u64) -> Target {
        Target::Cuda { vram_bytes }
    }

    /// Where the compiler runs the ops for this target.
    pub fn site(&self) -> ExecSite {
        match self {
            Target::Apple { .. } => ExecSite::Metal,
            Target::Cpu { .. } => ExecSite::Cpu,
            Target::Cuda { .. } => ExecSite::Cuda,
        }
    }
}

/// Where ops execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecSite {
    /// The compiler's CPU backend.
    Cpu,
    /// NVPTX kernels on the GPU.
    Cuda,
    /// Metal shaders on the GPU.
    Metal,
}

/// What kind of memory a region is, which decides how the program gets its base address
/// at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RegionKind {
    /// A window of a mapped GGUF part file. The program maps each part once, and the
    /// region's base is the mapping's base plus the window's file start. On Apple the
    /// window is also wrapped as a no-copy `MTLBuffer`.
    Mmap,
    /// A host allocation.
    Host,
    /// A device allocation, such as `cuMemAlloc`.
    Vram,
    /// An allocation both the CPU and an Apple GPU address, such as a shared
    /// `MTLBuffer`.
    Unified,
}

impl RegionKind {
    /// The alignment every entry in an allocated region of this kind gets, and the
    /// alignment the program must give the region's base.
    ///
    /// 256 bytes for GPU memory: `cuMemAlloc` returns 256-byte-aligned memory, and it
    /// covers every vector width a kernel uses. 64 bytes for host memory, one cache
    /// line. Mapped windows have no allocation alignment of their own; their entries sit
    /// at the GGUF's file offsets, which are aligned to the file's `general.alignment`.
    pub fn alloc_align(self) -> u64 {
        match self {
            RegionKind::Vram | RegionKind::Unified => 256,
            RegionKind::Host => 64,
            RegionKind::Mmap => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_target_runs_ops_where_its_memory_is() {
        assert_eq!(Target::apple(1, 1).site(), ExecSite::Metal);
        assert_eq!(Target::cpu(1).site(), ExecSite::Cpu);
        assert_eq!(Target::cuda(1).site(), ExecSite::Cuda);
    }

    #[test]
    fn targets_serialize_with_their_kind() {
        let json = serde_json::to_string(&Target::cuda(24)).unwrap();
        assert_eq!(json, r#"{"kind":"cuda","vram_bytes":24}"#);
    }
}
