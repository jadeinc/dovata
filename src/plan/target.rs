//! Author: Joe Ricks
//! File: src/plan/target.rs
//! Description: Describes the machine a plan is for and its kinds of memory.

use serde::Serialize;

/// Purpose: This enum describes the machine a plan is for, with at most one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Target {
    /// Apple Silicon. CPU and GPU share one memory pool, so nothing is copied to a
    /// device: weights stay in the mapped GGUF and the GPU reads them there.
    Apple {
        /// Memory the plan may use, in bytes. The Jade compiler decides how much of the
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
    /// Purpose: This function builds an Apple Silicon target with 16 KiB pages.
    ///
    /// Inputs:
    /// - `unified_bytes`: the memory the plan may use.
    /// - `max_buffer_bytes`: the largest single GPU buffer.
    ///
    /// Outputs: a [`Target::Apple`].
    pub fn apple(
        unified_bytes: u64,
        max_buffer_bytes: u64,
    ) -> Target
    {
        Target::Apple { unified_bytes, max_buffer_bytes, page_bytes: 16 * 1024 }
    }

    /// Purpose: This function builds a CPU-only target with 4 KiB pages.
    ///
    /// Inputs:
    /// - `ram_bytes`: the host memory the plan may allocate.
    ///
    /// Outputs: a [`Target::Cpu`].
    pub fn cpu(
        ram_bytes: u64,
    ) -> Target
    {
        Target::Cpu { ram_bytes, page_bytes: 4096 }
    }

    /// Purpose: This function builds a target with one NVIDIA GPU.
    ///
    /// Inputs:
    /// - `vram_bytes`: the VRAM the plan may use.
    ///
    /// Outputs: a [`Target::Cuda`].
    pub fn cuda(
        vram_bytes: u64,
    ) -> Target
    {
        Target::Cuda { vram_bytes }
    }

    /// Purpose: This function returns where the compiler runs the ops for this target.
    ///
    /// Inputs: none.
    ///
    /// Outputs: Metal, CPU or CUDA.
    pub fn site(&self) -> ExecSite
    {
        match self {
            Target::Apple { .. } => ExecSite::Metal,
            Target::Cpu { .. } => ExecSite::Cpu,
            Target::Cuda { .. } => ExecSite::Cuda,
        }
    }
}

/// Purpose: This enum lists where the compiled program runs the model's ops.
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

/// Purpose: This enum lists the kinds of memory a plan's regions can be.
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
    /// Purpose: This function returns the byte alignment for an allocated region of this
    /// kind.
    ///
    /// Inputs: none.
    ///
    /// Outputs: 256 for VRAM and unified memory, 64 for host memory, and 1 for a mapped
    /// window.
    pub fn alloc_align(self) -> u64
    {
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
    fn each_target_runs_ops_where_its_memory_is()
    {
        assert_eq!(Target::apple(1, 1).site(), ExecSite::Metal);
        assert_eq!(Target::cpu(1).site(), ExecSite::Cpu);
        assert_eq!(Target::cuda(1).site(), ExecSite::Cuda);
    }

    #[test]
    fn targets_serialize_with_their_kind()
    {
        let json = serde_json::to_string(&Target::cuda(24)).unwrap();
        assert_eq!(json, r#"{"kind":"cuda","vram_bytes":24}"#);
    }
}
