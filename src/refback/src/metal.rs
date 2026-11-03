//! Author: Joe Ricks
//! File: src/refback/src/metal.rs
//! Description: Sets up Apple plan regions as Metal buffers and runs the compute check's
//! matvec as a Metal shader bound at (buffer, offset).

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSPageSize, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::quant;

/// Purpose: This type alias names a Metal buffer refback holds.
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// The shader source, compiled when the GPU is set up.
const SOURCE: &str = include_str!("matvec.metal");

/// Purpose: This struct holds the shape the matvec kernels read from buffer 3. It
/// matches `Shape` in `matvec.metal` byte for byte.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Shape {
    /// Elements in one row of W, and the length of x.
    cols: u32,
    /// Rows of W, and the length of y.
    rows: u32,
    /// Bytes in one row of W.
    row_bytes: u64,
}

/// Purpose: This struct holds the Metal GPU: its device, a command queue and one matvec
/// pipeline per covered weight type.
pub struct Gpu {
    /// The default Metal device.
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// The queue every matvec is sent on.
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Each covered `ggml_type` with its compiled kernel.
    kernels: Vec<(u32, Retained<ProtocolObject<dyn MTLComputePipelineState>>)>,
}

/// Purpose: This function returns the VM page size, which every no-copy buffer's start
/// and length must be a multiple of.
///
/// Inputs: none.
///
/// Outputs: the page size in bytes: 16 KiB on Apple silicon.
pub fn page_bytes() -> usize
{
    NSPageSize()
}

/// Purpose: This function returns the address of a buffer's memory, as the CPU sees it.
///
/// Inputs:
/// - `b`: the buffer.
///
/// Outputs: the address.
pub fn contents(
    b: &Buffer,
) -> NonNull<u8>
{
    b.contents().cast()
}

impl Gpu {
    /// Purpose: This function opens the default Metal GPU and compiles the matvec
    /// kernels.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the [`Gpu`], or a message saying what is missing or what failed to
    /// compile.
    pub fn new() -> Result<Gpu, String>
    {
        let device = MTLCreateSystemDefaultDevice().ok_or("this Mac has no Metal GPU")?;
        let queue = device.newCommandQueue().ok_or("Metal gave no command queue")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), None)
            .map_err(|e| format!("matvec.metal does not compile: {}", e.localizedDescription()))?;
        let mut kernels = Vec::new();
        for (ty, name) in [(quant::F32, "matvec_f32"), (quant::F16, "matvec_f16"), (quant::Q8_0, "matvec_q8_0"), (quant::Q4_0, "matvec_q4_0")] {
            let f = library.newFunctionWithName(&NSString::from_str(name)).ok_or_else(|| format!("matvec.metal has no kernel {name}"))?;
            let p = device.newComputePipelineStateWithFunction_error(&f).map_err(|e| format!("{name}: {}", e.localizedDescription()))?;
            kernels.push((ty, p));
        }
        Ok(Gpu { device, queue, kernels })
    }

    /// Purpose: This function allocates a buffer that the CPU and the GPU both address.
    ///
    /// Inputs:
    /// - `len`: its size in bytes, at least 1.
    ///
    /// Outputs: the buffer, or a message when Metal cannot allocate it.
    pub fn shared(
        &self,
        len: usize,
    ) -> Result<Buffer, String>
    {
        self.device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("Metal cannot allocate a shared buffer of {len} bytes"))
    }

    /// Purpose: This function wraps existing memory in a Metal buffer without copying it,
    /// as ggml's Metal backend wraps a mapped model file.
    ///
    /// Inputs:
    /// - `ptr`: the memory's start, on a page boundary.
    /// - `len`: its length, a whole number of pages.
    ///
    /// Outputs: the buffer, or a message when Metal refuses the memory.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` must be mapped memory that stays mapped, and unchanged, for as
    /// long as the buffer lives.
    pub unsafe fn no_copy(
        &self,
        ptr: NonNull<u8>,
        len: usize,
    ) -> Result<Buffer, String>
    {
        // SAFETY: this function's contract keeps the memory mapped while the buffer
        // lives. No deallocator is given, so Metal never frees memory refback owns.
        unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr.cast::<c_void>(),
                len,
                MTLResourceOptions::StorageModeShared,
                None,
            )
        }
        .ok_or_else(|| {
            format!(
                "Metal refuses a no-copy buffer of {len} bytes at {:p}; its largest buffer is {} bytes",
                ptr.as_ptr(),
                self.device.maxBufferLength()
            )
        })
    }

    /// Purpose: This function computes y = W·x on the GPU, with W bound at its planned
    /// buffer and offset.
    ///
    /// Inputs:
    /// - `w`: the buffer W's region was set up as.
    /// - `offset`: W's offset in that buffer, from the plan.
    /// - `ty`: W's `ggml_type`, one [`quant::covered`] accepts.
    /// - `cols`: the elements in a row of W.
    /// - `rows`: the rows of W.
    /// - `x`: the input vector, `cols` long.
    ///
    /// Outputs: y, or a message when the range is outside the buffer or the GPU fails.
    pub fn matvec(
        &self,
        w: &Buffer,
        offset: u64,
        ty: u32,
        cols: usize,
        rows: usize,
        x: &[f32],
    ) -> Result<Vec<f32>, String>
    {
        let kernel = self.kernels.iter().find(|(t, _)| *t == ty).map(|(_, k)| k).ok_or("no Metal kernel for this type")?;
        let row_bytes = quant::row_bytes(ty, cols).ok_or("a row is not whole blocks")?;
        let offset = usize::try_from(offset).map_err(|_| "the offset does not fit in memory")?;
        let end = rows.checked_mul(row_bytes).and_then(|n| n.checked_add(offset));
        if end.is_none_or(|e| e > w.length()) || x.len() != cols {
            return Err(format!("W at offset {offset} with {rows} rows of {row_bytes} bytes does not fit its {}-byte buffer", w.length()));
        }
        if rows == 0 {
            return Ok(Vec::new());
        }
        let xb = self.shared((cols * 4).max(1))?;
        let yb = self.shared(rows * 4)?;
        // SAFETY: `xb` holds at least `cols` f32 values, and a shared buffer's contents
        // are page-aligned, so they are aligned for f32. Nothing else uses `xb` yet.
        unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), xb.contents().as_ptr().cast::<f32>(), cols) };
        let shape = Shape {
            cols: u32::try_from(cols).map_err(|_| "too many columns for one kernel")?,
            rows: u32::try_from(rows).map_err(|_| "too many rows for one kernel")?,
            row_bytes: row_bytes as u64,
        };

        let cb = self.queue.commandBuffer().ok_or("Metal gave no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or("Metal gave no compute encoder")?;
        enc.setComputePipelineState(kernel);
        // SAFETY: indices 0 to 3 are the kernel's [[buffer(n)]] arguments, with the types
        // matvec.metal declares. W's range was checked against its buffer above, and x
        // and y were sized for `cols` and `rows`. Every buffer outlives the command
        // buffer, which is waited on below. `shape` is copied by Metal at this call.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(w), offset, 0);
            enc.setBuffer_offset_atIndex(Some(&xb), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&yb), 0, 2);
            enc.setBytes_length_atIndex(NonNull::from(&shape).cast::<c_void>(), std::mem::size_of::<Shape>(), 3);
        }
        let width = kernel.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        enc.dispatchThreads_threadsPerThreadgroup(MTLSize { width: rows, height: 1, depth: 1 }, MTLSize { width, height: 1, depth: 1 });
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            let why = cb.error().map(|e| e.localizedDescription().to_string()).unwrap_or_else(|| "no reason given".into());
            return Err(format!("the GPU did not finish the matvec: {why}"));
        }
        // SAFETY: the command buffer has completed, so the GPU no longer writes `yb`,
        // which holds `rows` f32 values at a page-aligned address.
        Ok(unsafe { std::slice::from_raw_parts(yb.contents().as_ptr().cast::<f32>(), rows) }.to_vec())
    }
}
