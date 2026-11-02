//! Author: Joe Ricks
//! File: src/machine/metal.rs
//! Description: Reads the Mac GPU's memory limits from Apple's Metal framework.

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

use super::MetalLimits;

/// Purpose: This function reads the default Metal GPU's memory limits.
///
/// Inputs: none.
///
/// Outputs: the working-set and buffer limits, or `None` when the Mac has no Metal GPU.
pub(super) fn limits() -> Option<MetalLimits>
{
    let device = MTLCreateSystemDefaultDevice()?;
    Some(MetalLimits {
        working_set_bytes: device.recommendedMaxWorkingSetSize(),
        max_buffer_bytes: u64::try_from(device.maxBufferLength()).unwrap_or(u64::MAX),
        unified: device.hasUnifiedMemory(),
    })
}
