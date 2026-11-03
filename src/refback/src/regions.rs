//! Author: Joe Ricks
//! File: src/refback/src/regions.rs
//! Description: Sets up each region of a plan once, the way the program dovata plans for
//! does at startup, and gives bounds-checked access to the bytes in them.

use std::alloc::Layout;
use std::fs::File;
use std::path::PathBuf;
use std::ptr::NonNull;

use dovata::memplan::{Backing, Plan, Region};
use dovata::target::{RegionKind, Target};
use memmap2::Mmap;

use crate::{Check, Outcome, StartError};

/// Purpose: This enum lists what owns one region's memory, and so what frees it.
enum Owner {
    /// A window of a part file's mapping. [`Memory`] owns the mapping.
    Mapped,
    /// A host allocation made with this layout, freed when the area is dropped.
    Host(Layout),
    /// A Metal buffer: shared for an allocated region, no-copy for a mapped window.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Buffer(crate::metal::Buffer),
}

/// Purpose: This struct holds one region as set up: its base address, its length and
/// what owns it.
struct Area {
    /// The region's base address.
    base: NonNull<u8>,
    /// The region's size in bytes, as the plan gives it.
    len: usize,
    /// Whether refback may write it: allocated regions yes, mapped weights never.
    writable: bool,
    /// What owns the memory.
    owner: Owner,
}

impl Drop for Area {
    /// Purpose: This function frees a host allocation; mappings and Metal buffers free
    /// themselves.
    ///
    /// Inputs: none.
    ///
    /// Outputs: none.
    fn drop(&mut self)
    {
        if let Owner::Host(layout) = self.owner {
            // SAFETY: `base` came from `alloc_zeroed` with exactly this layout, and an
            // area is dropped once, so it is freed once.
            unsafe { std::alloc::dealloc(self.base.as_ptr(), layout) }
        }
    }
}

/// Purpose: This struct holds a plan's memory once set up: every region, the part files'
/// mappings, and the GPU on Apple.
pub struct Memory {
    /// One per region, in id order. Declared before `maps` so it is dropped first: a
    /// mapped area, and its no-copy Metal buffer, point into a mapping.
    areas: Vec<Area>,
    /// The Metal GPU, on an Apple plan.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    gpu: Option<crate::metal::Gpu>,
    /// Each part file mapped once, read-only, part 1 first.
    maps: Vec<Mmap>,
    /// Each part file open, for reads that do not go through the mapping.
    files: Vec<File>,
}

impl Memory {
    /// Purpose: This function sets up every region of a plan once.
    ///
    /// Inputs:
    /// - `parts`: the model's part files, part 1 first.
    /// - `plan`: the plan.
    ///
    /// Outputs: the [`Memory`] and one [`Check::Region`] outcome per region, saying
    /// whether its base meets its alignment; or a [`StartError`] when a file cannot be
    /// mapped or a region cannot be set up as described.
    pub fn set_up(
        parts: &[PathBuf],
        plan: &Plan,
    ) -> Result<(Memory, Vec<Outcome>), StartError>
    {
        let apple = match plan.target {
            Target::Cpu { .. } => false,
            Target::Apple { .. } => true,
            Target::Cuda { .. } => {
                return Err(StartError::Unsupported(
                    "the plan is for CUDA, which refback does not cover yet: this Mac has no NVIDIA GPU. \
                     src/refback/README.md says where CUDA goes"
                        .into(),
                ))
            }
        };
        if apple && !cfg!(all(feature = "metal", target_os = "macos")) {
            return Err(StartError::Unsupported(
                "the plan is for Apple's Metal, and refback was built without its `metal` feature; \
                 build it with --features metal on a Mac"
                    .into(),
            ));
        }

        let mut files = Vec::new();
        let mut maps = Vec::new();
        for p in parts {
            let io = |e: std::io::Error| StartError::Io(format!("{}: {e}", p.display()));
            let file = File::open(p).map_err(io)?;
            // SAFETY: the mapping is read-only, and refback never writes the file. Like
            // every reader of a mapped file, it trusts that no other process truncates or
            // rewrites the model while it runs.
            let map = unsafe { Mmap::map(&file) }.map_err(io)?;
            files.push(file);
            maps.push(map);
        }

        let mut memory = Memory {
            areas: Vec::new(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            gpu: if apple { Some(crate::metal::Gpu::new().map_err(StartError::Region)?) } else { None },
            maps,
            files,
        };
        let mut out = Vec::new();
        for (i, r) in plan.regions.iter().enumerate() {
            if usize::from(r.id) != i {
                return Err(StartError::Region(format!("region {} is listed at index {i}", r.id)));
            }
            let area = memory.area(r, apple)?;
            let base = area.base.as_ptr() as usize as u64;
            let subject = format!("region {}", r.id);
            out.push(if r.align != 0 && base.is_multiple_of(r.align) {
                Outcome::pass(Check::Region, subject, format!("{} of {} bytes at {base:#x}, aligned to {}", describe(r), r.bytes, r.align))
            } else {
                Outcome::fail(Check::Region, subject, format!("{} at {base:#x} is not aligned to {}", describe(r), r.align))
            });
            memory.areas.push(area);
        }
        Ok((memory, out))
    }

    /// Purpose: This function sets up one region.
    ///
    /// Inputs:
    /// - `r`: the region, as the plan gives it.
    /// - `apple`: whether the plan is for Apple's Metal.
    ///
    /// Outputs: the [`Area`], or a [`StartError::Region`] saying why it cannot be set up.
    fn area(
        &self,
        r: &Region,
        apple: bool,
    ) -> Result<Area, StartError>
    {
        let bad = |m: String| StartError::Region(format!("region {}: {m}", r.id));
        let len = usize::try_from(r.bytes).map_err(|_| bad(format!("{} bytes do not fit in memory", r.bytes)))?;
        match (r.backing, r.kind) {
            (Backing::File { part, file_start }, RegionKind::Mmap) => {
                let map = self.maps.get(usize::from(part)).ok_or_else(|| bad(format!("it maps part {}, and the model has {}", part + 1, self.maps.len())))?;
                let end = file_start.checked_add(r.bytes).filter(|e| *e <= map.len() as u64);
                let start = usize::try_from(file_start).ok().filter(|_| end.is_some()).ok_or_else(|| {
                    bad(format!("its window {file_start}..{} runs past the end of part {}, which is {} bytes", file_start.saturating_add(r.bytes), part + 1, map.len()))
                })?;
                // SAFETY: `start + len` is at most the mapping's length, checked above, so
                // the pointer stays inside the mapping.
                let ptr = unsafe { map.as_ptr().add(start) }.cast_mut();
                let base = NonNull::new(ptr).ok_or_else(|| bad("the mapping is at address 0".into()))?;
                if apple {
                    return self.no_copy(r, base, len, map.len() - start);
                }
                Ok(Area { base, len, writable: false, owner: Owner::Mapped })
            }
            (Backing::File { .. }, kind) => Err(bad(format!("it is a window of a part file, so it should be mmap, not {kind:?}"))),
            (Backing::Alloc, RegionKind::Vram) => Err(StartError::Unsupported(format!(
                "region {} is VRAM, which needs CUDA; refback does not cover CUDA yet",
                r.id
            ))),
            (Backing::Alloc, RegionKind::Host) if !apple => {
                let layout = Layout::from_size_align(len.max(1), usize::try_from(r.align).unwrap_or(0))
                    .map_err(|_| bad(format!("its alignment {} is not a power of two", r.align)))?;
                // SAFETY: the layout's size is at least 1, as `alloc_zeroed` requires.
                let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
                let base = NonNull::new(ptr).ok_or_else(|| bad(format!("the host has no {len} bytes to allocate")))?;
                Ok(Area { base, len, writable: true, owner: Owner::Host(layout) })
            }
            (Backing::Alloc, RegionKind::Unified) if apple => self.shared(r, len),
            (Backing::Alloc, kind) => Err(bad(format!("it is {kind:?} memory, which a {} plan does not allocate", if apple { "Metal" } else { "CPU" }))),
        }
    }

    /// Purpose: This function wraps a mapped window in a no-copy Metal buffer.
    ///
    /// Inputs:
    /// - `r`: the region.
    /// - `base`: the window's start in the mapping.
    /// - `len`: the window's length.
    /// - `rest`: the bytes of the mapping from `base` to its end.
    ///
    /// Outputs: the [`Area`], whose base is the buffer's contents, or a [`StartError`].
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn no_copy(
        &self,
        r: &Region,
        base: NonNull<u8>,
        len: usize,
        rest: usize,
    ) -> Result<Area, StartError>
    {
        let bad = |m: String| StartError::Region(format!("region {}: {m}", r.id));
        let gpu = self.gpu.as_ref().ok_or_else(|| bad("no GPU was set up".into()))?;
        // ggml_backend_metal_buffer_from_ptr, in ggml-metal-device.m, rounds a no-copy
        // buffer's length up to whole pages. The mapping's last page is mapped in full,
        // so the rounded window still lies inside it.
        let page = crate::metal::page_bytes();
        let pages = len.div_ceil(page) * page;
        if pages > rest.div_ceil(page) * page {
            return Err(bad(format!("its window rounded to {pages} bytes of pages runs past the mapping")));
        }
        if !(base.as_ptr() as usize).is_multiple_of(page) {
            return Err(bad(format!("its window starts at {:#x}, which is not on a {page}-byte page, so Metal cannot wrap it", base.as_ptr() as usize)));
        }
        // SAFETY: `base..base + pages` is page-aligned and inside the mapping, checked
        // above. The mapping outlives the buffer, because `Memory` drops `areas` first.
        let buffer = unsafe { gpu.no_copy(base, pages) }.map_err(bad)?;
        let contents = crate::metal::contents(&buffer);
        if contents != base {
            return Err(bad(format!("Metal placed the no-copy buffer at {:p}, not at the mapping's {:p}", contents.as_ptr(), base.as_ptr())));
        }
        Ok(Area { base: contents, len, writable: false, owner: Owner::Buffer(buffer) })
    }

    /// Purpose: This function stands in for the no-copy buffer on a build without Metal.
    ///
    /// Inputs:
    /// - `r`: the region.
    /// - `_base`, `_len`, `_rest`: unused.
    ///
    /// Outputs: always a [`StartError::Unsupported`]; `set_up` refuses an Apple plan
    /// before it gets here.
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn no_copy(
        &self,
        r: &Region,
        _base: NonNull<u8>,
        _len: usize,
        _rest: usize,
    ) -> Result<Area, StartError>
    {
        Err(StartError::Unsupported(format!("region {} needs Metal, and refback was built without it", r.id)))
    }

    /// Purpose: This function allocates an allocated region of an Apple plan as a shared
    /// Metal buffer.
    ///
    /// Inputs:
    /// - `r`: the region.
    /// - `len`: its size in bytes.
    ///
    /// Outputs: the [`Area`], whose base is the buffer's contents, or a [`StartError`].
    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn shared(
        &self,
        r: &Region,
        len: usize,
    ) -> Result<Area, StartError>
    {
        let bad = |m: String| StartError::Region(format!("region {}: {m}", r.id));
        let gpu = self.gpu.as_ref().ok_or_else(|| bad("no GPU was set up".into()))?;
        let buffer = gpu.shared(len.max(1)).map_err(bad)?;
        Ok(Area { base: crate::metal::contents(&buffer), len, writable: true, owner: Owner::Buffer(buffer) })
    }

    /// Purpose: This function stands in for the shared buffer on a build without Metal.
    ///
    /// Inputs:
    /// - `r`: the region.
    /// - `_len`: unused.
    ///
    /// Outputs: always a [`StartError::Unsupported`].
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    fn shared(
        &self,
        r: &Region,
        _len: usize,
    ) -> Result<Area, StartError>
    {
        Err(StartError::Unsupported(format!("region {} needs Metal, and refback was built without it", r.id)))
    }

    /// Purpose: This function returns a region's base address.
    ///
    /// Inputs:
    /// - `region`: the region's id.
    ///
    /// Outputs: the address, or `None` for a region the plan does not have.
    pub fn base(
        &self,
        region: u16,
    ) -> Option<u64>
    {
        self.areas.get(usize::from(region)).map(|a| a.base.as_ptr() as usize as u64)
    }

    /// Purpose: This function finds the area a range of a region lies in, checking the
    /// range against the region's size.
    ///
    /// Inputs:
    /// - `region`: the region's id.
    /// - `offset`: bytes from the region's base.
    /// - `len`: the length of the range.
    ///
    /// Outputs: the area and the range as `usize`, or a message saying why the range is
    /// not inside the region.
    fn range(
        &self,
        region: u16,
        offset: u64,
        len: u64,
    ) -> Result<(&Area, usize, usize), String>
    {
        let a = self.areas.get(usize::from(region)).ok_or_else(|| format!("region {region} does not exist"))?;
        let end = offset.checked_add(len).ok_or_else(|| format!("{offset} + {len} overflows"))?;
        if end > a.len as u64 {
            return Err(format!("bytes {offset}..{end} run past region {region}, which is {} bytes", a.len));
        }
        // Both fit in usize, since `end` is at most `a.len`, which is a usize.
        Ok((a, offset as usize, len as usize))
    }

    /// Purpose: This function reads a range of a region.
    ///
    /// Inputs:
    /// - `region`: the region's id.
    /// - `offset`: bytes from the region's base.
    /// - `len`: the length of the range.
    ///
    /// Outputs: the bytes, or a message saying why the range is not inside the region.
    pub fn bytes(
        &self,
        region: u16,
        offset: u64,
        len: u64,
    ) -> Result<&[u8], String>
    {
        let (a, off, len) = self.range(region, offset, len)?;
        // SAFETY: `off + len` is at most the area's length, checked by `range`, and the
        // area is valid memory for its length while `self` lives. Writes go only through
        // `bytes_mut`, which takes `&mut self`, so nothing writes while this borrow
        // lives. The GPU only reads weights and writes its own output buffers.
        Ok(unsafe { std::slice::from_raw_parts(a.base.as_ptr().add(off), len) })
    }

    /// Purpose: This function gives write access to a range of an allocated region.
    ///
    /// Inputs:
    /// - `region`: the region's id.
    /// - `offset`: bytes from the region's base.
    /// - `len`: the length of the range.
    ///
    /// Outputs: the bytes, or a message saying why the range is not inside a writable
    /// region.
    pub fn bytes_mut(
        &mut self,
        region: u16,
        offset: u64,
        len: u64,
    ) -> Result<&mut [u8], String>
    {
        let (a, off, len) = self.range(region, offset, len)?;
        if !a.writable {
            return Err(format!("region {region} is a mapped window of the model file, which nothing may write"));
        }
        // SAFETY: as in `bytes`, the range is inside the area. `&mut self` means no other
        // slice of any area is alive, and the area is an allocation refback owns.
        Ok(unsafe { std::slice::from_raw_parts_mut(a.base.as_ptr().add(off), len) })
    }

    /// Purpose: This function returns a part file, for reads that do not go through the
    /// mapping.
    ///
    /// Inputs:
    /// - `part`: the part's index, from 0.
    ///
    /// Outputs: the open file, or `None` for a part the model does not have.
    pub fn file(
        &self,
        part: u16,
    ) -> Option<&File>
    {
        self.files.get(usize::from(part))
    }

    /// Purpose: This function returns the Metal GPU of an Apple plan.
    ///
    /// Inputs: none.
    ///
    /// Outputs: the GPU, or `None` on a CPU plan.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn gpu(&self) -> Option<&crate::metal::Gpu>
    {
        self.gpu.as_ref()
    }

    /// Purpose: This function returns the Metal buffer a region was set up as.
    ///
    /// Inputs:
    /// - `region`: the region's id.
    ///
    /// Outputs: the buffer, or `None` on a CPU plan or for a region the plan does not
    /// have.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub fn buffer(
        &self,
        region: u16,
    ) -> Option<&crate::metal::Buffer>
    {
        match &self.areas.get(usize::from(region))?.owner {
            Owner::Buffer(b) => Some(b),
            _ => None,
        }
    }
}

/// Purpose: This function says in a few words what a region is.
///
/// Inputs:
/// - `r`: the region.
///
/// Outputs: a phrase such as `a window of part 1 from byte 0` or `host memory`.
fn describe(
    r: &Region,
) -> String
{
    match (r.backing, r.kind) {
        (Backing::File { part, file_start }, _) => format!("a window of part {} from byte {file_start}", part + 1),
        (Backing::Alloc, RegionKind::Host) => "host memory".into(),
        (Backing::Alloc, RegionKind::Unified) => "a shared Metal buffer".into(),
        (Backing::Alloc, kind) => format!("{kind:?} memory"),
    }
}
