//! The address table: every tensor's region and offset, the thing the compiler bakes in.
//!
//! ## Why region and offset, not addresses
//!
//! An absolute address cannot be known at build time. The driver picks where
//! `cuMemAlloc` lands, ASLR moves every mapping, `cuMemAddressReserve` may ignore the
//! address it is asked for, and `MTLBuffer.gpuAddress` is read-only. So the plan works
//! the way a linker's relocations do: offsets are fixed now, and the program adds each
//! region's base once, at startup. On Metal it binds `(buffer, offset)` instead.
//!
//! ## What goes where
//!
//! | target | weights | KV cache and state | scratch |
//! |---|---|---|---|
//! | Apple | the mapped GGUF, in windows no larger than one Metal buffer | `Unified` | `Unified` |
//! | CPU | the mapped GGUF, one window per part | `Host` | `Host` |
//! | CUDA | copied into `Vram` | `Vram` | `Vram` |
//!
//! Weights are never copied on Apple or CPU. On Apple the GPU reads the mapping through
//! no-copy buffers, as ggml's Metal backend does. Nothing moves while the model runs.
//!
//! ## The invariants
//!
//! [`Plan::check`] asserts, on every plan this module returns:
//! - every offset is a multiple of its entry's alignment;
//! - every entry ends inside its region;
//! - a mapped entry sits at its file offset minus its window's start;
//! - two entries share bytes only if both are scratch and never alive at once, or one
//!   is a view of the other;
//! - every tensor in the GGUF has exactly one weight entry.
//!
//! A violation means a kernel reads the wrong bytes, which no later check would catch,
//! so the planner refuses to return such a plan rather than trusting itself.

use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::arena;
use super::target::{ExecSite, RegionKind, Target};
use super::window;
use crate::load::{Model, PartIdentity};

/// The version of the plan's JSON form. Bumped on any change the compiler must notice.
pub const SCHEMA_VERSION: u32 = 1;

/// A non-weight tensor for the planner, from the bound layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanTensor {
    /// Its name, unique across the model and the layout.
    pub name: String,
    /// What it is for, which decides where it goes.
    pub role: Role,
    /// Shape, fastest-varying first.
    pub dims: Vec<u64>,
    /// Its `ggml_type`.
    pub ggml_type: u32,
    /// Its size in bytes.
    pub bytes: u64,
    /// The first op that touches it. Inputs start at 0.
    pub first: u32,
    /// The last op that touches it, inclusive. Outputs end at the last op.
    pub last: u32,
}

/// What a tensor is for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Role {
    /// Model weights, read from the GGUF.
    Weight,
    /// Filled by the program before each step: token ids, positions, masks.
    Input,
    /// An intermediate result of one step.
    Scratch,
    /// Read by the program after a step, such as the logits.
    Output,
    /// The KV cache. Lives for the whole program.
    Kv,
    /// Recurrent state of a hybrid model's non-attention layers. Lives for the whole
    /// program, like the KV cache.
    State,
    /// A view of another tensor. It owns no memory; its address is the other tensor's
    /// plus a byte offset.
    View {
        /// The tensor it views.
        of: String,
        /// Where the view starts, in bytes from the start of `of`.
        byte_offset: u64,
    },
}

impl Role {
    fn persists(&self) -> bool {
        matches!(self, Role::Kv | Role::State)
    }

    fn per_step(&self) -> bool {
        matches!(self, Role::Input | Role::Scratch | Role::Output)
    }
}

/// How a region gets its memory at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Backing {
    /// A window of a mapped part file: base is the part's mapping plus `file_start`.
    File {
        /// Which part, from 0.
        part: u16,
        /// Where the window starts in that file. A multiple of the page size.
        file_start: u64,
    },
    /// An allocation of `bytes` aligned to `align`.
    Alloc,
}

/// One block of memory the program sets up once at startup.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Region {
    /// Dense from 0, assigned in a fixed order: weights, then persistent state, then
    /// scratch.
    pub id: u16,
    /// The kind of memory.
    pub kind: RegionKind,
    /// How the program gets it.
    pub backing: Backing,
    /// Its size in bytes.
    pub bytes: u64,
    /// The alignment its base must have. The program checks this at startup.
    pub align: u64,
}

/// Where a tensor's bytes come from, before the first step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Source {
    /// Copied from, or found at, this offset in this part file.
    File {
        /// Which part, from 0.
        part: u16,
        /// The tensor's data offset in that file.
        file_offset: u64,
    },
    /// Not loaded from anywhere: the program writes it.
    None,
}

/// One tensor's address.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AddressEntry {
    /// The tensor's name.
    pub tensor: String,
    /// What it is for.
    pub role: Role,
    /// The region it lives in.
    pub region: u16,
    /// Bytes from the region's base.
    pub offset: u64,
    /// Its size in bytes.
    pub bytes: u64,
    /// The alignment `offset` has.
    pub align: u64,
    /// Its `ggml_type`.
    pub ggml_type: u32,
    /// Shape, fastest-varying first.
    pub dims: Vec<u64>,
    /// Where its initial bytes come from.
    pub source: Source,
    /// For scratch, inputs and outputs, the first and last op that touch it. Scratch
    /// entries may share bytes only when these ranges do not overlap.
    pub live: Option<(u32, u32)>,
}

/// Which model a plan is for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelIdentity {
    /// `general.architecture`.
    pub architecture: String,
    /// Each part's file name and identity, part 1 first.
    pub parts: Vec<PartFile>,
}

/// One part file of the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PartFile {
    /// The file name, without its directory, which differs between machines.
    pub file_name: String,
    /// What the program checks before trusting the offsets.
    #[serde(flatten)]
    pub identity: PartIdentity,
}

/// Bytes the plan uses, by kind of memory.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Totals {
    /// Mapped GGUF bytes, counted once per file even where windows overlap.
    pub mapped: u64,
    /// Allocated bytes by kind.
    pub allocated: BTreeMap<String, u64>,
}

/// The plan: every tensor's region and offset on one target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    /// [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The machine.
    pub target: Target,
    /// Where ops run.
    pub site: ExecSite,
    /// The model.
    pub model: ModelIdentity,
    /// Regions in id order.
    pub regions: Vec<Region>,
    /// One entry per tensor, sorted by name.
    pub entries: Vec<AddressEntry>,
    /// Bytes used.
    pub totals: Totals,
    /// Things the caller should know that are not errors.
    pub notes: Vec<String>,
    /// SHA-256 over everything above, in lowercase hex. Two plans with the same hash
    /// give the compiler the same constants.
    pub table_hash: String,
}

/// Why a model could not be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    /// The model does not fit in the memory the target has.
    DoesNotFit { need: u64, have: u64, what: &'static str },
    /// On CUDA, the model does not fit in VRAM. Placing part of it in host memory is
    /// deferred.
    DoesNotFitVram { need: u64, have: u64 },
    /// One tensor is larger than a GPU buffer can be.
    TooLargeForBuffer { tensor: String, bytes: u64, limit: u64 },
    /// A weight's size cannot be computed, because its `ggml_type` is unknown or its
    /// shape does not divide into blocks.
    UnsizedWeight { tensor: String, ggml_type: u32 },
    /// A view names a tensor that does not exist, or views form a cycle.
    BadView { tensor: String, of: String },
    /// A tensor name is used twice.
    DuplicateName(String),
    /// The plan broke one of its own invariants. A bug in dovata.
    Invariant(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlanError::DoesNotFit { need, have, what } => {
                write!(f, "the model needs {} of {what} and the target has {}", gib(*need), gib(*have))
            }
            PlanError::DoesNotFitVram { need, have } => write!(
                f,
                "the model needs {} of VRAM and the target has {}. Placing part of a model in host \
                 memory, including MoE expert paging, is not supported yet; it is deferred until \
                 that design is settled. Use a smaller quantization or a GPU with more memory.",
                gib(*need),
                gib(*have)
            ),
            PlanError::TooLargeForBuffer { tensor, bytes, limit } => write!(
                f,
                "{tensor} needs {bytes} bytes in one GPU buffer, and the target's largest buffer is {limit} bytes"
            ),
            PlanError::UnsizedWeight { tensor, ggml_type } => write!(
                f,
                "{tensor} has ggml_type {ggml_type}, whose size dovata does not know; \
                 the type table in src/gguf/mod.rs needs extending"
            ),
            PlanError::BadView { tensor, of } => write!(f, "{tensor} is a view of {of}, which cannot be resolved"),
            PlanError::DuplicateName(n) => write!(f, "tensor name {n} is used twice"),
            PlanError::Invariant(msg) => write!(f, "dovata produced an invalid plan, which is a bug: {msg}"),
        }
    }
}

impl std::error::Error for PlanError {}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

/// Lay out a model's weights and a bound layout's other tensors on one target.
///
/// `others` holds every non-weight tensor the layout uses, with the op range each is
/// alive for. Weights come from the model itself, so every GGUF tensor gets an entry.
pub fn place(model: &Model, others: &[PlanTensor], target: &Target) -> Result<Plan, PlanError> {
    let mut regions: Vec<Region> = Vec::new();
    let mut entries: Vec<AddressEntry> = Vec::new();
    let mut notes: Vec<String> = Vec::new();
    let mut totals = Totals::default();

    // ── Weights ──────────────────────────────────────────────────────────────────
    let mut weights = Vec::new();
    for t in model.tensors() {
        let bytes = t.info.bytes().ok_or_else(|| PlanError::UnsizedWeight { tensor: t.info.name.clone(), ggml_type: t.info.kind })?;
        weights.push((t, bytes));
    }
    let file_align = model.meta().alignment;

    let mapped = match *target {
        Target::Apple { max_buffer_bytes, page_bytes, .. } => Some((max_buffer_bytes, page_bytes)),
        Target::Cpu { page_bytes, .. } => Some((u64::MAX, page_bytes)),
        Target::Cuda { .. } => None,
    };
    if let Some((limit, page_bytes)) = mapped {
        for part in 0..model.part_count() {
            let part16 = u16::try_from(part).map_err(|_| PlanError::Invariant("more than 65535 parts".into()))?;
            let mine: Vec<_> = weights.iter().filter(|(t, _)| usize::from(t.part) == part).collect();
            let spans: Vec<window::Span> = mine.iter().map(|(t, b)| window::Span { file_offset: t.file_offset, bytes: *b }).collect();
            let (wins, which) = window::windows(&spans, limit, page_bytes).map_err(|e| match e {
                window::Error::TensorTooLarge { index, bytes, limit } => {
                    PlanError::TooLargeForBuffer { tensor: mine[index].0.info.name.clone(), bytes, limit }
                }
            })?;
            let first_id = regions.len();
            for w in &wins {
                regions.push(Region {
                    id: next_id(&regions)?,
                    kind: RegionKind::Mmap,
                    backing: Backing::File { part: part16, file_start: w.file_start },
                    bytes: w.bytes,
                    align: page_bytes,
                });
            }
            for ((t, b), w) in mine.iter().zip(&which) {
                let file_start = wins[*w].file_start;
                entries.push(weight_entry(t, *b, regions[first_id + w].id, t.file_offset - file_start, file_align));
            }
            totals.mapped += model.identity()[part].file_bytes;
        }
    } else {
        // CUDA: copy every weight into one VRAM region, in file order so the startup
        // copy reads each file front to back.
        let align = RegionKind::Vram.alloc_align();
        let mut at = 0u64;
        let id = next_id(&regions)?;
        let mut ordered: Vec<_> = weights.iter().collect();
        ordered.sort_by_key(|(t, _)| (t.part, t.file_offset));
        for (t, b) in ordered {
            at = align_up(at, align);
            entries.push(weight_entry(t, *b, id, at, align));
            at += b;
        }
        regions.push(Region { id, kind: RegionKind::Vram, backing: Backing::Alloc, bytes: at, align });
    }

    // ── Everything else ──────────────────────────────────────────────────────────
    let mut names: Vec<&str> = model.tensors().map(|t| t.info.name.as_str()).chain(others.iter().map(|t| t.name.as_str())).collect();
    names.sort_unstable();
    if let Some(w) = names.windows(2).find(|w| w[0] == w[1]) {
        return Err(PlanError::DuplicateName(w[0].to_owned()));
    }

    let alloc_kind = match target {
        Target::Apple { .. } => RegionKind::Unified,
        Target::Cpu { .. } => RegionKind::Host,
        Target::Cuda { .. } => RegionKind::Vram,
    };
    let align = alloc_kind.alloc_align();
    let buffer_limit = max_buffer_bytes_of(target).unwrap_or(u64::MAX);

    // Persistent tensors, packed in layout order, which is layer order. On Apple a
    // region is one Metal buffer, so a new one starts when the next tensor would not fit.
    let mut current: Option<(u16, u64)> = None;
    for t in others.iter().filter(|t| t.role.persists()) {
        if t.bytes > buffer_limit {
            return Err(PlanError::TooLargeForBuffer { tensor: t.name.clone(), bytes: t.bytes, limit: buffer_limit });
        }
        let (id, at) = match current {
            Some((id, used)) if align_up(used, align) + t.bytes <= buffer_limit => (id, align_up(used, align)),
            _ => {
                let id = next_id(&regions)?;
                regions.push(Region { id, kind: alloc_kind, backing: Backing::Alloc, bytes: 0, align });
                (id, 0)
            }
        };
        entries.push(AddressEntry {
            tensor: t.name.clone(),
            role: t.role.clone(),
            region: id,
            offset: at,
            bytes: t.bytes,
            align,
            ggml_type: t.ggml_type,
            dims: t.dims.clone(),
            source: Source::None,
            live: None,
        });
        regions[usize::from(id)].bytes = at + t.bytes;
        current = Some((id, at + t.bytes));
    }

    // Per-step tensors share one arena.
    let step: Vec<&PlanTensor> = others.iter().filter(|t| t.role.per_step()).collect();
    if !step.is_empty() {
        let requests: Vec<arena::Request> = step
            .iter()
            .map(|t| arena::Request { name: t.name.clone(), bytes: t.bytes, align, first: t.first, last: t.last })
            .collect();
        let placed = arena::assign(&requests);
        arena::check(&requests, &placed).map_err(PlanError::Invariant)?;
        if placed.size > buffer_limit {
            return Err(PlanError::TooLargeForBuffer { tensor: "the scratch arena".into(), bytes: placed.size, limit: buffer_limit });
        }
        let id = next_id(&regions)?;
        regions.push(Region { id, kind: alloc_kind, backing: Backing::Alloc, bytes: placed.size, align });
        for (t, off) in step.iter().zip(&placed.offsets) {
            entries.push(AddressEntry {
                tensor: t.name.clone(),
                role: t.role.clone(),
                region: id,
                offset: *off,
                bytes: t.bytes,
                align,
                ggml_type: t.ggml_type,
                dims: t.dims.clone(),
                source: Source::None,
                live: Some((t.first, t.last)),
            });
        }
    }

    // Views take their target's region and offset. Resolved in passes so a view of a
    // view works whatever order the layout lists them in; a pass that resolves nothing
    // means a missing target or a cycle.
    let mut pending: Vec<&PlanTensor> = others.iter().filter(|t| matches!(t.role, Role::View { .. })).collect();
    while !pending.is_empty() {
        let before = pending.len();
        let mut still = Vec::new();
        for t in pending {
            let Role::View { of, byte_offset } = &t.role else { unreachable!("filtered to views") };
            match entries.iter().find(|e| &e.tensor == of) {
                Some(base) => {
                    let e = AddressEntry {
                        tensor: t.name.clone(),
                        role: t.role.clone(),
                        region: base.region,
                        offset: base.offset + byte_offset,
                        bytes: t.bytes,
                        align: 1,
                        ggml_type: t.ggml_type,
                        dims: t.dims.clone(),
                        source: Source::None,
                        live: base.live,
                    };
                    entries.push(e);
                }
                None => still.push(t),
            }
        }
        if still.len() == before {
            let t = still[0];
            let Role::View { of, .. } = &t.role else { unreachable!("filtered to views") };
            return Err(PlanError::BadView { tensor: t.name.clone(), of: of.clone() });
        }
        pending = still;
    }

    // ── Totals and fit ───────────────────────────────────────────────────────────
    for r in regions.iter().filter(|r| r.backing == Backing::Alloc) {
        *totals.allocated.entry(kind_name(r.kind).to_owned()).or_insert(0) += r.bytes;
    }
    let allocated: u64 = totals.allocated.values().sum();
    match *target {
        Target::Apple { unified_bytes, .. } => {
            let need = totals.mapped + allocated;
            if need > unified_bytes {
                return Err(PlanError::DoesNotFit { need, have: unified_bytes, what: "unified memory" });
            }
        }
        Target::Cpu { ram_bytes, .. } => {
            if allocated > ram_bytes {
                return Err(PlanError::DoesNotFit { need: allocated, have: ram_bytes, what: "host memory" });
            }
            if totals.mapped + allocated > ram_bytes {
                notes.push(format!(
                    "the mapped weights ({}) and allocations ({}) exceed the {} of RAM given; weights \
                     will be read from disk as they are used, which is slow",
                    gib(totals.mapped),
                    gib(allocated),
                    gib(ram_bytes)
                ));
            }
        }
        Target::Cuda { vram_bytes } => {
            if allocated > vram_bytes {
                return Err(PlanError::DoesNotFitVram { need: allocated, have: vram_bytes });
            }
        }
    }

    entries.sort_by(|a, b| a.tensor.cmp(&b.tensor));
    let model_id = ModelIdentity {
        architecture: model.architecture().to_owned(),
        parts: model
            .paths
            .iter()
            .zip(model.identity())
            .map(|(p, id)| PartFile {
                file_name: p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                identity: id.clone(),
            })
            .collect(),
    };
    let mut plan = Plan {
        schema_version: SCHEMA_VERSION,
        target: *target,
        site: target.site(),
        model: model_id,
        regions,
        entries,
        totals,
        notes,
        table_hash: String::new(),
    };
    plan.table_hash = plan.compute_hash();
    plan.check(model).map_err(PlanError::Invariant)?;
    Ok(plan)
}

fn max_buffer_bytes_of(target: &Target) -> Option<u64> {
    match target {
        Target::Apple { max_buffer_bytes, .. } => Some(*max_buffer_bytes),
        Target::Cpu { .. } | Target::Cuda { .. } => None,
    }
}

fn kind_name(k: RegionKind) -> &'static str {
    match k {
        RegionKind::Mmap => "mmap",
        RegionKind::Host => "host",
        RegionKind::Vram => "vram",
        RegionKind::Unified => "unified",
    }
}

fn next_id(regions: &[Region]) -> Result<u16, PlanError> {
    u16::try_from(regions.len()).map_err(|_| PlanError::Invariant("more than 65535 regions".into()))
}

fn weight_entry(t: &crate::load::TensorRef<'_>, bytes: u64, region: u16, offset: u64, align: u64) -> AddressEntry {
    AddressEntry {
        tensor: t.info.name.clone(),
        role: Role::Weight,
        region,
        offset,
        bytes,
        align,
        ggml_type: t.info.kind,
        dims: t.info.dims.clone(),
        source: Source::File { part: t.part, file_offset: t.file_offset },
        live: None,
    }
}

impl Plan {
    /// The plan as JSON, the form the compiler reads.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a plan always serializes")
    }

    fn compute_hash(&self) -> String {
        let mut copy = self.clone();
        copy.table_hash = String::new();
        let bytes = serde_json::to_vec(&copy).expect("a plan always serializes");
        Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// One entry by tensor name.
    pub fn entry(&self, tensor: &str) -> Option<&AddressEntry> {
        self.entries.binary_search_by(|e| e.tensor.as_str().cmp(tensor)).ok().map(|i| &self.entries[i])
    }

    /// Check the invariants in the module docs against the model the plan was made from.
    pub fn check(&self, model: &Model) -> Result<(), String> {
        for (i, r) in self.regions.iter().enumerate() {
            if usize::from(r.id) != i {
                return Err(format!("region {} is at index {i}", r.id));
            }
        }
        for e in &self.entries {
            let r = self.regions.get(usize::from(e.region)).ok_or_else(|| format!("{} names region {}", e.tensor, e.region))?;
            if e.offset % e.align != 0 {
                return Err(format!("{} at {} is not aligned to {}", e.tensor, e.offset, e.align));
            }
            if e.offset + e.bytes > r.bytes {
                return Err(format!("{} ends at {} past region {} of {} bytes", e.tensor, e.offset + e.bytes, r.id, r.bytes));
            }
            if let (Backing::File { part, file_start }, Source::File { part: sp, file_offset }) = (r.backing, e.source) {
                if part != sp || file_start + e.offset != file_offset {
                    return Err(format!("{} is not at its file offset in its window", e.tensor));
                }
            }
        }
        let weights = self.entries.iter().filter(|e| e.role == Role::Weight).count();
        if weights != model.tensors().count() {
            return Err(format!("{weights} weight entries for {} tensors in the model", model.tensors().count()));
        }
        // Overlap: within one allocated region, two entries may share bytes only if both
        // are per-step and never alive together, or one is a view.
        let mut by_region: BTreeMap<u16, Vec<&AddressEntry>> = BTreeMap::new();
        for e in self.entries.iter().filter(|e| !matches!(e.role, Role::View { .. })) {
            by_region.entry(e.region).or_default().push(e);
        }
        for (id, mut es) in by_region {
            if matches!(self.regions[usize::from(id)].backing, Backing::File { .. }) {
                continue; // file bytes are disjoint by construction of the GGUF
            }
            es.sort_by_key(|e| e.offset);
            for (i, a) in es.iter().enumerate() {
                for b in es[i + 1..].iter().take_while(|b| b.offset < a.offset + a.bytes) {
                    let disjoint_in_time = match (a.live, b.live) {
                        (Some((af, al)), Some((bf, bl))) => al < bf || bl < af,
                        _ => false,
                    };
                    if !disjoint_in_time && a.bytes > 0 && b.bytes > 0 {
                        return Err(format!("{} and {} overlap in region {id}", a.tensor, b.tensor));
                    }
                }
            }
        }
        if self.table_hash != self.compute_hash() {
            return Err("table_hash does not match the table".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load::preflight::tests::header_with;

    /// A model with `n` F32 weights of `elems` elements each, in one file.
    fn model(n: usize, elems: u64) -> Model {
        Model::from_headers(&[("m.gguf".into(), header_bytes(n, elems, &[]))]).unwrap()
    }

    fn header_bytes(n: usize, elems: u64, extra: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b = header_with("llama", None, extra);
        b[8..16].copy_from_slice(&(n as u64).to_le_bytes());
        let size = align_up(elems * 4, 32);
        for i in 0..n {
            let name = format!("w{i:03}");
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes());
            b.extend_from_slice(&elems.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b.extend_from_slice(&(i as u64 * size).to_le_bytes());
        }
        b
    }

    fn t(name: &str, role: Role, bytes: u64, first: u32, last: u32) -> PlanTensor {
        PlanTensor { name: name.into(), role, dims: vec![bytes / 4], ggml_type: 0, bytes, first, last }
    }

    fn others() -> Vec<PlanTensor> {
        vec![
            t("inp_tokens", Role::Input, 64, 0, 1),
            t("cache_k_l0", Role::Kv, 4096, 0, 0),
            t("cache_v_l0", Role::Kv, 4096, 0, 0),
            t("x0", Role::Scratch, 1024, 1, 2),
            t("x1", Role::Scratch, 1024, 3, 4),
            t("logits", Role::Output, 512, 4, 5),
            PlanTensor { name: "x0_view".into(), role: Role::View { of: "x0".into(), byte_offset: 256 }, dims: vec![64], ggml_type: 0, bytes: 256, first: 2, last: 2 },
        ]
    }

    #[test]
    fn apple_maps_weights_and_allocates_the_rest_unified() {
        let m = model(4, 1024);
        let p = place(&m, &others(), &Target::apple(1 << 30, 1 << 30)).unwrap();
        assert!(p.entries.iter().filter(|e| e.role == Role::Weight).all(|e| p.regions[usize::from(e.region)].kind == RegionKind::Mmap));
        assert!(p.entries.iter().filter(|e| e.role != Role::Weight).all(|e| p.regions[usize::from(e.region)].kind == RegionKind::Unified));
        assert_eq!(p.site, ExecSite::Metal);
    }

    #[test]
    fn apple_over_budget_does_not_fit() {
        let m = model(4, 1024);
        let err = place(&m, &others(), &Target::apple(10_000, 1 << 30)).unwrap_err();
        assert!(matches!(err, PlanError::DoesNotFit { .. }), "{err:?}");
    }

    #[test]
    fn apple_windows_split_at_the_buffer_limit() {
        // 8 weights of 64 KiB against a 160 KiB buffer limit and 16 KiB pages.
        let m = model(8, 16384);
        let p = place(&m, &[], &Target::Apple { unified_bytes: 1 << 30, max_buffer_bytes: 160 * 1024, page_bytes: 16384 }).unwrap();
        let mapped = p.regions.iter().filter(|r| r.kind == RegionKind::Mmap).count();
        assert!(mapped > 1, "{mapped} windows");
        assert!(p.regions.iter().all(|r| r.bytes <= 160 * 1024));
    }

    #[test]
    fn cpu_mmap_offsets_equal_file_offsets() {
        let m = model(4, 1024);
        let p = place(&m, &others(), &Target::cpu(1 << 30)).unwrap();
        for e in p.entries.iter().filter(|e| e.role == Role::Weight) {
            let Source::File { file_offset, .. } = e.source else { panic!("weights come from the file") };
            assert_eq!(e.offset, file_offset, "{}", e.tensor);
        }
        assert!(p.entries.iter().filter(|e| e.role != Role::Weight).all(|e| p.regions[usize::from(e.region)].kind == RegionKind::Host));
    }

    #[test]
    fn cuda_fit_all_vram() {
        let m = model(4, 1024);
        let p = place(&m, &others(), &Target::cuda(1 << 30)).unwrap();
        assert!(p.regions.iter().all(|r| r.kind == RegionKind::Vram));
        assert!(p.entries.iter().all(|e| e.offset % 256 == 0 || matches!(e.role, Role::View { .. })));
    }

    #[test]
    fn cuda_over_vram_is_deferred_error() {
        let m = model(4, 1024);
        let err = place(&m, &others(), &Target::cuda(4096)).unwrap_err();
        assert!(matches!(err, PlanError::DoesNotFitVram { .. }), "{err:?}");
        assert!(err.to_string().contains("deferred"), "{err}");
    }

    #[test]
    fn scratch_reuses_bytes_and_views_follow_their_base() {
        let m = model(1, 64);
        let p = place(&m, &others(), &Target::cpu(1 << 30)).unwrap();
        let (x0, x1) = (p.entry("x0").unwrap(), p.entry("x1").unwrap());
        assert_eq!(x0.offset, x1.offset, "never alive together, so they share bytes");
        let v = p.entry("x0_view").unwrap();
        assert_eq!((v.region, v.offset), (x0.region, x0.offset + 256));
    }

    #[test]
    fn region_ids_dense_ordered_and_one_entry_per_weight() {
        let m = model(3, 1024);
        let p = place(&m, &others(), &Target::cuda(1 << 30)).unwrap();
        for (i, r) in p.regions.iter().enumerate() {
            assert_eq!(usize::from(r.id), i);
        }
        assert_eq!(p.entries.iter().filter(|e| e.role == Role::Weight).count(), 3);
        p.check(&m).unwrap();
    }

    #[test]
    fn plan_is_deterministic_and_round_trips_json() {
        let m = model(4, 1024);
        let a = place(&m, &others(), &Target::apple(1 << 30, 1 << 30)).unwrap();
        let b = place(&m, &others(), &Target::apple(1 << 30, 1 << 30)).unwrap();
        assert_eq!(a.to_json(), b.to_json());
        // Scratch placement ignores input order; persistent tensors keep layout order,
        // which is layer order, on purpose.
        let mut scratch_reversed = others();
        scratch_reversed[3..6].reverse();
        let c = place(&m, &scratch_reversed, &Target::apple(1 << 30, 1 << 30)).unwrap();
        assert_eq!(a.entries, c.entries);
        let v: serde_json::Value = serde_json::from_str(&a.to_json()).unwrap();
        assert_eq!(v["schema_version"], 1);
        assert_eq!(v["entries"].as_array().unwrap().len(), a.entries.len());
    }

    #[test]
    fn fingerprint_changes_with_header() {
        let a = model(2, 1024);
        let b = model(2, 2048);
        let pa = place(&a, &[], &Target::cpu(1 << 30)).unwrap();
        let pb = place(&b, &[], &Target::cpu(1 << 30)).unwrap();
        assert_ne!(pa.model.parts[0].identity.header_sha256, pb.model.parts[0].identity.header_sha256);
        assert_ne!(pa.table_hash, pb.table_hash);
    }

    #[test]
    fn a_tampered_plan_fails_its_check() {
        let m = model(2, 1024);
        let mut p = place(&m, &others(), &Target::cuda(1 << 30)).unwrap();
        p.entries[0].offset += 1;
        assert!(p.check(&m).is_err());
    }

    #[test]
    fn a_view_of_nothing_is_refused() {
        let m = model(1, 64);
        let bad = [PlanTensor { name: "v".into(), role: Role::View { of: "missing".into(), byte_offset: 0 }, dims: vec![1], ggml_type: 0, bytes: 4, first: 0, last: 0 }];
        assert!(matches!(place(&m, &bad, &Target::cpu(1 << 30)).unwrap_err(), PlanError::BadView { .. }));
    }
}
