//! The planner: where every tensor of a bound layout lives on one target machine.
//!
//! In progress. The pieces that do not depend on the layout format are here:
//! [`target`] for the machine and its kinds of memory, [`arena`] for reusing scratch
//! memory between tensors that are never alive together, and [`window`] for binding a
//! mapped GGUF through GPU buffers of limited size. `plan()` and the address table
//! follow once the layout format is reviewed.

pub mod address;
pub mod arena;
pub mod target;
pub mod window;

pub use address::{place, AddressEntry, Backing, Plan, PlanError, PlanTensor, Region, Role, Source};
pub use target::{ExecSite, RegionKind, Target};
