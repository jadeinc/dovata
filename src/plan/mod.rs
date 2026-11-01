//! Author: Joe Ricks
//! File: src/plan/mod.rs
//! Description: Declares the planner's parts and re-exports its main types.

pub mod address;
pub mod arena;
pub mod graph;
pub mod target;
pub mod window;

pub use address::{place, AddressEntry, Backing, Plan, PlanError, PlanTensor, Region, Role, Source};
pub use graph::{plan, plan_graph, Step, StepOp};
pub use target::{ExecSite, RegionKind, Target};
