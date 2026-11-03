//! Author: Joe Ricks
//! File: src/memplan/mod.rs
//! Description: Declares the planner's parts and re-exports its main types.

pub mod address;
pub mod graph;

pub use address::{place, AddressEntry, Backing, Plan, PlanError, PlanTensor, Region, Role, Source};
pub use graph::{plan, plan_graph, Step, StepOp};
