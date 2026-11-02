//! Author: Joe Ricks
//! File: src/refback/src/identity.rs
//! Description: Refuses model files other than the ones a plan was made for, by each part
//! file's size and header hash.

use std::path::PathBuf;

use dovata::load::Model;
use dovata::memplan::Plan;

use crate::{Check, Outcome, StartError};

/// Purpose: This function checks that every part file is the one the plan was made for.
///
/// Inputs:
/// - `parts`: the model's part files, part 1 first.
/// - `plan`: the plan.
///
/// Outputs: one passing [`Outcome`] per part, or [`StartError::Identity`] naming the first
/// part whose size or header hash differs from the plan's.
pub fn check(
    parts: &[PathBuf],
    plan: &Plan,
) -> Result<Vec<Outcome>, StartError>
{
    if parts.len() != plan.model.parts.len() {
        return Err(StartError::Identity(format!("the plan is for {} part files and {} were given", plan.model.parts.len(), parts.len())));
    }
    let model = Model::from_parts(parts).map_err(|e| StartError::Identity(format!("{}: {e}", parts[0].display())))?;
    let mut out = Vec::new();
    for (i, ((path, found), want)) in parts.iter().zip(model.identity()).zip(&plan.model.parts).enumerate() {
        let name = path.display().to_string();
        let on_disk = std::fs::metadata(path).map_err(|e| StartError::Io(format!("{name}: {e}")))?.len();
        if found.header_sha256 != want.identity.header_sha256 {
            return Err(StartError::Identity(format!(
                "part {} ({name}) has header sha256 {}, and the plan was made for {}",
                i + 1,
                found.header_sha256,
                want.identity.header_sha256
            )));
        }
        if on_disk != want.identity.file_bytes {
            return Err(StartError::Identity(format!(
                "part {} ({name}) is {on_disk} bytes, and the plan was made for a file of {}",
                i + 1,
                want.identity.file_bytes
            )));
        }
        out.push(Outcome::pass(
            Check::Identity,
            format!("part {}", i + 1),
            format!("{name} is {on_disk} bytes with header sha256 {}, as the plan says", found.header_sha256),
        ));
    }
    Ok(out)
}
