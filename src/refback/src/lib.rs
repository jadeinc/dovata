//! Author: Joe Ricks
//! File: src/refback/src/lib.rs
//! Description: A test-only backend that sets up a dovata plan's memory for real and
//! checks that every tensor is where the plan says.

use std::fmt;
use std::path::PathBuf;

use dovata::memplan::Plan;

pub mod check;
pub mod fixture;
pub mod identity;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;
pub mod quant;
pub mod regions;
pub mod sharing;

#[cfg(test)]
mod tests;

/// Purpose: This enum lists the kinds of check refback runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Check {
    /// A part file's size and header hash match the plan.
    Identity,
    /// A region was set up and its base meets the region's alignment.
    Region,
    /// An entry's offset, and its address, are multiples of its alignment.
    Align,
    /// A weight's bytes at its planned address equal the file's bytes.
    Bytes,
    /// A matvec reading a weight from its planned address matches an f64 reference.
    Matvec,
    /// Walking the step's ops, every read finds the bytes its own tensor was given.
    Sharing,
}

impl Check {
    /// Purpose: This function returns the check's short name for a report line.
    ///
    /// Inputs: none.
    ///
    /// Outputs: a name such as `bytes`.
    pub fn name(self) -> &'static str
    {
        match self {
            Check::Identity => "identity",
            Check::Region => "region",
            Check::Align => "align",
            Check::Bytes => "bytes",
            Check::Matvec => "matvec",
            Check::Sharing => "sharing",
        }
    }
}

/// Purpose: This struct holds the result of one check on one subject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Which check it is.
    pub check: Check,
    /// What was checked: a part file, a region, or a tensor's name.
    pub subject: String,
    /// Whether it passed.
    pub pass: bool,
    /// What was found, in one sentence.
    pub detail: String,
}

impl Outcome {
    /// Purpose: This function builds a passing outcome.
    ///
    /// Inputs:
    /// - `check`: which check it is.
    /// - `subject`: what was checked.
    /// - `detail`: what was found.
    ///
    /// Outputs: the [`Outcome`].
    pub fn pass(
        check: Check,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) -> Outcome
    {
        Outcome { check, subject: subject.into(), pass: true, detail: detail.into() }
    }

    /// Purpose: This function builds a failing outcome.
    ///
    /// Inputs:
    /// - `check`: which check it is.
    /// - `subject`: what was checked.
    /// - `detail`: what was wrong.
    ///
    /// Outputs: the [`Outcome`].
    pub fn fail(
        check: Check,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) -> Outcome
    {
        Outcome { check, subject: subject.into(), pass: false, detail: detail.into() }
    }
}

impl fmt::Display for Outcome {
    /// Purpose: This function formats an outcome as one report line.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        let status = if self.pass { "pass" } else { "FAIL" };
        write!(f, "{status}  {:<8}  {}: {}", self.check.name(), self.subject, self.detail)
    }
}

/// Purpose: This struct holds every outcome of one refback run, in the order the checks
/// ran.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Report {
    /// One per check and subject.
    pub outcomes: Vec<Outcome>,
}

impl Report {
    /// Purpose: This function checks whether every outcome passed.
    ///
    /// Inputs: none.
    ///
    /// Outputs: `true` when nothing failed.
    pub fn passed(&self) -> bool
    {
        self.outcomes.iter().all(|o| o.pass)
    }

    /// Purpose: This function lists the failing outcomes.
    ///
    /// Inputs: none.
    ///
    /// Outputs: each failure, in run order.
    pub fn failures(&self) -> Vec<&Outcome>
    {
        self.outcomes.iter().filter(|o| !o.pass).collect()
    }

    /// Purpose: This function lists the subjects one kind of check failed on.
    ///
    /// Inputs:
    /// - `check`: the kind of check.
    ///
    /// Outputs: the subjects, sorted, each once.
    pub fn failed(
        &self,
        check: Check,
    ) -> Vec<&str>
    {
        let mut s: Vec<&str> = self.outcomes.iter().filter(|o| o.check == check && !o.pass).map(|o| o.subject.as_str()).collect();
        s.sort_unstable();
        s.dedup();
        s
    }
}

/// Purpose: This enum lists the reasons refback refuses to start.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StartError {
    /// A part file is not the one the plan was made for, or the parts do not match.
    Identity(String),
    /// The plan is for a target refback does not cover, or one it was built without.
    Unsupported(String),
    /// A region cannot be set up as the plan describes it.
    Region(String),
    /// A part file could not be opened, read or mapped.
    Io(String),
}

impl fmt::Display for StartError {
    /// Purpose: This function formats the reason refback refuses to start.
    ///
    /// Inputs:
    /// - `f`: the formatter Rust writes the text into.
    ///
    /// Outputs: `Ok(())`, or an error if writing fails.
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result
    {
        match self {
            StartError::Identity(m) => write!(f, "refusing the model: {m}"),
            StartError::Unsupported(m) => write!(f, "{m}"),
            StartError::Region(m) => write!(f, "cannot set up the plan's regions: {m}"),
            StartError::Io(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StartError {}

/// Purpose: This function checks a plan against its model files on real memory.
///
/// Inputs:
/// - `parts`: the model's part files, part 1 first.
/// - `plan`: the plan, as dovata made it or as a test changed it.
///
/// Outputs: the [`Report`] of every check, or a [`StartError`] when the files are not the
/// ones the plan was made for or its regions cannot be set up.
pub fn run(
    parts: &[PathBuf],
    plan: &Plan,
) -> Result<Report, StartError>
{
    let mut outcomes = identity::check(parts, plan)?;
    let (mut memory, set_up) = regions::Memory::set_up(parts, plan)?;
    outcomes.extend(set_up);
    outcomes.extend(check::alignment(&memory, plan));
    outcomes.extend(check::weights(&memory, plan));
    outcomes.extend(sharing::walk(&mut memory, plan));
    Ok(Report { outcomes })
}
