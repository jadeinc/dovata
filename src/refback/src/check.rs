//! Author: Joe Ricks
//! File: src/refback/src/check.rs
//! Description: Checks every entry's alignment, and every weight's bytes and a matvec
//! over them, at the address the plan gives.

use std::os::unix::fs::FileExt;

use dovata::plan::{AddressEntry, Plan, Role, Source};

use crate::regions::Memory;
use crate::{quant, Check, Outcome};

/// Purpose: This function checks that every entry that owns memory sits at a multiple of
/// its alignment, as an offset and as an address.
///
/// Inputs:
/// - `memory`: the plan's regions, set up.
/// - `plan`: the plan.
///
/// Outputs: one [`Check::Align`] outcome per entry; views, which own no memory and have
/// alignment 1, are skipped.
pub fn alignment(memory: &Memory, plan: &Plan) -> Vec<Outcome> {
    let mut out = Vec::new();
    for e in plan.entries.iter().filter(|e| !matches!(e.role, Role::View { .. })) {
        let o = match memory.base(e.region) {
            None => Outcome::fail(Check::Align, &e.tensor, format!("region {} does not exist", e.region)),
            Some(_) if e.align == 0 => Outcome::fail(Check::Align, &e.tensor, "its alignment is 0"),
            Some(_) if !e.offset.is_multiple_of(e.align) => Outcome::fail(
                Check::Align,
                &e.tensor,
                format!("offset {} in region {} is not a multiple of {}", e.offset, e.region, e.align),
            ),
            Some(base) if !(base + e.offset).is_multiple_of(e.align) => Outcome::fail(
                Check::Align,
                &e.tensor,
                format!("address {:#x} is not a multiple of {}, because region {}'s base is not", base + e.offset, e.align, e.region),
            ),
            Some(base) => Outcome::pass(
                Check::Align,
                &e.tensor,
                format!("offset {} in region {}, address {:#x}, a multiple of {}", e.offset, e.region, base + e.offset, e.align),
            ),
        };
        out.push(o);
    }
    out
}

/// Purpose: This function checks every weight's bytes at its planned address against the
/// file, and runs a matvec over the weights whose type the compute check covers.
///
/// Inputs:
/// - `memory`: the plan's regions, set up.
/// - `plan`: the plan.
///
/// Outputs: one [`Check::Bytes`] outcome per weight, and one [`Check::Matvec`] outcome
/// per 2-D weight of type F32, F16, Q8_0 or Q4_0.
pub fn weights(memory: &Memory, plan: &Plan) -> Vec<Outcome> {
    let mut out = Vec::new();
    for e in plan.entries.iter().filter(|e| e.role == Role::Weight) {
        let file_bytes = match read_file(memory, e) {
            Ok(b) => b,
            Err(m) => {
                out.push(Outcome::fail(Check::Bytes, &e.tensor, m));
                continue;
            }
        };
        out.push(match memory.bytes(e.region, e.offset, e.bytes) {
            Err(m) => Outcome::fail(Check::Bytes, &e.tensor, m),
            Ok(got) => match got.iter().zip(&file_bytes).position(|(a, b)| a != b) {
                None => Outcome::pass(
                    Check::Bytes,
                    &e.tensor,
                    format!("{} bytes at region {} + {} equal the file's", e.bytes, e.region, e.offset),
                ),
                Some(i) => Outcome::fail(
                    Check::Bytes,
                    &e.tensor,
                    format!(
                        "byte {i} at region {} + {} is {:#04x}, and the file has {:#04x} there",
                        e.region,
                        e.offset,
                        got[i],
                        file_bytes[i]
                    ),
                ),
            },
        });
        if let Some((cols, rows)) = matvec_shape(e) {
            out.push(matvec(memory, e, cols, rows, &file_bytes));
        }
    }
    out
}

/// Purpose: This function reads a weight's bytes from its part file with a positioned
/// read, not through the mapping.
///
/// Inputs:
/// - `memory`: the plan's regions, which hold the open part files.
/// - `e`: the weight's entry.
///
/// Outputs: the bytes, or a message saying why they cannot be read.
fn read_file(memory: &Memory, e: &AddressEntry) -> Result<Vec<u8>, String> {
    let Source::File { part, file_offset } = e.source else {
        return Err("a weight should come from the file, and its entry has no file source".into());
    };
    let file = memory.file(part).ok_or_else(|| format!("its source is part {}, which the model does not have", part + 1))?;
    let len = usize::try_from(e.bytes).map_err(|_| format!("{} bytes do not fit in memory", e.bytes))?;
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, file_offset)
        .map_err(|err| format!("cannot read {len} bytes of part {} at {file_offset}: {err}", part + 1))?;
    Ok(buf)
}

/// Purpose: This function says whether the compute check covers a weight, and its shape.
///
/// Inputs:
/// - `e`: the weight's entry.
///
/// Outputs: `(columns, rows)` for a 2-D weight of a covered type, else `None`.
fn matvec_shape(e: &AddressEntry) -> Option<(usize, usize)> {
    if !quant::covered(e.ggml_type) || e.dims.len() < 2 || e.dims[2..].iter().any(|d| *d != 1) {
        return None;
    }
    let cols = usize::try_from(e.dims[0]).ok()?;
    let rows = usize::try_from(e.dims[1]).ok()?;
    quant::row_bytes(e.ggml_type, cols)?;
    Some((cols, rows))
}

/// Purpose: This function runs y = W·x reading W from its planned address and compares y
/// with the f64 reference over the file's bytes.
///
/// Inputs:
/// - `memory`: the plan's regions, set up.
/// - `e`: the weight's entry.
/// - `cols`: the elements in one row.
/// - `rows`: the number of rows.
/// - `file_bytes`: the weight's bytes, read from the file.
///
/// Outputs: the [`Check::Matvec`] outcome.
fn matvec(memory: &Memory, e: &AddressEntry, cols: usize, rows: usize, file_bytes: &[u8]) -> Outcome {
    let ty = dovata::gguf::ggml_type_name(e.ggml_type);
    let rb = quant::row_bytes(e.ggml_type, cols).unwrap_or(0);
    if file_bytes.len() < rows * rb {
        return Outcome::fail(Check::Matvec, &e.tensor, format!("{} bytes are too few for {rows} rows of {cols} {ty} values", file_bytes.len()));
    }
    let x = quant::input(cols);
    let want = quant::reference(e.ggml_type, file_bytes, cols, rows, &x);
    let (got, on) = match kernel(memory, e, cols, rows, &x) {
        Ok(y) => y,
        Err(m) => return Outcome::fail(Check::Matvec, &e.tensor, m),
    };
    match quant::compare(&got, &want) {
        Ok(worst) => Outcome::pass(
            Check::Matvec,
            &e.tensor,
            format!("{ty} [{cols} x {rows}] {on} matches the f64 reference, the largest error {worst:.1e} of its bound"),
        ),
        Err(m) => Outcome::fail(Check::Matvec, &e.tensor, format!("{ty} {on}: {m}")),
    }
}

/// Purpose: This function runs the matvec kernel for the plan's target over W at its
/// planned address.
///
/// Inputs:
/// - `memory`: the plan's regions, set up.
/// - `e`: the weight's entry.
/// - `cols`: the elements in one row.
/// - `rows`: the number of rows.
/// - `x`: the input vector.
///
/// Outputs: y and where it ran, or a message saying why it could not run.
fn kernel(memory: &Memory, e: &AddressEntry, cols: usize, rows: usize, x: &[f32]) -> Result<(Vec<f32>, &'static str), String> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Some(gpu) = memory.gpu() {
        let buffer = memory.buffer(e.region).ok_or_else(|| format!("region {} has no Metal buffer", e.region))?;
        return gpu.matvec(buffer, e.offset, e.ggml_type, cols, rows, x).map(|y| (y, "on the GPU"));
    }
    let rb = quant::row_bytes(e.ggml_type, cols).unwrap_or(0) as u64;
    let w = memory.bytes(e.region, e.offset, rows as u64 * rb)?;
    Ok((quant::matvec(e.ggml_type, w, cols, rows, x), "on the CPU"))
}
