//! Author: Joe Ricks
//! File: src/refback/src/sharing.rs
//! Description: Walks a plan's ops in run order with a byte pattern unique to each
//! tensor, to prove that tensors sharing memory are never alive together.

use std::collections::HashMap;

use dovata::gguf::ggml_type_layout;
use dovata::plan::{AddressEntry, Plan, Role};

use crate::regions::Memory;
use crate::{Check, Outcome};

/// Purpose: This function walks a plan's ops and checks every read sees its own tensor's
/// pattern.
///
/// Inputs:
/// - `memory`: the plan's regions, set up. The allocated regions are written.
/// - `plan`: the plan, with its step.
///
/// Outputs: one failing [`Check::Sharing`] outcome per tensor found overwritten or
/// misplaced, or one passing outcome for the whole walk.
pub fn walk(memory: &mut Memory, plan: &Plan) -> Vec<Outcome> {
    let Some(step) = &plan.step else {
        return vec![Outcome::fail(Check::Sharing, "the step", "the plan has no ops, so there is nothing to walk")];
    };
    let mut w = Walk {
        entries: &plan.entries,
        index: plan.entries.iter().enumerate().map(|(i, e)| (e.tensor.as_str(), i)).collect(),
        filled: vec![false; plan.entries.len()],
        reported: vec![false; plan.entries.len()],
        out: Vec::new(),
        reads: 0,
    };

    // Persistent tensors are written once; inputs before the step starts.
    for i in 0..plan.entries.len() {
        if matches!(plan.entries[i].role, Role::Kv | Role::State | Role::Input) {
            w.fill(memory, i, "before the step");
        }
    }
    for (n, op) in step.ops.iter().enumerate() {
        let at = format!("op {n} ({})", op.op);
        // The op's first touch of a scratch or output tensor gives it its pattern. A
        // view owns no memory, so the tensor it views is the one filled.
        for name in op.inputs.iter().chain(&op.outputs) {
            match w.owner(name) {
                Ok(Some(root)) if matches!(w.entries[root].role, Role::Scratch | Role::Output) && !w.filled[root] => w.fill(memory, root, &at),
                Ok(_) => {}
                Err(m) => w.fail_once(None, name, format!("{at}: {m}")),
            }
        }
        // Then every input must still be its owner's bytes, outputs written included, so
        // an output placed over one of the op's own inputs is caught here.
        for name in &op.inputs {
            w.read(memory, name, &at);
        }
    }
    // The program reads the outputs after the step.
    for name in &step.outputs {
        w.read(memory, name, "after the step");
    }

    if w.out.is_empty() {
        let tensors = w.filled.iter().filter(|f| **f).count();
        w.out.push(Outcome::pass(
            Check::Sharing,
            "the step",
            format!("{} ops walked; {} reads of {tensors} tensors each found their own bytes", step.ops.len(), w.reads),
        ));
    }
    w.out
}

/// Purpose: This struct holds the state of one walk over a plan's ops.
struct Walk<'a> {
    /// The plan's entries.
    entries: &'a [AddressEntry],
    /// Each entry's index by tensor name.
    index: HashMap<&'a str, usize>,
    /// Whether each entry has been given its pattern.
    filled: Vec<bool>,
    /// Whether a failure has been reported for each entry, so each is reported once.
    reported: Vec<bool>,
    /// The failures found so far.
    out: Vec<Outcome>,
    /// How many reads were checked.
    reads: u64,
}

impl Walk<'_> {
    /// Purpose: This function finds the entry that owns a tensor's memory.
    ///
    /// Inputs:
    /// - `name`: the tensor's name.
    ///
    /// Outputs: the owner's index, `None` for a weight or a view of one, which lives in
    /// the model file and is covered by the weight check, or a message when the name or
    /// the tensor it views has no entry.
    fn owner(&self, name: &str) -> Result<Option<usize>, String> {
        let i = *self.index.get(name).ok_or_else(|| format!("it names {name}, which has no entry"))?;
        let root = match &self.entries[i].role {
            Role::View { of, .. } => {
                let r = *self.index.get(of.as_str()).ok_or_else(|| format!("{name} views {of}, which has no entry"))?;
                if matches!(self.entries[r].role, Role::View { .. }) {
                    return Err(format!("{name} views {of}, which is itself a view"));
                }
                r
            }
            _ => i,
        };
        Ok((self.entries[root].role != Role::Weight).then_some(root))
    }

    /// Purpose: This function records a failure, once per tensor.
    ///
    /// Inputs:
    /// - `entry`: the entry the failure is about, when it has one.
    /// - `subject`: the tensor's name.
    /// - `detail`: what was wrong.
    ///
    /// Outputs: none.
    fn fail_once(&mut self, entry: Option<usize>, subject: &str, detail: String) {
        if let Some(i) = entry {
            if std::mem::replace(&mut self.reported[i], true) {
                return;
            }
        }
        self.out.push(Outcome::fail(Check::Sharing, subject, detail));
    }

    /// Purpose: This function writes a tensor's pattern over all of its bytes.
    ///
    /// Inputs:
    /// - `memory`: the plan's regions.
    /// - `i`: the entry's index; it owns its memory.
    /// - `at`: when it is written, for a message.
    ///
    /// Outputs: none; a range outside its region is recorded as a failure.
    fn fill(&mut self, memory: &mut Memory, i: usize, at: &str) {
        let e = &self.entries[i];
        self.filled[i] = true;
        match memory.bytes_mut(e.region, e.offset, e.bytes) {
            Ok(buf) => fill(buf, seed(&e.tensor), 0),
            Err(m) => self.fail_once(Some(i), &e.tensor, format!("written {at}: {m}")),
        }
    }

    /// Purpose: This function checks one read of a tensor against its owner's pattern.
    ///
    /// Inputs:
    /// - `memory`: the plan's regions.
    /// - `name`: the tensor read, a view or a tensor that owns its memory.
    /// - `at`: which op reads it, for a message.
    ///
    /// Outputs: none; a broken pattern is recorded as a failure.
    fn read(&mut self, memory: &Memory, name: &str, at: &str) {
        let root = match self.owner(name) {
            Ok(Some(r)) => r,
            Ok(None) => return,
            Err(m) => return self.fail_once(None, name, format!("{at}: {m}")),
        };
        let e = &self.entries[self.index[name]];
        let owner = &self.entries[root];
        if !self.filled[root] {
            return self.fail_once(Some(root), name, format!("read by {at} before anything wrote {}", owner.tensor));
        }
        self.reads += 1;
        // Where the read's bytes sit in the owner: all of it, or a view's spans.
        let (start, spans) = match &e.role {
            Role::View { byte_offset, nb, .. } => match spans(e, *nb) {
                Ok(s) => (*byte_offset, s),
                Err(m) => return self.fail_once(Some(root), name, format!("read by {at}: {m}")),
            },
            _ => (0, vec![(0, e.bytes)]),
        };
        let s = seed(&owner.tensor);
        for (rel, len) in spans {
            if start + rel + len > owner.bytes {
                let m = format!("read by {at}: its bytes {}..{} run past {}, which is {} bytes", start + rel, start + rel + len, owner.tensor, owner.bytes);
                return self.fail_once(Some(root), name, m);
            }
            // The read goes through the tensor's own entry, as the program's kernel does.
            let got = match memory.bytes(e.region, e.offset + rel, len) {
                Ok(b) => b,
                Err(m) => return self.fail_once(Some(root), name, format!("read by {at}: {m}")),
            };
            if let Some(k) = mismatch(got, s, start + rel) {
                let pos = start + rel + k as u64;
                let m = format!(
                    "read by {at}, byte {pos} of {} at region {} + {} is {:#04x} where its pattern has {:#04x}: \
                     another tensor's bytes were placed there while {} was alive",
                    owner.tensor,
                    e.region,
                    e.offset + rel + k as u64,
                    got[k],
                    byte(s, pos),
                    owner.tensor
                );
                return self.fail_once(Some(root), name, m);
            }
        }
    }
}

/// Purpose: This function lists the byte spans a view covers, relative to its start.
///
/// Inputs:
/// - `e`: the view's entry, with its dims and type.
/// - `nb`: its byte strides, ggml's `nb`.
///
/// Outputs: each `(offset, length)` span, adjacent spans merged, or a message for a type
/// whose size is unknown or a shape that is not whole blocks.
fn spans(e: &AddressEntry, nb: [u64; 4]) -> Result<Vec<(u64, u64)>, String> {
    let (blck, tsize) = ggml_type_layout(e.ggml_type).ok_or_else(|| format!("ggml_type {} has no known size", e.ggml_type))?;
    let mut ne = [1u64; 4];
    for (n, d) in ne.iter_mut().zip(&e.dims) {
        *n = *d;
    }
    if ne[0] % blck != 0 {
        return Err(format!("{} elements are not whole blocks of {blck}", ne[0]));
    }
    let units = ne[0] / blck;
    // A row is one span when its blocks are packed, else one span per block.
    let (per_row, len) = if nb[0] == tsize { (1, units * tsize) } else { (units, tsize) };
    let mut out: Vec<(u64, u64)> = Vec::new();
    for i3 in 0..ne[3] {
        for i2 in 0..ne[2] {
            for i1 in 0..ne[1] {
                for u in 0..per_row {
                    let at = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + u * nb[0];
                    match out.last_mut() {
                        Some((s, l)) if *s + *l == at => *l += len,
                        _ => out.push((at, len)),
                    }
                }
            }
        }
    }
    Ok(out)
}

/// Purpose: This function turns a tensor's name into the seed of its pattern.
///
/// Inputs:
/// - `name`: the tensor's name.
///
/// Outputs: a 64-bit FNV-1a hash of the name.
fn seed(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Purpose: This function computes one 8-byte word of a tensor's pattern.
///
/// Inputs:
/// - `seed`: the tensor's seed.
/// - `word`: the word's index, its byte position divided by 8.
///
/// Outputs: the word, from the splitmix64 finalizer, so neighbouring words and
/// neighbouring tensors share no structure.
fn word(seed: u64, word: u64) -> u64 {
    let mut z = (seed ^ word.wrapping_mul(0x9e37_79b9_7f4a_7c15)).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Purpose: This function returns one byte of a tensor's pattern.
///
/// Inputs:
/// - `seed`: the tensor's seed.
/// - `pos`: the byte's position in the tensor.
///
/// Outputs: the byte.
fn byte(seed: u64, pos: u64) -> u8 {
    word(seed, pos / 8).to_le_bytes()[(pos % 8) as usize]
}

/// Purpose: This function writes a tensor's pattern into a buffer.
///
/// Inputs:
/// - `buf`: the bytes to write.
/// - `seed`: the tensor's seed.
/// - `start`: the position in the tensor of `buf`'s first byte.
///
/// Outputs: none.
fn fill(buf: &mut [u8], seed: u64, start: u64) {
    for (k, b) in buf.iter_mut().enumerate() {
        *b = byte(seed, start + k as u64);
    }
}

/// Purpose: This function finds the first byte of a buffer that differs from a tensor's
/// pattern.
///
/// Inputs:
/// - `got`: the bytes read.
/// - `seed`: the tensor's seed.
/// - `start`: the position in the tensor of `got`'s first byte.
///
/// Outputs: the index of the first differing byte, or `None` when all match.
fn mismatch(got: &[u8], seed: u64, start: u64) -> Option<usize> {
    let mut k = 0usize;
    while k < got.len() {
        let pos = start + k as u64;
        let w = word(seed, pos / 8).to_le_bytes();
        let from = (pos % 8) as usize;
        let n = (8 - from).min(got.len() - k);
        if got[k..k + n] != w[from..from + n] {
            return (k..k + n).find(|&j| got[j] != w[from + j - k]);
        }
        k += n;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pattern_checks_from_any_start_and_differs_between_tensors() {
        let mut buf = vec![0u8; 100];
        fill(&mut buf, seed("a"), 0);
        assert_eq!(mismatch(&buf, seed("a"), 0), None);
        assert_eq!(mismatch(&buf[13..57], seed("a"), 13), None, "a view checks a span in the middle");
        assert_eq!(mismatch(&buf[13..57], seed("a"), 14), Some(0), "shifted by one byte, it differs at once");
        assert!(mismatch(&buf, seed("b"), 0).is_some());
        buf[70] ^= 1;
        assert_eq!(mismatch(&buf, seed("a"), 0), Some(70));
    }

    #[test]
    fn a_permuted_view_is_one_span_per_row_and_a_packed_one_merges() {
        let mut e = AddressEntry {
            tensor: "v".into(),
            role: Role::View { of: "t".into(), byte_offset: 0, nb: [4, 16, 64, 64] },
            region: 0,
            offset: 0,
            bytes: 0,
            align: 1,
            ggml_type: 0,
            dims: vec![4, 4],
            source: dovata::plan::Source::None,
            live: None,
        };
        assert_eq!(spans(&e, [4, 16, 64, 64]).unwrap(), vec![(0, 64)]);
        e.dims = vec![2, 3];
        assert_eq!(spans(&e, [4, 32, 96, 96]).unwrap(), vec![(0, 8), (32, 8), (64, 8)]);
        assert_eq!(spans(&e, [16, 4, 64, 64]).unwrap(), vec![(0, 4), (16, 4), (4, 4), (20, 4), (8, 4), (24, 4)], "a transposed view");
    }
}
