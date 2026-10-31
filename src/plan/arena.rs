//! Placing short-lived tensors in one region so that ones never alive at the same time
//! share bytes.
//!
//! ## Why it matters
//!
//! One forward step of a 48-layer model makes thousands of intermediate tensors, and
//! almost all of them die a few ops after they are made. Giving each its own bytes would
//! need gigabytes of scratch. Reusing memory between tensors whose lifetimes do not
//! overlap brings that down to roughly the largest few that are alive at once.
//!
//! ## How
//!
//! Greedy by size, the method TensorFlow Lite's arena planner uses: largest request
//! first, each at the lowest aligned offset that does not overlap any already-placed
//! request alive at the same time. It is not optimal; the problem is NP-hard, and greedy
//! by size is close in practice.
//!
//! ## Why determinism is a requirement
//!
//! The compiler bakes these offsets into the program as constants. If the same model
//! and layout could produce different offsets on two runs, a rebuild would silently
//! change the binary. Ties are broken by first use and then by name, so the result
//! depends only on the requests, never on their order or on hashing.

/// One tensor the arena must place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The tensor's name. Used only to break ties.
    pub name: String,
    /// Bytes it needs.
    pub bytes: u64,
    /// The alignment its offset needs. A power of two.
    pub align: u64,
    /// The index of the first op that touches it.
    pub first: u32,
    /// The index of the last op that touches it. Inclusive.
    pub last: u32,
}

impl Request {
    fn overlaps_in_time(&self, other: &Request) -> bool {
        self.first <= other.last && other.first <= self.last
    }
}

/// Where each request landed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// `offsets[i]` is the offset of `requests[i]`.
    pub offsets: Vec<u64>,
    /// The region size needed: the end of the highest request.
    pub size: u64,
}

fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

/// Place every request. Same requests in any order give the same offsets.
pub fn assign(requests: &[Request]) -> Placement {
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (&requests[a], &requests[b]);
        rb.bytes.cmp(&ra.bytes).then(ra.first.cmp(&rb.first)).then(ra.name.cmp(&rb.name))
    });

    let mut offsets = vec![0u64; requests.len()];
    let mut placed: Vec<usize> = Vec::with_capacity(requests.len());
    let mut size = 0u64;

    for &i in &order {
        let r = &requests[i];
        if r.bytes == 0 {
            continue;
        }
        // Everything already placed that is alive at the same time, lowest offset first.
        let mut conflicts: Vec<usize> = placed.iter().copied().filter(|&j| requests[j].overlaps_in_time(r)).collect();
        conflicts.sort_by_key(|&j| (offsets[j], j));

        // Sweep upward. Once the candidate has passed a conflict's start without fitting
        // below it, it has to clear that conflict's end.
        let mut at = 0u64;
        for &j in &conflicts {
            if at + r.bytes <= offsets[j] {
                break;
            }
            at = at.max(align_up(offsets[j] + requests[j].bytes, r.align));
        }
        offsets[i] = at;
        size = size.max(at + r.bytes);
        placed.push(i);
    }

    Placement { offsets, size }
}

/// Check a placement: every offset aligned, and no two requests alive at the same time
/// share a byte. The planner calls this on its own output, because a violation here
/// means two tensors overwrite each other on the GPU, which no later check would catch.
pub fn check(requests: &[Request], p: &Placement) -> Result<(), String> {
    for (i, r) in requests.iter().enumerate() {
        if p.offsets[i] % r.align != 0 {
            return Err(format!("{} at offset {} is not aligned to {}", r.name, p.offsets[i], r.align));
        }
        if p.offsets[i] + r.bytes > p.size {
            return Err(format!("{} ends past the arena", r.name));
        }
    }
    for i in 0..requests.len() {
        for j in i + 1..requests.len() {
            let (a, b) = (&requests[i], &requests[j]);
            if a.bytes == 0 || b.bytes == 0 || !a.overlaps_in_time(b) {
                continue;
            }
            let (ao, bo) = (p.offsets[i], p.offsets[j]);
            if ao < bo + b.bytes && bo < ao + a.bytes {
                return Err(format!("{} and {} are alive together and overlap in memory", a.name, b.name));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(name: &str, bytes: u64, first: u32, last: u32) -> Request {
        Request { name: name.into(), bytes, align: 256, first, last }
    }

    #[test]
    fn disjoint_lifetimes_share_offset() {
        let rs = [req("a", 1000, 0, 1), req("b", 1000, 2, 3)];
        let p = assign(&rs);
        assert_eq!(p.offsets[0], p.offsets[1]);
        assert_eq!(p.size, 1000);
        check(&rs, &p).unwrap();
    }

    #[test]
    fn overlapping_lifetimes_do_not() {
        let rs = [req("a", 1000, 0, 2), req("b", 1000, 2, 3)];
        let p = assign(&rs);
        assert_ne!(p.offsets[0], p.offsets[1]);
        assert_eq!(p.size, 1024 + 1000, "the second starts at the next 256-byte boundary");
        check(&rs, &p).unwrap();
    }

    #[test]
    fn a_small_tensor_fits_in_a_gap_below_a_live_one() {
        // c is alive with b but not with a, so it can take a's bytes below b.
        let rs = [req("a", 4096, 0, 1), req("b", 4096, 0, 5), req("c", 1024, 2, 5)];
        let p = assign(&rs);
        check(&rs, &p).unwrap();
        assert_eq!(p.size, 8192, "c reuses a's bytes instead of growing the arena");
    }

    #[test]
    fn offsets_honor_each_alignment() {
        let mut rs = vec![req("a", 100, 0, 9)];
        rs.push(Request { name: "b".into(), bytes: 100, align: 4096, first: 0, last: 9 });
        let p = assign(&rs);
        assert_eq!(p.offsets[1] % 4096, 0);
        check(&rs, &p).unwrap();
    }

    #[test]
    fn input_order_does_not_change_the_result() {
        let rs: Vec<Request> = (0..40)
            .map(|i: u32| req(&format!("t{i:02}"), 512 * u64::from(1 + (i * 7 % 5)), i % 9, i % 9 + (i % 4)))
            .collect();
        let forward = assign(&rs);
        check(&rs, &forward).unwrap();

        let mut reversed = rs.clone();
        reversed.reverse();
        let back = assign(&reversed);
        for (i, r) in rs.iter().enumerate() {
            let j = reversed.iter().position(|x| x.name == r.name).unwrap();
            assert_eq!(forward.offsets[i], back.offsets[j], "{}", r.name);
        }
        assert_eq!(forward.size, back.size);
    }

    #[test]
    fn check_catches_an_overlap() {
        let rs = [req("a", 1000, 0, 2), req("b", 1000, 1, 3)];
        let bad = Placement { offsets: vec![0, 512], size: 1512 };
        assert!(check(&rs, &bad).unwrap_err().contains("overlap"));
    }
}
