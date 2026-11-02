# arena: sharing memory between short-lived tensors

`arena` packs the short-lived tensors of one forward step into one shared memory region.
Two tensors can use the same bytes as long as they are never alive at the same time.
`assign` takes each tensor's size, alignment and lifetime, which is the first and last op
that touches it, and returns each tensor's offset and the region's total size. `check`
confirms that the result is aligned and that no two live tensors share a byte.

```rust
let a = Request { name: "a".into(), bytes: 4096, align: 256, first: 0, last: 1 };
let b = Request { name: "b".into(), bytes: 4096, align: 256, first: 2, last: 3 };
let p = arena::assign(&[a, b]);
// a and b are never alive together, so both get offset 0, and p.size == 4096
```

## Why it exists

The arena keeps scratch memory small. One forward step of a 48-layer model makes
thousands of intermediate tensors, and almost all of them die a few ops after they are
made. Giving each its own bytes would need gigabytes of scratch. Reusing memory between
tensors whose lifetimes do not overlap brings that down to roughly the largest few that
are alive at once.

## How it packs

The arena uses greedy by size, the method TensorFlow Lite's arena planner uses. It
places the largest tensor first, and each one at the lowest aligned offset that does not
overlap a tensor alive at the same time. It is not optimal, since the problem is
NP-hard, which means no known method solves it exactly in reasonable time, but it is
close in practice.

It is also deterministic. The compiler bakes the offsets into the program, so the same
tensors must always give the same offsets, or a rebuild would silently change the
binary. Ties break by first use and then by name, so the result never depends on input
order or hashing.

## Why it is its own directory

It is a pure function of sizes and lifetimes. It knows nothing about models, layouts or
targets, and its tests check it alone. The planner decides which tensors are
short-lived and what their lifetimes are; the arena only decides where they go.

## Files

- `mod.rs`: `Request`, `Placement`, `assign` and `check`, with tests that tensors alive
  at different times share bytes, that tensors alive together do not, and that input
  order does not change the result.

## Who uses it

`memplan::place` calls it once per plan, for every input, scratch and output tensor, and
makes one region from the result. It runs `check` on the result before using it.
