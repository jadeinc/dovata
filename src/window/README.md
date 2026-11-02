# window: fitting a mapped model file into GPU buffers

`window` splits one mapped GGUF file into windows: page-aligned ranges of the file, each
no larger than one GPU buffer, so that every tensor lies wholly inside one window.
`windows` takes each tensor's file offset and size, the buffer limit and the page size.
It returns the windows and the window each tensor is in.

```rust
let spans = [Span { file_offset: 0, bytes: 6 << 20 }, Span { file_offset: 6 << 20, bytes: 6 << 20 }];
let (wins, which) = window::windows(&spans, 8 << 20, 16384)?;
// two windows, one per tensor: which == [0, 1]
```

## Why it exists

On Apple and on the CPU, the weights are never copied. The program maps each part file
once and reads every weight where it sits in the file. On Apple, Metal reads the mapping
through no-copy buffers, made with `newBufferWithBytesNoCopy`, the way ggml's Metal
backend does in `ggml-metal-device.m`. A Metal buffer has a size limit,
`maxBufferLength`, which can be smaller than the model, so the mapping is bound as
several windows. Every tensor must lie wholly inside one, since a kernel addresses a
tensor as one buffer plus an offset.

On the CPU there is no buffer limit, so the planner passes `u64::MAX` and gets one
window per file, starting at byte 0, so every offset is the file offset.

## How it packs

ggml's Metal backend, in `ggml_backend_metal_buffer_from_ptr` in `ggml-metal-device.m`,
makes its windows as fixed-stride views that overlap by the largest tensor. `windows`
packs greedily instead. A window grows tensor by tensor until the next one would not
fit, and the next window starts at the page holding that tensor. Every tensor gets
exactly one window, and windows overlap by at most a page.

Each window starts on a page boundary, so the program can hand `base + file_start` to
Metal without copying. Metal also wants the length in whole pages, so the program rounds
each window up to the end of its last page. `windows` keeps every window within the
buffer limit rounded down to whole pages, so the rounded length still fits.

A tensor that is larger than the limit by itself is refused with
`Error::TensorTooLarge`.

## Why it is its own directory

It is a pure function of offsets and sizes. It knows nothing about models or plans, and
its tests check it alone. Planning a model is its one user, and keeping it out of
`memplan/` leaves the planner with the job of deciding where things go, not how a file
fits in a buffer.

## Files

- `mod.rs`: `Span`, `Window`, `Error` and `windows`, with tests that every tensor fits
  one window, that rounded windows stay within the limit, and that input order does not
  change the result.

## Who uses it

`memplan::place` calls it for each part file on Apple and CPU targets, and makes one
`mmap` region per window. CUDA targets copy the weights into VRAM and have no windows.
