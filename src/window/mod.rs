//! Author: Joe Ricks
//! File: src/window/mod.rs
//! Description: Splits a mapped GGUF file into windows no larger than one GPU buffer.

/// Purpose: This struct holds one tensor's file offset and size in a mapped file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Where its data starts in the file.
    pub file_offset: u64,
    /// Its size in bytes.
    pub bytes: u64,
}

/// Purpose: This struct holds one window over a mapped file: where it starts and how long
/// it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Where the window starts in the file. A multiple of the page size, so the
    /// program can hand `base + file_start` to Metal without copying.
    pub file_start: u64,
    /// Its length in bytes, no more than the buffer limit.
    pub bytes: u64,
}

/// Purpose: This enum lists the errors splitting a file into windows can return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// One tensor, padded out to the pages it touches, is larger than a GPU buffer.
    TensorTooLarge { index: usize, bytes: u64, limit: u64 },
}

/// Purpose: This function splits one mapped GGUF file's tensors into windows a GPU can
/// bind.
///
/// Inputs:
/// - `spans`: each tensor's file offset and size.
/// - `limit`: the largest window in bytes.
/// - `page`: the page size each window must start on.
///
/// Outputs: the windows, and the window index of each tensor, or an error when one tensor
/// is larger than `limit`.
pub fn windows(
    spans: &[Span],
    limit: u64,
    page: u64,
) -> Result<(Vec<Window>, Vec<usize>), Error>
{
    // A Metal no-copy buffer takes a length in whole pages, so the program rounds each
    // window up to the end of its last page, as ggml's Metal backend does. A limit of
    // whole pages keeps that rounded length within the GPU's buffer limit.
    let page_size = page.max(1);
    let limit = if limit == u64::MAX { limit } else { limit / page_size * page_size };
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by_key(|&i| (spans[i].file_offset, i));

    let mut out: Vec<Window> = Vec::new();
    let mut which = vec![0usize; spans.len()];
    for &i in &order {
        let s = spans[i];
        let end = s.file_offset + s.bytes;
        let fits_current = out.last().is_some_and(|w| end - w.file_start <= limit);
        if !fits_current {
            let file_start = if out.is_empty() { 0 } else { s.file_offset / page * page };
            if end - file_start > limit {
                return Err(Error::TensorTooLarge { index: i, bytes: end - file_start, limit });
            }
            out.push(Window { file_start, bytes: 0 });
        }
        let w = out.last_mut().expect("a window was just ensured");
        w.bytes = w.bytes.max(end - w.file_start);
        which[i] = out.len() - 1;
    }
    Ok((out, which))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 16384;

    /// Purpose: This function builds one test tensor span.
    ///
    /// Inputs:
    /// - `file_offset`: where the tensor starts in the file.
    /// - `bytes`: its size in bytes.
    ///
    /// Outputs: the [`Span`].
    fn span(
        file_offset: u64,
        bytes: u64,
    ) -> Span
    {
        Span { file_offset, bytes }
    }

    #[test]
    fn without_a_limit_one_window_starts_at_zero()
    {
        let spans = [span(6_000_000, 1000), span(6_001_024, 5000)];
        let (ws, which) = windows(&spans, u64::MAX, 4096).unwrap();
        assert_eq!(ws, vec![Window { file_start: 0, bytes: 6_001_024 + 5000 }]);
        assert_eq!(which, vec![0, 0]);
    }

    #[test]
    fn a_limit_splits_and_every_tensor_fits_one_window()
    {
        let spans: Vec<Span> = (0..10).map(|i| span(PAGE + i * 300_000, 300_000)).collect();
        let (ws, which) = windows(&spans, 1_000_000, PAGE).unwrap();
        assert!(ws.len() > 1);
        for (s, &w) in spans.iter().zip(&which) {
            let w = ws[w];
            assert!(w.file_start <= s.file_offset && s.file_offset + s.bytes <= w.file_start + w.bytes);
            assert!(w.bytes <= 1_000_000);
            assert_eq!(w.file_start % PAGE, 0);
        }
    }

    #[test]
    fn a_window_rounded_up_to_whole_pages_stays_within_the_limit()
    {
        // A limit that is not whole pages, as a GPU could report.
        let limit = 10 * PAGE + 123;
        let spans: Vec<Span> = (0..40).map(|i| span(PAGE + i * 70_001, 70_001)).collect();
        let (ws, _) = windows(&spans, limit, PAGE).unwrap();
        assert!(ws.len() > 1);
        assert!(ws.iter().all(|w| w.bytes.div_ceil(PAGE) * PAGE <= limit), "{ws:?}");
    }

    #[test]
    fn a_tensor_larger_than_the_limit_is_refused()
    {
        let err = windows(&[span(PAGE, 10), span(2 * PAGE, 5_000_000)], 1_000_000, PAGE).unwrap_err();
        assert!(matches!(err, Error::TensorTooLarge { index: 1, .. }), "{err:?}");
    }

    #[test]
    fn input_order_does_not_matter()
    {
        let a = [span(PAGE, 600_000), span(PAGE + 600_000, 600_000)];
        let b = [a[1], a[0]];
        let (wa, ia) = windows(&a, 1_000_000, PAGE).unwrap();
        let (wb, ib) = windows(&b, 1_000_000, PAGE).unwrap();
        assert_eq!(wa, wb);
        assert_eq!((ia[0], ia[1]), (ib[1], ib[0]));
    }
}
