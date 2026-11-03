//! Author: Joe Ricks
//! File: src/refback/src/quant.rs
//! Description: Decodes the weight types the compute check covers, and runs the plain
//! Rust matvec and its f64 reference.
//!
//! The block layouts are ggml's, from `ggml/src/ggml-common.h` and
//! `ggml/src/ggml-quants.c` at llama.cpp commit 52b3df0023659b142ce29f75c7a82cf437769c33:
//!
//! - `block_q8_0` is `{ ggml_half d; int8_t qs[QK8_0]; }` with `QK8_0 = 32`, 34 bytes.
//!   `dequantize_row_q8_0` gives element `j` of a block as `qs[j] * d`.
//! - `block_q4_0` is `{ ggml_half d; uint8_t qs[QK4_0 / 2]; }` with `QK4_0 = 32`, 18
//!   bytes. `dequantize_row_q4_0` gives element `j` as `((qs[j] & 0x0F) - 8) * d` and
//!   element `j + 16` as `((qs[j] >> 4) - 8) * d`, for `j` in 0..16.
//! - `ggml_half` is an IEEE 754 binary16, little-endian, which `GGML_FP16_TO_FP32`
//!   widens exactly.

/// `GGML_TYPE_F32`.
pub const F32: u32 = 0;
/// `GGML_TYPE_F16`.
pub const F16: u32 = 1;
/// `GGML_TYPE_Q4_0`.
pub const Q4_0: u32 = 2;
/// `GGML_TYPE_Q8_0`.
pub const Q8_0: u32 = 8;

/// Elements in one Q4_0 or Q8_0 block: `QK4_0` and `QK8_0`.
pub const QK: usize = 32;
/// Bytes in one `block_q8_0`.
pub const Q8_0_BLOCK: usize = 2 + QK;
/// Bytes in one `block_q4_0`.
pub const Q4_0_BLOCK: usize = 2 + QK / 2;

/// Purpose: This function checks whether the compute check covers a ggml type.
///
/// Inputs:
/// - `ty`: the `ggml_type`.
///
/// Outputs: `true` for F32, F16, Q8_0 and Q4_0.
pub fn covered(
    ty: u32,
) -> bool
{
    matches!(ty, F32 | F16 | Q4_0 | Q8_0)
}

/// Purpose: This function returns the bytes one row of a covered type takes.
///
/// Inputs:
/// - `ty`: the `ggml_type`.
/// - `cols`: the elements in the row.
///
/// Outputs: the row's size, or `None` for a type the check does not cover or a row that
/// is not whole blocks.
pub fn row_bytes(
    ty: u32,
    cols: usize,
) -> Option<usize>
{
    match ty {
        F32 => Some(cols * 4),
        F16 => Some(cols * 2),
        Q8_0 if cols.is_multiple_of(QK) => Some(cols / QK * Q8_0_BLOCK),
        Q4_0 if cols.is_multiple_of(QK) => Some(cols / QK * Q4_0_BLOCK),
        _ => None,
    }
}

/// Purpose: This function widens an IEEE binary16 value to f32, exactly.
///
/// Inputs:
/// - `h`: the half's bits.
///
/// Outputs: the same value as an f32.
pub fn f16_to_f32(
    h: u16,
) -> f32
{
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // A subnormal half is man * 2^-24, which f32 holds exactly.
        (0, m) => {
            let v = m as f32 * f32::from_bits(0x3380_0000);
            return if sign != 0 { -v } else { v };
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Purpose: This function rounds an f32 to the nearest IEEE binary16, ties to even.
///
/// Inputs:
/// - `v`: the value.
///
/// Outputs: the half's bits.
pub fn f32_to_f16(
    v: f32,
) -> u16
{
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        // A subnormal half counts units of 2^-24.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let (r, rem, half) = (m >> shift, m & ((1 << shift) - 1), 1u32 << (shift - 1));
        let r = if rem > half || (rem == half && r & 1 == 1) { r + 1 } else { r };
        return sign | r as u16;
    }
    let r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    // A carry out of the mantissa moves into the exponent, which is the right rounding.
    let r = if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) { r + 1 } else { r };
    sign | r as u16
}

/// Purpose: This function reads a little-endian u16 from the start of a byte slice.
///
/// Inputs:
/// - `b`: at least two bytes.
///
/// Outputs: the value.
fn le16(
    b: &[u8],
) -> u16
{
    u16::from_le_bytes([b[0], b[1]])
}

/// Purpose: This function decodes one row of a covered type into f32 values, as ggml's
/// `dequantize_row_*` functions do.
///
/// Inputs:
/// - `ty`: the `ggml_type`, one [`covered`] accepts.
/// - `row`: the row's bytes, [`row_bytes`] long.
/// - `out`: where the values go, one per element of the row.
///
/// Outputs: none. Every value is exact in f32: a Q8_0 or Q4_0 element is a small integer
/// times an f16 scale.
pub fn decode_row(
    ty: u32,
    row: &[u8],
    out: &mut [f32],
)
{
    match ty {
        F32 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(4)) {
                *o = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
        }
        F16 => {
            for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = f16_to_f32(le16(b));
            }
        }
        Q8_0 => {
            for (o, blk) in out.chunks_exact_mut(QK).zip(row.chunks_exact(Q8_0_BLOCK)) {
                let d = f16_to_f32(le16(blk));
                for (v, q) in o.iter_mut().zip(&blk[2..]) {
                    *v = f32::from(*q as i8) * d;
                }
            }
        }
        Q4_0 => {
            for (o, blk) in out.chunks_exact_mut(QK).zip(row.chunks_exact(Q4_0_BLOCK)) {
                let d = f16_to_f32(le16(blk));
                let (lo, hi) = o.split_at_mut(QK / 2);
                for (j, q) in blk[2..].iter().enumerate() {
                    lo[j] = f32::from(i16::from(q & 0x0f) - 8) * d;
                    hi[j] = f32::from(i16::from(q >> 4) - 8) * d;
                }
            }
        }
        _ => unreachable!("decode_row is called only for covered types"),
    }
}

/// Purpose: This function builds the fixed input vector the compute check multiplies by.
///
/// Inputs:
/// - `cols`: its length, a weight's row length.
///
/// Outputs: values in [-1, 1], each a multiple of 1/8 so it is exact in f32.
pub fn input(
    cols: usize,
) -> Vec<f32>
{
    (0..cols).map(|c| ((c * 7 + 3) % 17) as f32 / 8.0 - 1.0).collect()
}

/// Purpose: This function computes y = W·x the way a naive CPU kernel does: f32 values,
/// f32 sums, one row at a time.
///
/// Inputs:
/// - `ty`: W's `ggml_type`, one [`covered`] accepts.
/// - `w`: W's bytes, `rows` rows of [`row_bytes`] each.
/// - `cols`: the elements in a row.
/// - `rows`: the number of rows.
/// - `x`: the input vector, `cols` long.
///
/// Outputs: y, one value per row.
pub fn matvec(
    ty: u32,
    w: &[u8],
    cols: usize,
    rows: usize,
    x: &[f32],
) -> Vec<f32>
{
    let rb = row_bytes(ty, cols).expect("matvec is called only for covered types and whole rows");
    let mut row = vec![0f32; cols];
    (0..rows)
        .map(|r| {
            decode_row(ty, &w[r * rb..(r + 1) * rb], &mut row);
            row.iter().zip(x).fold(0f32, |acc, (a, b)| acc + a * b)
        })
        .collect()
}

/// Purpose: This struct holds the f64 reference for y = W·x and how far an f32 result may
/// be from it.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    /// y, one value per row, summed in f64.
    pub y: Vec<f64>,
    /// For each row, the largest error an f32 kernel may have, in any summation order.
    pub bound: Vec<f64>,
}

/// Purpose: This function computes the f64 reference for y = W·x from W's bytes.
///
/// Inputs:
/// - `ty`: W's `ggml_type`, one [`covered`] accepts.
/// - `w`: W's bytes, read from the file.
/// - `cols`: the elements in a row.
/// - `rows`: the number of rows.
/// - `x`: the input vector, `cols` long.
///
/// Outputs: the [`Reference`]. Each product of two f32 values is exact in f64. An f32
/// kernel rounds each product, or each of a product's two steps, and each of its `cols`
/// sums, so its error is at most `(cols + 2) * 2^-23 * sum(|w * x|)`.
pub fn reference(
    ty: u32,
    w: &[u8],
    cols: usize,
    rows: usize,
    x: &[f32],
) -> Reference
{
    let rb = row_bytes(ty, cols).expect("reference is called only for covered types and whole rows");
    let mut row = vec![0f32; cols];
    let mut y = Vec::with_capacity(rows);
    let mut bound = Vec::with_capacity(rows);
    for r in 0..rows {
        decode_row(ty, &w[r * rb..(r + 1) * rb], &mut row);
        let (mut sum, mut abs) = (0f64, 0f64);
        for (a, b) in row.iter().zip(x) {
            let p = f64::from(*a) * f64::from(*b);
            sum += p;
            abs += p.abs();
        }
        y.push(sum);
        bound.push((cols as f64 + 2.0) * f64::from(f32::EPSILON) * abs + f64::from(f32::MIN_POSITIVE));
    }
    Reference { y, bound }
}

/// Purpose: This function compares a kernel's y with the reference, row by row.
///
/// Inputs:
/// - `got`: the kernel's y.
/// - `want`: the reference.
///
/// Outputs: the largest error as a fraction of its row's bound, or a message naming the
/// first row past its bound.
pub fn compare(
    got: &[f32],
    want: &Reference,
) -> Result<f64, String>
{
    if got.len() != want.y.len() {
        return Err(format!("the kernel gave {} values for {} rows", got.len(), want.y.len()));
    }
    let mut worst = 0f64;
    for (r, ((g, w), b)) in got.iter().zip(&want.y).zip(&want.bound).enumerate() {
        let err = (f64::from(*g) - w).abs();
        if err.is_nan() || err > *b {
            return Err(format!("row {r} is {g}, and the reference is {w}, off by {err:.3e} where at most {b:.3e} is allowed"));
        }
        worst = worst.max(err / b);
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_widen_and_round_trip()
    {
        for (bits, v) in [(0x3c00u16, 1.0f32), (0xc000, -2.0), (0x3555, 1365.0 / 4096.0), (0x0001, 1.0 / 16_777_216.0), (0x7bff, 65504.0), (0x8000, -0.0)] {
            assert_eq!(f16_to_f32(bits).to_bits(), v.to_bits(), "{bits:#06x}");
            assert_eq!(f32_to_f16(v), bits, "{v}");
        }
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00, "a tie rounds to even");
        assert_eq!(f32_to_f16(1e6), 0x7c00, "too large is infinity");
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn q8_0_decodes_as_ggml_does()
    {
        // d = 0.5, qs = -3, 0, 1, ... : element j is qs[j] * d.
        let mut blk = f32_to_f16(0.5).to_le_bytes().to_vec();
        blk.extend((0..32).map(|j: i32| (j - 3) as i8 as u8));
        let mut out = [0f32; 32];
        decode_row(Q8_0, &blk, &mut out);
        assert_eq!(out[0], -1.5);
        assert_eq!(out[3], 0.0);
        assert_eq!(out[31], 14.0);
    }

    #[test]
    fn q4_0_puts_low_nibbles_first_and_high_nibbles_after()
    {
        // d = 2; byte j holds low nibble j % 16 and high nibble 15 - j % 16.
        let mut blk = f32_to_f16(2.0).to_le_bytes().to_vec();
        blk.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let mut out = [0f32; 32];
        decode_row(Q4_0, &blk, &mut out);
        assert_eq!(out[0], (0.0 - 8.0) * 2.0, "element 0 is byte 0's low nibble");
        assert_eq!(out[15], (15.0 - 8.0) * 2.0);
        assert_eq!(out[16], (15.0 - 8.0) * 2.0, "element 16 is byte 0's high nibble");
        assert_eq!(out[31], (0.0 - 8.0) * 2.0);
    }

    #[test]
    fn the_cpu_matvec_meets_its_bound_and_a_wrong_matrix_does_not()
    {
        let (cols, rows) = (64, 3);
        let w: Vec<u8> = (0..cols * rows).flat_map(|i| f32_to_f16((i % 29) as f32 / 16.0 - 0.9).to_le_bytes()).collect();
        let x = input(cols);
        let r = reference(F16, &w, cols, rows, &x);
        assert!(compare(&matvec(F16, &w, cols, rows, &x), &r).is_ok());
        let shifted = &w[2..];
        let mut wrong = shifted.to_vec();
        wrong.extend([0, 0]);
        assert!(compare(&matvec(F16, &wrong, cols, rows, &x), &r).is_err());
    }
}
