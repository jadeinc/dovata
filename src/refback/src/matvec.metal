// Author: Joe Ricks
// File: src/refback/src/matvec.metal
// Description: Naive y = W·x kernels for F32, F16, Q8_0 and Q4_0 weights, one thread per
// row of W. refback binds W at its planned (buffer, offset) and compares y with an f64
// reference computed from the model file.
//
// Block layouts follow ggml/src/ggml-common.h (block_q8_0, block_q4_0) and the
// dequantize_row_q8_0 and dequantize_row_q4_0 functions of ggml/src/ggml-quants.c, at
// llama.cpp commit 52b3df0023659b142ce29f75c7a82cf437769c33. W is read one byte at a
// time, so a weight at any offset, including a misaligned one, is read without a
// misaligned load.

#include <metal_stdlib>
using namespace metal;

// Purpose: This struct holds the matvec's shape. It matches `Shape` in metal.rs.
struct Shape {
    uint cols;       // elements in one row of W, and the length of x
    uint rows;       // rows of W, and the length of y
    ulong row_bytes; // bytes in one row of W
};

// Purpose: This function reads a little-endian 16-bit value from any address.
//
// Inputs:
// - `p`: the first byte.
//
// Outputs: the value.
static inline ushort ld16(device const uchar *p) {
    return ushort(p[0]) | (ushort(p[1]) << 8);
}

// Purpose: This function reads a little-endian 32-bit value from any address.
//
// Inputs:
// - `p`: the first byte.
//
// Outputs: the value.
static inline uint ld32(device const uchar *p) {
    return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
}

// Purpose: This function computes one row of y = W·x for F32 weights.
//
// Inputs:
// - `w`: W, bound at its planned offset.
// - `x`: the input vector.
// - `y`: the output vector.
// - `s`: the shape.
// - `r`: the row this thread computes.
//
// Outputs: none; it writes y[r].
kernel void matvec_f32(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]], constant Shape &s [[buffer(3)]],
                       uint r [[thread_position_in_grid]]) {
    if (r >= s.rows) return;
    device const uchar *row = w + ulong(r) * s.row_bytes;
    float acc = 0.0f;
    for (uint c = 0; c < s.cols; ++c) {
        acc += as_type<float>(ld32(row + 4 * c)) * x[c];
    }
    y[r] = acc;
}

// Purpose: This function computes one row of y = W·x for F16 weights.
//
// Inputs: as matvec_f32.
//
// Outputs: none; it writes y[r].
kernel void matvec_f16(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                       device float *y [[buffer(2)]], constant Shape &s [[buffer(3)]],
                       uint r [[thread_position_in_grid]]) {
    if (r >= s.rows) return;
    device const uchar *row = w + ulong(r) * s.row_bytes;
    float acc = 0.0f;
    for (uint c = 0; c < s.cols; ++c) {
        acc += float(as_type<half>(ld16(row + 2 * c))) * x[c];
    }
    y[r] = acc;
}

// Purpose: This function computes one row of y = W·x for Q8_0 weights: 34-byte blocks of
// an f16 scale d and 32 signed bytes, element j being qs[j] * d.
//
// Inputs: as matvec_f32.
//
// Outputs: none; it writes y[r].
kernel void matvec_q8_0(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]], constant Shape &s [[buffer(3)]],
                        uint r [[thread_position_in_grid]]) {
    if (r >= s.rows) return;
    device const uchar *row = w + ulong(r) * s.row_bytes;
    float acc = 0.0f;
    for (uint b = 0; b < s.cols / 32; ++b) {
        device const uchar *blk = row + 34 * b;
        float d = float(as_type<half>(ld16(blk)));
        for (uint j = 0; j < 32; ++j) {
            acc += float(as_type<char>(blk[2 + j])) * d * x[32 * b + j];
        }
    }
    y[r] = acc;
}

// Purpose: This function computes one row of y = W·x for Q4_0 weights: 18-byte blocks of
// an f16 scale d and 16 bytes of nibbles. Byte j's low nibble is element j and its high
// nibble element j + 16, each stored plus 8.
//
// Inputs: as matvec_f32.
//
// Outputs: none; it writes y[r].
kernel void matvec_q4_0(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]], constant Shape &s [[buffer(3)]],
                        uint r [[thread_position_in_grid]]) {
    if (r >= s.rows) return;
    device const uchar *row = w + ulong(r) * s.row_bytes;
    float acc = 0.0f;
    for (uint b = 0; b < s.cols / 32; ++b) {
        device const uchar *blk = row + 18 * b;
        float d = float(as_type<half>(ld16(blk)));
        for (uint j = 0; j < 16; ++j) {
            uchar q = blk[2 + j];
            acc += float(int(q & 0x0F) - 8) * d * x[32 * b + j];
            acc += float(int(q >> 4) - 8) * d * x[32 * b + j + 16];
        }
    }
    y[r] = acc;
}
