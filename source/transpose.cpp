#include "transpose.hpp"
#include "cpu/x86/transpose_x86.hpp"

namespace fastnum {
namespace cpu {

namespace kernel {

void transpose_common(const float* A, int a_rows, int a_cols, int lda, float* B, int ldb) {
    for (int c = 0; c < a_cols; ++c) {
        for (int r = 0; r < a_rows; ++r) {
            B[c * ldb + r] = A[r * lda + c];
        }
    }
}

}  // namespace kernel

static inline void transpose_block_16X16(const float* A, int lda, float* B, int ldb) {

    kernel::transpose_8x8(A, lda, B, ldb);
    kernel::transpose_8x8(A + 8, lda, B + 8 * ldb, ldb);
    kernel::transpose_8x8(A + 8 * lda, lda, B + 8, ldb);
    kernel::transpose_8x8(A + 8 * lda + 8, lda, B + 8 * ldb + 8, ldb);
}

static inline void transpose_block_32X16(const float* A, int lda, float* B, int ldb) {
    transpose_block_16X16(A, lda, B, ldb);
    transpose_block_16X16(A + 16 * lda, lda, B + 16, ldb);
}

static inline void transpose_block_32X32(const float* A, int lda, float* B, int ldb) {
    transpose_block_16X16(A, lda, B, ldb);
    transpose_block_16X16(A + 16, lda, B + 16 * ldb, ldb);
    transpose_block_16X16(A + 16 * lda, lda, B + 16, ldb);
    transpose_block_16X16(A + 16 * lda + 16, lda, B + 16 * ldb + 16, ldb);
}

static inline void transpose_block_16X32(const float* A, int lda, float* B, int ldb) {
    transpose_block_16X16(A, lda, B, ldb);
    transpose_block_16X16(A + 16, lda, B + 16 * ldb, ldb);
}

void transpose(int M, int N, const float* A, int lda, float* B, int ldb) {
    int r = 0;
    for (r = 0; r < M - 31; r += 32) {
        int c = 0;
        for (c = 0; c < N - 31; c += 32) {
            transpose_block_32X32(A + r * lda + c, lda, B + c * ldb + r, ldb);
        }
        for (; c < N - 15; c += 16) {
            transpose_block_32X16(A + r * lda + c, lda, B + c * ldb + r, ldb);
        }
        if (c < N)
            kernel::transpose_common(A + r * lda + c, 32, N - c, lda, B + c * ldb + r, ldb);
    }

    for (; r < M - 15; r += 16) {
        int c = 0;
        for (c = 0; c < N - 31; c += 32) {
            transpose_block_16X32(A + r * lda + c, lda, B + c * ldb + r, ldb);
        }
        for (; c < N - 15; c += 16) {
            transpose_block_16X16(A + r * lda + c, lda, B + c * ldb + r, ldb);
        }
        if (c < N)
            kernel::transpose_common(A + r * lda + c, 16, N - c, lda, B + c * ldb + r, ldb);
    }
    if (r < M)
        kernel::transpose_common(A + r * lda, M - r, N, lda, B + r, ldb);
}

}  // namespace cpu
}  // namespace fastnum