#include "gemv.hpp"
#include <math.h>
#include <memory.h>
#include <stdlib.h>

#include "cpu/mma_block.hpp"
#include "cpu/mpack_block.hpp"
#include "cpu/x86/mmpack_kernel_x86.hpp"

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, const float* A, int lda, const float* B, float* C) {
    float* packA = (float*)malloc(sizeof(float) * 8 * N);

    int m = 0;
    for (m = 0; m < M - 7; m += 8) {
        mpack_block_h8(8, N, packA, A, lda, N, N);
        mma_block_8x1(8, 1, N, packA, B, C, 1);
        A += 8 * lda;
        C += 8;
    }
    for (; m < M; ++m) {
        kernel::mma_1x1(A, B, N, C, 1);
        A += lda;
        C += 1;
    }
    free(packA);
}

void sgemv_t(int M, int N, const float* A, int lda, const float* B, float* C) {
    float* packA = (float*)malloc(sizeof(float) * 8 * M);

    int m = 0;
    for (m = 0; m < M - 7; m += 8) {
        mpack_block_v8(M, 8, packA, A, lda, M, M);
        mma_block_8x1(8, 1, N, packA, B, C, 1);
        A += 8;
        C += 8;
    }

    kernel::mpack_vx(packA, A, lda, (M - m), M, M);
    kernel::mma_mx1(M - m, packA, B, M, C, 1);

    free(packA);
}

}  // namespace cpu
}  // namespace fastnum