#pragma once

#include "x86/mma_kernel_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

void mma_block_8x8(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_4x4(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_4x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_8x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_1x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_6x16(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_16x6(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void mma_block_16x6(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);


void mma_block(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

}  // namespace cpu
}  // namespace fastnum