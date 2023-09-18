#pragma once

#include "x86/mma_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

void mma_block(int mc, int nc, int kc, const float *packA, const float *packB, float *C, int ldc);

}  // namespace cpu
}  // namespace fastnum