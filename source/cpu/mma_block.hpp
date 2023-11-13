#pragma once

#include "x86/smma_x86.hpp"
#include "x86/dmma_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

void smma_block(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc);

void dmma_block(int mc, int nc, int kc, const double* packA, const double* packB, double* C, int ldc);

}  // namespace cpu
}  // namespace fastnum