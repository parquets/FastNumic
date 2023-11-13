#pragma once

#include <math.h>

namespace fastnum {
namespace cpu {

void transpose(int M, int N, const float* A, int lda, float* B, int ldb);

void transpose(int M, int N, const double* A, int lda, double* B, int ldb);
}  // namespace cpu
}  // namespace fastnum