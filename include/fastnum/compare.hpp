#pragma once

namespace fastnum {
namespace cpu {

void compare_min(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void compare_max(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);

}  // namespace cpu
}  // namespace fastnum