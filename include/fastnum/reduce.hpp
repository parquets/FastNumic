#pragma once

namespace fastnum {
namespace cpu {

void reduce_add(int M, int N, const float* A, int lda, float* C, int axis = -1);
void reduce_max(int M, int N, const float* A, int lda, float* C, int axis = -1);
void reduce_min(int M, int N, const float* A, int lda, float* C, int axis = -1);




}  // namespace cpu
}  // namespace fastnum