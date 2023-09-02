#pragma once

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, const float* A, int lda, const float* B, float* C);
void sgemv_t(int M, int N, const float* A, int lda, const float* B, float* C);

}
}