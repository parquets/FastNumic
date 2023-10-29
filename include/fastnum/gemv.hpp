#pragma once

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C);
void sgemv_t(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C);

// void sgemv(bool AT, int M, int N, const float* A, int lda, const float* B, float* C);

}
}