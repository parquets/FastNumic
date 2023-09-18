#pragma once

#include <math.h>

namespace fastnum {
namespace cpu {

void sgemm_nn(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

void sgemm_nt(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

void sgemm_tn(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

void sgemm_tt(int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc);

}  // namespace cpu
}  // namespace fastnum