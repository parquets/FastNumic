#pragma once

#include <math.h>

namespace fastnum {
namespace cpu {

void sgemm_nn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void sgemm_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void sgemm_tn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void sgemm_tt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

void sgemm(bool AT,
           bool BT,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc);

void dgemm_nn(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

void dgemm_nt(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

void dgemm_tn(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

void dgemm_tt(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc);

void dgemm(bool AT,
           bool BT,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc);

}  // namespace cpu


namespace cuda {

void sgemm_nn(int M,
              int N,
              int K,
              float alpha,
              const float* A,
              int lda,
              const float* B,
              int ldb,
              float beta,
              float* C,
              int ldc);
           
}  // namespace cuda
}  // namespace fastnum