#pragma once

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C);
void dgemv_n(int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C);
void sgemv_t(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C);
void dgemv_t(int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C);

void sgemv(bool AT, int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C);
void dgemv(bool AT, int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C);

}  // namespace cpu
}  // namespace fastnum