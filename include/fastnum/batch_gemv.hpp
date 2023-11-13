#pragma once

namespace fastnum {
namespace cpu {

void batch_sgemv_n(int batch,
                   int M,
                   int N,
                   float alpha,
                   const float* A,
                   int ldbatch_a,
                   int lda,
                   const float* B,
                   int ldbatch_b,
                   float beta,
                   float* C,
                   int ldbatch_c);
void batch_dgemv_n(int batch,
                   int M,
                   int N,
                   double alpha,
                   const double* A,
                   int ldbatch_a,
                   int lda,
                   const double* B,
                   int ldbatch_b,
                   double beta,
                   double* C,
                   int ldbatch_c);
void batch_sgemv_t(int batch,
                   int M,
                   int N,
                   float alpha,
                   const float* A,
                   int ldbatch_a,
                   int lda,
                   const float* B,
                   int ldbatch_b,
                   float beta,
                   float* C,
                   int ldbatch_c);

void batch_dgemv_t(int batch,
                   int M,
                   int N,
                   double alpha,
                   const double* A,
                   int ldbatch_a,
                   int lda,
                   const double* B,
                   int ldbatch_b,
                   double beta,
                   double* C,
                   int ldbatch_c);

void batch_sgemv(bool AT,
                 int batch,
                 int M,
                 int N,
                 double alpha,
                 const double* A,
                 int ldbatch_a,
                 int lda,
                 const double* B,
                 int ldbatch_b,
                 double beta,
                 double* C,
                 int ldbatch_c);

void batch_dgemv(bool AT,
                 int batch,
                 int M,
                 int N,
                 double alpha,
                 const double* A,
                 int ldbatch_a,
                 int lda,
                 const double* B,
                 int ldbatch_b,
                 double beta,
                 double* C,
                 int ldbatch_c);

}  // namespace cpu
}  // namespace fastnum