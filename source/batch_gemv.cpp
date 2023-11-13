#include "fastnum/batch_gemv.hpp"
#include "fastnum/gemv.hpp"

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
                   int ldbatch_c) {
    for (int b = 0; b < batch; ++b) {
        sgemv_t(M, N, alpha, A, lda, B, beta, C);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}
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
                   int ldbatch_c) {
    for (int b = 0; b < batch; ++b) {
        dgemv_n(M, N, alpha, A, lda, B, beta, C);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}
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
                   int ldbatch_c) {
    for (int b = 0; b < batch; ++b) {
        sgemv_t(M, N, alpha, A, lda, B, beta, C);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}
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
                   int ldbatch_c) {
    for (int b = 0; b < batch; ++b) {
        dgemv_t(M, N, alpha, A, lda, B, beta, C);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}

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
                 int ldbatch_c) {
    if (AT) {
        batch_sgemv_t(batch, M, N, alpha, A, ldbatch_a, lda, B, ldbatch_b, beta, C, ldbatch_c);
    } else {
        batch_sgemv_n(batch, M, N, alpha, A, ldbatch_a, lda, B, ldbatch_b, beta, C, ldbatch_c);
    }
}

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
                 int ldbatch_c) {
    if (AT) {
        batch_dgemv_t(batch, M, N, alpha, A, ldbatch_a, lda, B, ldbatch_b, beta, C, ldbatch_c);
    } else {
        batch_dgemv_n(batch, M, N, alpha, A, ldbatch_a, lda, B, ldbatch_b, beta, C, ldbatch_c);
    }
}

}  // namespace cpu
}  // namespace fastnum