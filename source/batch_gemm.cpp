#include "fastnum/batch_gemm.hpp"
#include "fastnum/gemm.hpp"

namespace fastnum {
namespace cpu {

void batch_sgemm_nn(int batch,
                    int M,
                    int N,
                    int K,
                    float alpha,
                    const float* A,
                    int ldbatch_a,
                    int lda,
                    const float* B,
                    int ldbatch_b,
                    int ldb,
                    float beta,
                    float* C,
                    int ldbatch_c,
                    int ldc) {
    for (int b = 0; b < batch; ++b) {
        batch_sgemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}

void batch_sgemm_nt(int batch,
                    int M,
                    int N,
                    int K,
                    float alpha,
                    const float* A,
                    int ldbatch_a,
                    int lda,
                    const float* B,
                    int ldbatch_b,
                    int ldb,
                    float beta,
                    float* C,
                    int ldbatch_c,
                    int ldc) {
    for (int b = 0; b < batch; ++b) {
        batch_sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}

void batch_sgemm_tn(int batch,
                    int M,
                    int N,
                    int K,
                    float alpha,
                    const float* A,
                    int ldbatch_a,
                    int lda,
                    const float* B,
                    int ldbatch_b,
                    int ldb,
                    float beta,
                    float* C,
                    int ldbatch_c,
                    int ldc) {
    for (int b = 0; b < batch; ++b) {
        batch_sgemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}

void batch_sgemm_tt(int batch,
                    int M,
                    int N,
                    int K,
                    float alpha,
                    const float* A,
                    int ldbatch_a,
                    int lda,
                    const float* B,
                    int ldbatch_b,
                    int ldb,
                    float beta,
                    float* C,
                    int ldbatch_c,
                    int ldc) {
    for (int b = 0; b < batch; ++b) {
        batch_sgemm_tt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        A += ldbatch_a;
        B += ldbatch_b;
        C += ldbatch_c;
    }
}

void batch_sgemm(bool AT,
                 bool BT,
                 int batch,
                 int M,
                 int N,
                 int K,
                 float alpha,
                 const float* A,
                 int ldbatch_a,
                 int lda,
                 const float* B,
                 int ldbatch_b,
                 int ldb,
                 float beta,
                 float* C,
                 int ldbatch_c,
                 int ldc) {
    if (AT) {
        if (BT) {
            batch_sgemm_tt(batch, M, N, K, alpha, A, ldbatch_a, lda, B, ldbatch_b, ldb, beta, C, ldbatch_c, ldc);
        } else {
            batch_sgemm_tn(batch, M, N, K, alpha, A, ldbatch_a, lda, B, ldbatch_b, ldb, beta, C, ldbatch_c, ldc);
        }
    } else {
        if (BT) {
            batch_sgemm_nt(batch, M, N, K, alpha, A, ldbatch_a, lda, B, ldbatch_b, ldb, beta, C, ldbatch_c, ldc);
        } else {
            batch_sgemm_nn(batch, M, N, K, alpha, A, ldbatch_a, lda, B, ldbatch_b, ldb, beta, C, ldbatch_c, ldc);
        }
    }
}

}  // namespace cpu
}  // namespace fastnum