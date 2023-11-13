#pragma once

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
                    int ldc);

void batch_sgemm_nt(int batch,
                    int M,
                    int N,
                    int K,
                    int ldbc,
                    float alpha,
                    const float* A,
                    int lda,
                    const float* B,
                    int ldb,
                    float beta,
                    float* C,
                    int ldc);

void batch_sgemm_tn(int batch,
                    int M,
                    int N,
                    int K,
                    int ldbc,
                    float alpha,
                    const float* A,
                    int lda,
                    const float* B,
                    int ldb,
                    float beta,
                    float* C,
                    int ldc);

void batch_sgemm_tt(int batch,
                    int M,
                    int N,
                    int K,
                    int ldbc,
                    float alpha,
                    const float* A,
                    int lda,
                    const float* B,
                    int ldb,
                    float beta,
                    float* C,
                    int ldc);

void batch_sgemm(bool AT,
                 bool BT,
                 int batch,
                 int M,
                 int N,
                 int K,
                 int ldbc,
                 float alpha,
                 const float* A,
                 int lda,
                 const float* B,
                 int ldb,
                 float beta,
                 float* C,
                 int ldc);

}  // namespace cpu
}  // namespace fastnum