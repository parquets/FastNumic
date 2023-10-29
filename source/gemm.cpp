#include "gemm.hpp"
#include <algorithm>
#include "cpu/mma_block.hpp"
#include "cpu/mpack_block.hpp"

#define MC 144
#define NC 144
#define KC 256

namespace fastnum {
namespace cpu {

void sgemm_nn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
              int ldc) {
    /*
    A -> M x K
    B -> K x N
    C -> M x N
    */

    float* packA = (float*)malloc(MC * KC * sizeof(float));
    float* packB = (float*)malloc(KC * NC * sizeof(float));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            h_packA(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc, KC);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                v_packB(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc, KC);
                mma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
              int ldc) {

    /*
    A -> M x K
    B -> N x K
    C -> M x N
    */

    float* packA = (float*)malloc(MC * KC * sizeof(float));
    float* packB = (float*)malloc(KC * NC * sizeof(float));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            h_packA(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc, KC);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                h_packB(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc, KC);
                mma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_tn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
              int ldc) {
    /*
    A -> K x M
    B -> K x N
    C -> M x N
    */

    float* packA = (float*)malloc(MC * KC * sizeof(float));
    float* packB = (float*)malloc(KC * NC * sizeof(float));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            v_packA(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc, KC);

            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                v_packB(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc, KC);
                mma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_tt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C,
              int ldc) {
    /*
    A -> K x M
    B -> N x K
    C -> M x N
    */

    float* packA = (float*)malloc(MC * KC * sizeof(float));
    float* packB = (float*)malloc(KC * NC * sizeof(float));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);

            v_packA(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc, KC);

            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                h_packB(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc, KC);
                mma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm(bool AT, bool BT, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if(AT) {
        if(BT) sgemm_tt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else sgemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        if(BT) sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else sgemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

}  // namespace cpu
}  // namespace fastnum