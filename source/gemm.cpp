#include "gemm.hpp"
#include <algorithm>
#include "cpu/mma_block.hpp"
#include "cpu/pack_block.hpp"

#define MC 288
#define NC 288
#define KC 64

namespace fastnum {
namespace cpu {

void sgemm_nn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
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
            spackA_h(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                spackB_v(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc);
                smma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void dgemm_nn(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
    double* packA = (double*)malloc(MC * KC * sizeof(double));
    double* packB = (double*)malloc(KC * NC * sizeof(double));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            dpackA_h(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                dpackB_v(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc);
                dmma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
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
            spackA_h(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                spackB_h(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc);
                smma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void dgemm_nt(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
    double* packA = (double*)malloc(MC * KC * sizeof(double));
    double* packB = (double*)malloc(KC * NC * sizeof(double));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            dpackA_h(s_mc, s_kc, packA, A + mm * lda + kk, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                dpackB_h(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc);
                dmma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_tn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
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
            spackA_v(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc);

            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                spackB_v(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc);
                smma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void dgemm_tn(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
    double* packA = (double*)malloc(MC * KC * sizeof(double));
    double* packB = (double*)malloc(KC * NC * sizeof(double));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            dpackA_v(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc);

            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                dpackB_v(s_kc, s_nc, packB, B + kk * ldb + nn, ldb, s_kc);
                dmma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void sgemm_tt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
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
            spackA_v(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                spackB_h(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc);
                smma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

void dgemm_tt(int M, int N, int K, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc) {
        double* packA = (double*)malloc(MC * KC * sizeof(double));
    double* packB = (double*)malloc(KC * NC * sizeof(double));

    int mm, nn, kk;
    for (mm = 0; mm < M; mm += MC) {
        int s_mc = std::min(M - mm, MC);
        for (kk = 0; kk < K; kk += KC) {
            int s_kc = std::min(K - kk, KC);
            dpackA_v(s_kc, s_mc, packA, A + kk * lda + mm, lda, s_kc);
            for (nn = 0; nn < N; nn += NC) {
                int s_nc = std::min(N - nn, NC);
                dpackB_h(s_nc, s_kc, packB, B + nn * ldb + kk, ldb, s_kc);
                dmma_block(s_mc, s_nc, s_kc, packA, packB, C + mm * ldc + nn, ldc);
            }
        }
    }

    free(packA);
    free(packB);
}

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
           int ldc) {
    if (AT) {
        if (BT)
            sgemm_tt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            sgemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        if (BT)
            sgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            sgemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

void dgemm(bool AT,
           bool BT,
           int M,
           int N,
           int K,
           double alpha,
           const double* A,
           int lda,
           const double* B,
           int ldb,
           double beta,
           double* C,
           int ldc) {
    if (AT) {
        if (BT)
            dgemm_tt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            dgemm_tn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        if (BT)
            dgemm_nt(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            dgemm_nn(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

}  // namespace cpu
}  // namespace fastnum