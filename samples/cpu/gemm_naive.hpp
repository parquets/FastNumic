#pragma once

#include <math.h>
#include <stdio.h>

inline void
sgemm_naive_nn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

inline void
sgemm_naive_nt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[m * lda + k] * B[n * ldb + k];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

inline void
sgemm_naive_tn(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[k * lda + m] * B[k * ldb + n];
            }
            C[m * ldc + n] += tmp;
        }
    }
}

inline void
sgemm_naive_tt(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = beta * C[m * ldc + n];
            for (int k = 0; k < K; ++k) {
                tmp += alpha * A[k * lda + m] * B[n * ldb + k];
            }
            C[m * ldc + n] = tmp;
        }
    }
}