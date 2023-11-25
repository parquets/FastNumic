#include "gemv.hpp"
#include <math.h>
#include <memory.h>
#include <stdlib.h>

#include "cpu/x86/dmvma_x86.hpp"
#include "cpu/x86/smvma_x86.hpp"

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C) {
    int m = 0;
    for (; m < M - 5; m += 6) {
        kernel::smvma_nr6(N, alpha, A, lda, B, beta, C);
        A += 6 * lda;
        C += 6;
    }
    for (; m < M - 3; m += 4) {
        kernel::smvma_nr4(N, alpha, A, lda, B, beta, C);
        A += 4 * lda;
        C += 4;
    }
    for (; m < M; ++m) {
        kernel::smvma_nr1(N, alpha, A, lda, B, beta, C);
        A += lda;
        C += 1;
    }
}

void dgemv_n(int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    int m = 0;
    for (; m < M - 5; m += 6) {
        kernel::dmvma_nr6(N, alpha, A, lda, B, beta, C);
        A += 6 * lda;
        C += 6;
    }
    for (; m < M - 3; m += 4) {
        kernel::dmvma_nr4(N, alpha, A, lda, B, beta, C);
        A += 4 * lda;
        C += 4;
    }
    for (; m < M; ++m) {
        kernel::dmvma_nr1(N, alpha, A, lda, B, beta, C);
        A += lda;
        C += 1;
    }
}

void sgemv_t(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C) {
    int m = 0;
    for (; m < M - 5; m += 6) {
        kernel::smvma_tr6(N, alpha, A, lda, B, beta, C);
        A += 6 * lda;
        B += 6;
    }
    for (; m < M - 3; m += 4) {
        kernel::smvma_tr4(N, alpha, A, lda, B, beta, C);
        A += 4 * lda;
        B += 4;
    }
    for (; m < M; ++m) {
        kernel::smvma_tr1(N, alpha, A, lda, B, beta, C);
        A += lda;
        B += 1;
    }
}

void dgemv_t(int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    int m = 0;
    for (; m < M - 5; m += 6) {
        kernel::dmvma_tr6(N, alpha, A, lda, B, beta, C);
        A += 6 * lda;
        B += 6;
    }
    for (; m < M - 3; m += 4) {
        kernel::dmvma_tr4(N, alpha, A, lda, B, beta, C);
        A += 4 * lda;
        B += 4;
    }
    for (; m < M; ++m) {
        kernel::dmvma_tr1(N, alpha, A, lda, B, beta, C);
        A += lda;
        B += 1;
    }
}

void sgemv(bool AT, int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C) {
    if (AT) {
        sgemv_t(M, N, alpha, A, lda, B, beta, C);
    } else {
        sgemv_n(M, N, alpha, A, lda, B, beta, C);
    }
}
void dgemv(bool AT, int M, int N, double alpha, const double* A, int lda, const double* B, double beta, double* C) {
    if (AT) {
        dgemv_t(M, N, alpha, A, lda, B, beta, C);
    } else {
        dgemv_n(M, N, alpha, A, lda, B, beta, C);
    }
}

}  // namespace cpu
}  // namespace fastnum