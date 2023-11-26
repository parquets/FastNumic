#include "arithmetic.hpp"
#include "cpu/x86/arithmetic_x86.hpp"

namespace fastnum {
namespace cpu {

void add(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    int m = 0;

    for (; m < M - 3; m += 4) {
        kernel::add(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
        kernel::add(C + (m + 1) * ldc, A + (m + 1) * lda, B + (m + 1) * ldb, N);
        kernel::add(C + (m + 2) * ldc, A + (m + 2) * lda, B + (m + 2) * ldb, N);
        kernel::add(C + (m + 3) * ldc, A + (m + 3) * lda, B + (m + 3) * ldb, N);
    }

    for(; m<M; ++m) {
        kernel::add(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
    }
}

void sub(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    int m = 0;

    for (; m < M - 3; m += 4) {
        kernel::sub(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
        kernel::sub(C + (m + 1) * ldc, A + (m + 1) * lda, B + (m + 1) * ldb, N);
        kernel::sub(C + (m + 2) * ldc, A + (m + 2) * lda, B + (m + 2) * ldb, N);
        kernel::add(C + (m + 3) * ldc, A + (m + 3) * lda, B + (m + 3) * ldb, N);
    }

    for(; m<M; ++m) {
        kernel::sub(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
    }
}

void mul(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    int m = 0;

    for (; m < M - 3; m += 4) {
        kernel::mul(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
        kernel::mul(C + (m + 1) * ldc, A + (m + 1) * lda, B + (m + 1) * ldb, N);
        kernel::mul(C + (m + 2) * ldc, A + (m + 2) * lda, B + (m + 2) * ldb, N);
        kernel::mul(C + (m + 3) * ldc, A + (m + 3) * lda, B + (m + 3) * ldb, N);
    }

    for(; m<M; ++m) {
        kernel::mul(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
    }
}

void div(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    int m = 0;

    for (; m < M - 3; m += 4) {
        kernel::div(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
        kernel::div(C + (m + 1) * ldc, A + (m + 1) * lda, B + (m + 1) * ldb, N);
        kernel::div(C + (m + 2) * ldc, A + (m + 2) * lda, B + (m + 2) * ldb, N);
        kernel::div(C + (m + 3) * ldc, A + (m + 3) * lda, B + (m + 3) * ldb, N);
    }

    for(; m<M; ++m) {
        kernel::div(C + (m + 0) * ldc, A + (m + 0) * lda, B + (m + 0) * ldb, N);
    }
}

void add_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    int m = 0;

    for (; m < M - 7; m += 8) {
        kernel::add_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::add_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::add_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::add_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
        kernel::add_scaler(C + (m + 4) * ldc, A + (m + 4) * lda, B, N);
        kernel::add_scaler(C + (m + 5) * ldc, A + (m + 5) * lda, B, N);
        kernel::add_scaler(C + (m + 6) * ldc, A + (m + 6) * lda, B, N);
        kernel::add_scaler(C + (m + 7) * ldc, A + (m + 7) * lda, B, N);
    }
    for (; m < M - 3; m += 4) {
        kernel::add_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::add_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::add_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::add_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
    }
    for(; m<M; ++m) {
        kernel::add_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
    }
}

void sub_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    int m = 0;

    for (; m < M - 7; m += 8) {
        kernel::sub_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::sub_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::sub_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::sub_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
        kernel::sub_scaler(C + (m + 4) * ldc, A + (m + 4) * lda, B, N);
        kernel::sub_scaler(C + (m + 5) * ldc, A + (m + 5) * lda, B, N);
        kernel::sub_scaler(C + (m + 6) * ldc, A + (m + 6) * lda, B, N);
        kernel::sub_scaler(C + (m + 7) * ldc, A + (m + 7) * lda, B, N);
    }
    for (; m < M - 3; m += 4) {
        kernel::sub_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::sub_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::sub_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::sub_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
    }
    for(; m<M; ++m) {
        kernel::sub_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
    }
}

void mul_scaler(int M, int N, const float* A, int lda, const float B,  float* C, int ldc) {

    if(A == C && lda == ldc && B == 1) return;

    int m = 0;
    for (; m < M - 3; m += 4) {
        kernel::mul_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::mul_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::mul_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::mul_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
    }
    for(; m<M; ++m) {
        kernel::mul_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
    }
}

void div_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    int m = 0;

    for (; m < M - 7; m += 8) {
        kernel::div_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::div_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::div_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::div_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
        kernel::div_scaler(C + (m + 4) * ldc, A + (m + 4) * lda, B, N);
        kernel::div_scaler(C + (m + 5) * ldc, A + (m + 5) * lda, B, N);
        kernel::div_scaler(C + (m + 6) * ldc, A + (m + 6) * lda, B, N);
        kernel::div_scaler(C + (m + 7) * ldc, A + (m + 7) * lda, B, N);
    }
    for (; m < M - 3; m += 4) {
        kernel::div_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
        kernel::div_scaler(C + (m + 1) * ldc, A + (m + 1) * lda, B, N);
        kernel::div_scaler(C + (m + 2) * ldc, A + (m + 2) * lda, B, N);
        kernel::div_scaler(C + (m + 3) * ldc, A + (m + 3) * lda, B, N);
    }
    for(; m<M; ++m) {
        kernel::div_scaler(C + (m + 0) * ldc, A + (m + 0) * lda, B, N);
    }
}

void fma(int M, int N, const float* A, int lda, const float* B, int ldb, const float* C, int ldc, float* D, int ldd) {
    for (int m = 0; m < M; ++m) {
        kernel::fma(D, A, B, C, N);
    }
}

}  // namespace cpu
}  // namespace fastnum