#include "arithmetic.hpp"
#include "cpu/x86/arithmetic_x86.hpp"

namespace fastnum {
namespace cpu {

void add(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::add(C, A, B, N);
    }
}

void sub(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::sub(C, A, B, N);
    }
}

void mul(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::mul(C, A, B, N);
    }
}

void div(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::div(C, A, B, N);
    }
}

void add_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::add_scaler(C, A, B, N);
    }
}

void sub_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::sub_scaler(C, A, B, N);
    }
}

void mul_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::mul_scaler(C, A, B, N);
    }
}

void div_scaler(int M, int N, const float* A, int lda, const float B, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::div_scaler(C, A, B, N);
    }
}

void fma(int M, int N, const float* A, int lda, const float* B, int ldb, const float* C, int ldc, float* D, int ldd) {
    for (int m = 0; m < M; ++m) {
        kernel::fma(D, A, B, C, N);
    }
}

}  // namespace cpu
}  // namespace fastnum