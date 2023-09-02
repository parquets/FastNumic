#include "eltwise.hpp"
#include "cpu/x86/eltwise_x86.hpp"

namespace fastnum {
namespace cpu {

void exp2(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::exp2(B, A, N);
    }
}

void exp10(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::exp10(B, A, N);
    }
}

void log2(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::log2(B, A, N);
    }
}

void log10(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::log10(B, A, N);
    }
}

void relu(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::relu(B, A, N);
    }
}

void sigmoid(int M, int N, const float* A, int lda, float* B, int ldb) {
    for (int m = 0; m < M; ++m) {
        kernel::sigmoid(B, A, N);
    }
}

}  // namespace cpu
}  // namespace fastnum