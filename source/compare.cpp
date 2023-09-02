#include "compare.hpp"
#include "cpu/x86/compare_x86.hpp"

namespace fastnum {
namespace cpu {

void compare_min(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::compare_min(C, A, B, N);
        A += lda;
        B += ldb;
        C += ldc;
    }
}

void compare_max(int M, int N, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        kernel::compare_max(C, A, B, N);
        A += lda;
        B += ldb;
        C += ldc;
    }
}

}  // namespace cpu
}  // namespace fastnum