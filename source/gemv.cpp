#include "gemv.hpp"
#include <math.h>
#include <memory.h>
#include <stdlib.h>

#include "cpu/x86/vma_x86.hpp"

namespace fastnum {
namespace cpu {

void sgemv_n(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C) {
    int m = 0;
    for (m = 0; m < M - 5; m += 6) {
        kernel::vma6(N, A, lda, B, C);
        A += 6*lda;
        C += 6;    
    }
    for (; m < M-3; m+=4) {
        kernel::vma4(N, A, lda, B, C);
        A += 4*lda;
        C += 4; 
    }
    for (; m < M; ++m) {
        kernel::vma1(N, A, lda, B, C);
        A += lda;
        C += 1; 
    }
}

void sgemv_t(int M, int N, float alpha, const float* A, int lda, const float* B, float beta, float* C) {
    
    
}

}  // namespace cpu
}  // namespace fastnum