#pragma once

#include <math.h>

namespace fastnum {
namespace cpu {

void add(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void sub(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void mul(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);
void div(int M, int N, const float *A, int lda, const float *B, int ldb, float *C, int ldc);

void add_scaler(int M, int N, const float *A, int lda, const float B, float *C, int ldc);
void sub_scaler(int M, int N, const float *A, int lda, const float B, float *C, int ldc);
void mul_scaler(int M, int N, const float *A, int lda, const float B, float *C, int ldc);
void div_scaler(int M, int N, const float *A, int lda, const float B, float *C, int ldc);

void fma(int M, int N, const float *A, int lda, const float *B, int ldb, const float *C, int ldc, float *D, int ldd);

}  // namespace cpu
}  // namespace fastnum