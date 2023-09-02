#pragma once

#include <math.h>

namespace fastnum {
namespace cpu {

void exp2(int M, int N, const float* A, int lda, float* B, int ldb);
void exp10(int M, int N, const float* A, int lda, float* B, int ldb);

void log2(int M, int N, const float* A, int lda, float* B, int ldb);
void log10(int M, int N, const float* A, int lda, float* B, int ldb);

void relu(int M, int N, const float* A, int lda, float* B, int ldb);
void sigmoid(int M, int N, const float* A, int lda, float* B, int ldb);

}
}