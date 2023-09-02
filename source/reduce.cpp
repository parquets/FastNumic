#include "reduce.hpp"
#include "cpu/x86/reduce_x86.hpp"
#include "transpose.hpp"

namespace fastnum {
namespace cpu {

static void reduce_add_row(int M, int N, const float* A, int lda, float* C) {
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_add(A, N);
        A += lda;
        C += 1;
    }
}

static void reduce_add_col(int M, int N, const float* A, int lda, float* C) {
    float* tmp = (float*)malloc(sizeof(float) * N * M);
    transpose(M, N, A, lda, tmp, M);
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_add(tmp, M);
    }
    free(tmp);
}

static void reduce_max_row(int M, int N, const float* A, int lda, float* C) {
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_max(A, N);
        A += lda;
        C += 1;
    }
}

static void reduce_max_col(int M, int N, const float* A, int lda, float* C) {
    float* tmp = (float*)malloc(sizeof(float) * N * M);
    transpose(M, N, A, lda, tmp, M);
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_max(tmp, M);
    }
    free(tmp);
}

static void reduce_min_row(int M, int N, const float* A, int lda, float* C) {
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_min(A, N);
        A += lda;
        C += 1;
    }
}

static void reduce_min_col(int M, int N, const float* A, int lda, float* C) {
    float* tmp = (float*)malloc(sizeof(float) * N * M);
    transpose(M, N, A, lda, tmp, M);
    for (int i = 0; i < M; ++i) {
        *C = kernel::reduce_min(tmp, M);
    }
    free(tmp);
}

void reduce_add(int M, int N, const float* A, int lda, float* C, int axis = -1) {
    if (axis == 0) {
        reduce_add_row(M, N, A, lda, C);
    } else if (axis == 1) {
        reduce_add_col(M, N, A, lda, C);
    } else {
        float tmp = 0;
        for (int i = 0; i < M; ++i) {
            tmp += kernel::reduce_add(A, N);
            A += lda;
            C += 1;
        }
        *C = tmp;
    }
}

void reduce_max(int M, int N, const float* A, int lda, float* C, int axis = -1) {
    if (axis == 0) {
        reduce_max_row(M, N, A, lda, C);
    } else if (axis == 1) {
        reduce_max_col(M, N, A, lda, C);
    } else {
        float tmp = 99999999;
        for (int i = 0; i < M; ++i) {
            tmp = std::max(kernel::reduce_max(A, N), tmp);
            A += lda;
            C += 1;
        }
        *C = tmp;
    }
}

void reduce_min(int M, int N, const float* A, int lda, float* C, int axis = -1) {
    if (axis == 0) {
        reduce_min_row(M, N, A, lda, C);
    } else if (axis == 1) {
        reduce_min_col(M, N, A, lda, C);
    } else {
        float tmp = -99999999;
        for (int i = 0; i < M; ++i) {
            tmp = std::min(kernel::reduce_min(A, N), tmp);
            A += lda;
            C += 1;
        }
        *C = tmp;
    }
}

}  // namespace cpu
}  // namespace fastnum