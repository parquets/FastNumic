#pragma once

template <class _Tp>
void transpose_naive(const _Tp* A, int a_rows, int a_cols, int lda, _Tp* B, int ldb) {
    for (int c = 0; c < a_cols; ++c) {
        for (int r = 0; r < a_rows; ++r) {
            B[c * ldb + r] = A[r * lda + c];
        }
    }
}
