#pragma once

#include <random>
#include <ctime>
#include <cmath>

template <class _Tp>
inline void random_matrix(_Tp* data, int rows, int cols) {
    std::default_random_engine e;
    std::uniform_real_distribution<_Tp> u(0,1);
    for(int i=0;i<rows*cols;++i) {
        data[i] = u(e)/5.0;
    }
}

template <class _Tp>
inline bool check_matrix(_Tp* A, _Tp* B, int rows, int cols) {
    bool is_same = true;
    for(int r=0; r < rows; ++r) {
        for(int c=0; c < cols; ++c) {
            int offset = r*cols+c;
            if(std::fabs(A[offset]-B[offset]) > 5e-4) {
                printf("offset=%d A=%f, B=%f\n", offset, A[offset], B[offset]);
                return false;
            }
        }
    }
    return is_same;
}

inline void print_matrix(float* data, int rows, int cols) {
    for(int r=0; r<rows; ++r) {
        for(int c=0; c<cols; ++c) {
            printf("%.3f ", data[r*cols+c]);
        }
        printf("\n");
    }
}