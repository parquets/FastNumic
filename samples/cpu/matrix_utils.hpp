#pragma once

#include <random>
#include <ctime>
#include <cmath>

inline void random_matrix(float* data, int rows, int cols) {
    std::default_random_engine e;
    std::uniform_real_distribution<float> u(0,2);
    for(int i=0;i<rows*cols;++i) {
        data[i] = u(e);
    }
}

inline bool check_matrix(float* A, float* B, int rows, int cols) {
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
