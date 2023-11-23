#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "stranspose_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

static void weightTransLeftUnit4(float* dest, const float* source) {
    dest[0] = source[0];
    dest[1] = source[1];
    dest[2] = source[2];
    dest[3] = (source[0] + source[3] + source[6]) / 2;
    dest[4] = (source[1] + source[4] + source[7]) / 2;
    dest[5] = (source[2] + source[5] + source[8]) / 2;
    dest[6] = (source[0] - source[3] + source[6]) / 2;
    dest[7] = (source[1] - source[4] + source[7]) / 2;
    dest[8] = (source[2] - source[5] + source[8]) / 2;
    dest[9] = source[6];
    dest[10] = source[7];
    dest[11] = source[8];
}

static void weightTransRightUnit4(float* dest, const float* source) {
    dest[0] = source[0];
    dest[1] = (source[0] + source[1] + source[2]) / 2;
    dest[2] = (source[0] - source[1] + source[2]) / 2;
    dest[3] = source[2];

    dest[4] = source[3];
    dest[5] = (source[3] + source[4] + source[5]) / 2;
    dest[6] = (source[3] - source[4] + source[5]) / 2;
    dest[7] = source[5];

    dest[8] = source[6];
    dest[9] = (source[6] + source[7] + source[8]) / 2;
    dest[10] = (source[6] - source[7] + source[8]) / 2;
    dest[11] = source[8];

    dest[12] = source[9];
    dest[13] = (source[9] + source[10] + source[11]) / 2;
    dest[14] = (source[9] - source[10] + source[11]) / 2;
    dest[15] = source[11];
}





static void weightTransLeftUnit8(float* dest, const float* source)  {
    /*
        Gg -> [8, 3]
        g  -> [3, 3]
        */
        const float c0 = static_cast<float>(2.0 / 9);
        const float c1 = static_cast<float>(1.0 / 90);
        const float c2 = static_cast<float>(1.0 / 45);
        const float c3 = static_cast<float>(2.0 / 45);
        const float c4 = static_cast<float>(1.0 / 180);

        dest[0 * 3 + 0] = source[0];
        dest[0 * 3 + 1] = source[1];
        dest[0 * 3 + 2] = source[2];

        dest[1 * 3 + 0] = -c0 * (source[0] + source[3] + source[6]);
        dest[1 * 3 + 1] = -c0 * (source[1] + source[4] + source[7]);
        dest[1 * 3 + 2] = -c0 * (source[2] + source[5] + source[8]);

        dest[2 * 3 + 0] = -c0 * (source[0] - source[3] + source[6]);
        dest[2 * 3 + 1] = -c0 * (source[1] - source[4] + source[7]);
        dest[2 * 3 + 2] = -c0 * (source[2] - source[5] + source[8]);

        dest[3 * 3 + 0] = c1 * source[0] + c2 * source[3] + c3 * source[6];
        dest[3 * 3 + 1] = c1 * source[1] + c2 * source[4] + c3 * source[7];
        dest[3 * 3 + 2] = c1 * source[2] + c2 * source[5] + c3 * source[8];

        dest[4 * 3 + 0] = c1 * source[0] - c2 * source[3] + c3 * source[6];
        dest[4 * 3 + 1] = c1 * source[1] - c2 * source[4] + c3 * source[7];
        dest[4 * 3 + 2] = c1 * source[2] - c2 * source[5] + c3 * source[8];

        dest[5 * 3 + 0] = c2 * source[0] + c1 * source[3] + c4 * source[6];
        dest[5 * 3 + 1] = c2 * source[1] + c1 * source[4] + c4 * source[7];
        dest[5 * 3 + 2] = c2 * source[2] + c1 * source[5] + c4 * source[8];

        dest[6 * 3 + 0] = c2 * source[0] - c1 * source[3] + c4 * source[6];
        dest[6 * 3 + 1] = c2 * source[1] - c1 * source[4] + c4 * source[7];
        dest[6 * 3 + 2] = c2 * source[2] - c1 * source[5] + c4 * source[8];

        dest[7 * 3 + 0] = source[6];
        dest[7 * 3 + 1] = source[7];
        dest[7 * 3 + 2] = source[8];
}

static void weightTransRightUnit8(float* dest, const float* source) {
    const float c0 = static_cast<float>(2.0 / 9);
    const float c1 = static_cast<float>(1.0 / 90);
    const float c2 = static_cast<float>(1.0 / 45);
    const float c3 = static_cast<float>(2.0 / 45);
    const float c4 = static_cast<float>(1.0 / 180);

    for (int i = 0; i < 8; ++i) {
        int i0 = i * 3;
        dest[i * 8 + 0] = source[i0];
        dest[i * 8 + 1] = -c0 * (source[i0] + source[i0 + 1] + source[i0 + 2]);
        dest[i * 8 + 2] = -c0 * (source[i0] - source[i0 + 1] + source[i0 + 2]);
        dest[i * 8 + 3] = c1 * source[i0] + c2 * source[i0 + 1] + c3 * source[i0 + 2];
        dest[i * 8 + 4] = c1 * source[i0] - c2 * source[i0 + 1] + c3 * source[i0 + 2];
        dest[i * 8 + 5] = c2 * source[i0] + c1 * source[i0 + 1] + c4 * source[i0 + 2];
        dest[i * 8 + 6] = c2 * source[i0] - c1 * source[i0 + 1] + c4 * source[i0 + 2];
        dest[i * 8 + 7] = source[i0 + 2];
    }
}


// G = [
    //     [1.0f, 0.0f, 0.0f],
    //     [1.0f / 2, 1.0f / 2, 1.0f / 2],
    //     [1.0f / 2, -1.0f / 2, 1.0f / 2],
    //     [0.0f, 0.0f, 1.0f]
    // ]
inline void winogradWeightTransUnit4K3S1Pack8(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[8][4][3] = {0};

    for (int i = 0; i < 8; ++i) {
        weightTransLeftUnit4(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[8][4][4] = {0};

    for (int i = 0; i < 8; ++i) {
        weightTransRightUnit4(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 8; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}

inline void winogradWeightTransUnit4K3S1Pack4(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[4][4][3] = {0};

    for (int i = 0; i < 4; ++i) {
        weightTransLeftUnit4(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[4][4][4] = {0};

    for (int i = 0; i < 4; ++i) {
        weightTransRightUnit4(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}

inline void winogradWeightTransUnit4K3S1Pack1(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[1][4][3] = {0};

    for (int i = 0; i < 1; ++i) {
        weightTransLeftUnit4(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[1][4][4] = {0};

    for (int i = 0; i < 1; ++i) {
        weightTransRightUnit4(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 1; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}


/*
    G = np.array([
        [1.0,      0.0,       0.0],
        [-2.0 / 9, -2.0 / 9,  -2.0 / 9],
        [-2.0 / 9, 2.0 / 9,  -2.0 / 9],
        [1.0 / 90, 1.0 / 45,  2.0 / 45],
        [1.0 / 90, -1.0 / 45, 2.0 / 45],
        [1.0 / 45, 1.0 / 90,  1.0 / 180],
        [1.0 / 45, -1.0 / 90, 1.0 / 180],
        [0.0,      0.0,       1.0]
    ])
    */
    /* u = G  g GT*/
    /*
    G -> [8, 3]
    g -> [3, 3]

   */
inline void winogradWeightTransUnit8K3S1Pack8(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[8][8][3] = {0};

    for (int i = 0; i < 8; ++i) {
        weightTransLeftUnit8(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[8][8][8] = {0};

    for (int i = 0; i < 8; ++i) {
        weightTransRightUnit8(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}

inline void winogradWeightTransUnit8K3S1Pack4(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[4][8][3] = {0};

    for (int i = 0; i < 4; ++i) {
        weightTransLeftUnit8(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[4][8][8] = {0};

    for (int i = 0; i < 4; ++i) {
        weightTransRightUnit8(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 4; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}

inline void winogradWeightTransUnit8K3S1Pack1(float* dest, const float* source) {
    int kernel_stride = 3 * 3;

    float Gg[1][8][3] = {0};

    for (int i = 0; i < 1; ++i) {
        weightTransLeftUnit8(&Gg[i][0][0], source + i * kernel_stride);
    }

    float GgGt[1][8][8] = {0};

    for (int i = 0; i < 1; ++i) {
        weightTransRightUnit8(&GgGt[i][0][0], &Gg[i][0][0]);
    }

    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 1; ++k) {
                *dest++ = GgGt[k][i][j];
            }
        }
    }
}



}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum