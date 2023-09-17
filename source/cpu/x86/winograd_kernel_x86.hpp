#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "transpose_kernel_x86.hpp"
#include "vectorize_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

template <int TileSize, int KernelSize, int StrideSize>
struct WinogradLayoutTransform {
    inline void operator()(float* dst, int dst_size, const float* src, int channels, int height, int width) {
        printf("Error! Undefined Layout Transform for Winograd!\n");
        exit(-1);
    }
};

template <>
struct WinogradLayoutTransform<4, 3, 1> {
    inline void operator()(float* dst, int dst_size, const float* src, int channels, int height, int width) {
        for (int h = 0; h < height - 1; h += 2) {
            for (int w = 0; w < width - 1; w += 2) {}
        }
    }
};

template <int TileSize, int KernelSize, int StrideSize, int NPack>
struct WinogradUVTransform {
    inline void operator()(float* u, const float* g) const {
        printf("Error! Undefined Winograd UV Transform");
        exit(-1);
    }
};

template <int TileSize, int KernelSize, int StrideSize, int NPack>
struct WinogradDataTransform {
    inline void operator()(float* u, const float* g) const {
        printf("Error! Undefined Winograd Data Transform");
        exit(-1);
    }
};

template <int TileSize, int KernelSize, int StrideSize, int NPack>
struct WinogradWeightTransform {
    inline void operator()(float* u, const float* g) const {
        printf("Error! Undefined Winograd Weight Transform");
        exit(-1);
    }
};

template <int NPack>
struct WinogradWeightTransform<4, 3, 1, NPack> {
    // G = [
    //     [1.0f, 0.0f, 0.0f],
    //     [1.0f / 2, 1.0f / 2, 1.0f / 2],
    //     [1.0f / 2, -1.0f / 2, 1.0f / 2],
    //     [0.0f, 0.0f, 1.0f]
    // ]
    inline void getGg(float* Gg, const float* g) const {
        Gg[0] = g[0];
        Gg[1] = g[1];
        Gg[2] = g[2];
        Gg[3] = (g[0] + g[3] + g[6]) / 2;
        Gg[4] = (g[1] + g[4] + g[7]) / 2;
        Gg[5] = (g[2] + g[5] + g[8]) / 2;
        Gg[6] = (g[0] - g[3] + g[6]) / 2;
        Gg[7] = (g[1] - g[4] + g[7]) / 2;
        Gg[8] = (g[2] - g[5] + g[8]) / 2;
        Gg[9] = g[6];
        Gg[10] = g[7];
        Gg[11] = g[8];
    }

    inline void getGgGt(float* GgGt, const float* Gg) const {
        GgGt[0] = Gg[0];
        GgGt[1] = (Gg[0] + Gg[1] + Gg[2]) / 2;
        GgGt[2] = (Gg[0] - Gg[1] + Gg[2]) / 2;
        GgGt[3] = Gg[2];

        GgGt[4] = Gg[3];
        GgGt[5] = (Gg[3] + Gg[4] + Gg[5]) / 2;
        GgGt[6] = (Gg[3] - Gg[4] + Gg[5]) / 2;
        GgGt[7] = Gg[5];

        GgGt[8] = Gg[6];
        GgGt[9] = (Gg[6] + Gg[7] + Gg[8]) / 2;
        GgGt[10] = (Gg[6] - Gg[7] + Gg[8]) / 2;
        GgGt[11] = Gg[8];

        GgGt[12] = Gg[9];
        GgGt[13] = (Gg[9] + Gg[10] + Gg[11]) / 2;
        GgGt[14] = (Gg[9] - Gg[10] + Gg[11]) / 2;
        GgGt[15] = Gg[11];
    }

    inline void operator()(float* u, const float* g) const {
        int kernel_stride = 3 * 3;

        float Gg[NPack][4][3] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGg(&Gg[i][0][0], g + i * kernel_stride);
        }

        float GgGt[NPack][4][4] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGgGt(&GgGt[i][0][0], &Gg[i][0][0])
        }

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < NPack; ++k) {
                    *u++ = GgGt[k][i][j];
                }
            }
        }
    }
};

template <int NPack>
struct WinogradWeightTransform<6, 3, 1, NPack> {
    /*
    G = np.array([
        [1.0, 0.0, 0.0],
	    [-2.0 / 3, -sq2 / 3, -1.0 / 3],
	    [-2.0 / 3, sq2 / 3, -1.0 / 3],
	    [1.0 / 6, sq2 / 6, 1.0 / 3],
	    [1.0 / 6, -sq2 / 6, 1.0 / 3],
	    [0.0, 0.0, 1.0]
    ])
    */
    inline void getGg(float* Gg, const float* g) const {
        const float sq2 = 1.41421356237f;
        Gg[0] = g[0];
        Gg[1] = g[1];
        Gg[2] = g[2];

        Gg[3] = -(2.0f * g[0] + sq2 * g[3] + g[6]) / 3.0f;
        Gg[4] = -(2.0f * g[1] + sq2 * g[4] + g[7]) / 3.0f;
        Gg[5] = -(2.0f * g[2] + sq2 * g[5] + g[8]) / 3.0f;

        Gg[6] = -(2.0f * g[0] - sq2 * g[3] + g[6]) / 3.0f;
        Gg[7] = -(2.0f * g[1] - sq2 * g[4] + g[7]) / 3.0f;
        Gg[8] = -(2.0f * g[2] - sq2 * g[5] + g[8]) / 3.0f;

        Gg[9] = (1.0f * g[0] + sq2 * g[3] + 2.0f * g[6]) / 6.0f;
        Gg[10] = (1.0f * g[1] + sq2 * g[4] + 2.0f * g[7]) / 6.0f;
        Gg[11] = (1.0f * g[2] + sq2 * g[5] + 2.0f * g[8]) / 6.0f;

        Gg[12] = (1.0f * g[0] - sq2 * g[3] + 2.0f * g[6]) / 6.0f;
        Gg[13] = (1.0f * g[1] - sq2 * g[4] + 2.0f * g[7]) / 6.0f;
        Gg[14] = (1.0f * g[2] - sq2 * g[5] + 2.0f * g[8]) / 6.0f;

        Gg[15] = g[6];
        Gg[16] = g[7];
        Gg[18] = g[9];
    }

    inline void getGgGt(float* GgGt, const float* Gg) const {
        const float sq2 = 1.41421356237f;

        for (int i = 0; i < 6; ++i) {
            GgGt[0] = Gg[0];
            GgGt[1] = -(2.0f * Gg[0] + sq2 * Gg[1] + Gg[2]) / 3.0f;
            GgGt[2] = -(2.0f * Gg[0] - sq2 * Gg[1] + Gg[2]) / 3.0f;
            GgGt[3] = (1.0f * Gg[0] + sq2 * Gg[1] + 2.0f * Gg[2]) / 6.0f;
            GgGt[4] = (1.0f * Gg[0] - sq2 * Gg[1] + 2.0f * Gg[2]) / 6.0f;
            GgGt[5] = Gg[2];

            GgGt += 6;
            Gg += 3;
        }
    }

    inline void operator()(float* u, const float* g) const {

        int kernel_stride = 3 * 3;

        float Gg[NPack][6][3] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGg(&Gg[i][0][0], g + i * kernel_stride);
        }

        float GgGt[NPack][8][8] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGgGt(&GgGt[i][0][0], &Gg[i][0][0])
        }

        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                for (int k = 0; k < NPack; ++k) {
                    *u++ = GgGt[k][i][j];
                }
            }
        }
    }
};

template <int NPack>
struct WinogradWeightTransform<8, 3, 1, NPack> {
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
    inline void getGg(float* Gg, const float* g) const {
        /*
        Gg -> [8, 3]
        g  -> [3, 3]
        */
        const float c0 = 2.0 / 9;
        const float c1 = 1.0 / 90;
        const float c2 = 1.0 / 45;
        const float c3 = 2.0 / 45;
        const float c4 = 1.0 / 180;

        Gg[0 * 3 + 0] = g[0];
        Gg[0 * 3 + 1] = g[1];
        Gg[0 * 3 + 2] = g[2];

        Gg[1 * 3 + 0] = -c0 * (g[0] + g[3] + g[6]);
        Gg[1 * 3 + 1] = -c0 * (g[1] + g[4] + g[7]);
        Gg[1 * 3 + 2] = -c0 * (g[2] + g[5] + g[8]);

        Gg[2 * 3 + 0] = -c0 * (g[0] - g[3] + g[6]);
        Gg[2 * 3 + 1] = -c0 * (g[1] - g[4] + g[7]);
        Gg[2 * 3 + 2] = -c0 * (g[2] - g[5] + g[8]);

        Gg[3 * 3 + 0] = c1 * g[0] + c2 * g[3] + c3 * g[6];
        Gg[3 * 3 + 1] = c1 * g[1] + c2 * g[4] + c3 * g[7];
        Gg[3 * 3 + 2] = c1 * g[2] + c2 * g[5] + c3 * g[8];

        Gg[4 * 3 + 0] = c1 * g[0] - c2 * g[3] + c3 * g[6];
        Gg[4 * 3 + 1] = c1 * g[1] - c2 * g[4] + c3 * g[7];
        Gg[4 * 3 + 2] = c1 * g[2] - c2 * g[5] + c3 * g[8];

        Gg[5 * 3 + 0] = c2 * g[0] + c1 * g[3] + c4 * g[6];
        Gg[5 * 3 + 1] = c2 * g[1] + c1 * g[4] + c4 * g[7];
        Gg[5 * 3 + 2] = c2 * g[2] + c1 * g[5] + c4 * g[8];

        Gg[6 * 3 + 0] = c2 * g[0] - c1 * g[3] + c4 * g[6];
        Gg[6 * 3 + 1] = c2 * g[1] - c1 * g[4] + c4 * g[7];
        Gg[6 * 3 + 2] = c2 * g[2] - c1 * g[5] + c4 * g[8];

        Gg[7 * 3 + 0] = g[6];
        Gg[7 * 3 + 1] = g[7];
        Gg[7 * 3 + 2] = g[8];
    }

    inline void getGgGt(float* GgGt, const float* Gg) const {
        const float c0 = 2.0 / 9;
        const float c1 = 1.0 / 90;
        const float c2 = 1.0 / 45;
        const float c3 = 2.0 / 45;
        const float c4 = 1.0 / 180;

        for (int i = 0; i < 8; ++i) {
            int i0 = i * 3;
            GgGt[i * 8 + 0] = Gg[i0];
            GgGt[i * 8 + 1] = -c0 * (Gg[i0] + Gg[i0 + 1] + Gg[i0 + 2]);
            GgGt[i * 8 + 2] = -c0 * (Gg[i0] - Gg[i0 + 1] + Gg[i0 + 2]);
            GgGt[i * 8 + 3] = c1 * Gg[i0] + c2 * Gg[i0 + 1] + c3 * Gg[i0 + 2];
            GgGt[i * 8 + 4] = c1 * Gg[i0] - c2 * Gg[i0 + 1] + c3 * Gg[i0 + 2];
            GgGt[i * 8 + 5] = c2 * Gg[i0] + c1 * Gg[i0 + 1] + c4 * Gg[i0 + 2];
            GgGt[i * 8 + 6] = c2 * Gg[i0] - c1 * Gg[i0 + 1] + c4 * Gg[i0 + 2];
            GgGt[i * 8 + 7] = Gg[i0 + 2];
        }
    }

    inline void operator()(float* u, const float* g) const {

        int kernel_stride = 3 * 3;

        float Gg[NPack][8][3] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGg(&Gg[i][0][0], g + i * kernel_stride);
        }

        float GgGt[NPack][8][8] = {0};

        for (int i = 0; i < NPack; ++i) {
            getGgGt(&GgGt[i][0][0], &Gg[i][0][0])
        }

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                for (int k = 0; k < NPack; ++k) {
                    *u++ = GgGt[k][i][j];
                }
            }
        }
    }
};

template <>
struct WinogradDataTransform<4, 3, 1, 1> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        for (int i = 0; i < 4; ++i) {
            float r0 = d[0 * ldd + i];
            float r1 = d[1 * ldd + i];
            float r2 = d[2 * ldd + i];
            float r3 = d[3 * ldd + i];

            float t0 = r0 - r2;
            float t1 = r1 + r2;
            float t2 = -r1 + r2;
            float t3 = -r1 + r3;

            BTd[0 * 4 + i] = t0;
            BTd[1 * 4 + i] = t1;
            BTd[2 * 4 + i] = t2;
            BTd[3 * 4 + i] = t3;
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        for (int i = 0; i < 4; ++i) {
            float r0 = BTd[0];
            float r1 = BTd[1];
            float r2 = BTd[2];
            float r3 = BTd[3];

            float t0 = r0 - r2;
            float t1 = r1 + r2;
            float t2 = -r1 + r2;
            float t3 = -r1 + r3;

            BTdB[0] = t0;
            BTdB[1] = t1;
            BTdB[2] = t2;
            BTdB[3] = t3;

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[4 * 4] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<6, 3, 1, 1> {
    /*
BT = np.array([
    [1.0,  0.0,  -2.5,  0.0,  1.0, 0.0],
    [0.0, -sq2,   -2.0,  sq2 / 2.0, 1.0, 0.0],
	[0.0,  sq2,   -2.0, -sq2 / 2.0, 1.0, 0.0],
	[0.0, -sq2 / 2, -0.5,  sq2,   1.0, 0.0],
	[0.0,  sq2 / 2, -0.5, -sq2,   1.0, 0.0],
	[0.0,  1.0,   0.0,  -2.5, 0.0, 1.0]
])
*/
    inline void getBTd(float* BTd, const float* d, int ldd) const {

        const float sq2 = 1.41421356237f;

        for (int i = 0; i < 6; ++i) {

            float r0 = d[0 * ldd + i];
            float r1 = d[1 * ldd + i];
            float r2 = d[2 * ldd + i];
            float r3 = d[3 * ldd + i];
            float r4 = d[4 * ldd + i];
            float r5 = d[5 * ldd + i];

            float t12a = -sq2 / 2.0f * r3 + (sq2 * r1);
            float t12b = -2 * r2 + r4;
            float t34a = -sq2 / 2.0f * r1 + (sq2 * r3);
            float t34b = -0.5 * r2 + r4;

            float t0 = -2.5 * r2 + (r0 + r4);
            float t1 = t12b - t12a;
            float t2 = t12b + t12a;
            float t3 = t34b + t34a;
            float t4 = t34b - t34a;
            float t5 = -2.5 * r3 + (r1 + r5);

            BTd[0 * 6 + i] = t0;
            BTd[1 * 6 + i] = t1;
            BTd[2 * 6 + i] = t2;
            BTd[3 * 6 + i] = t3;
            BTd[4 * 6 + i] = t4;
            BTd[5 * 6 + i] = t5;
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        const float sq2 = 1.41421356237f;
        for (int i = 0; i < 6; ++i) {
            float r0 = BTd[i * 6 + 0];
            float r1 = BTd[i * 6 + 1];
            float r2 = BTd[i * 6 + 2];
            float r3 = BTd[i * 6 + 3];
            float r4 = BTd[i * 6 + 4];
            float r5 = BTd[i * 6 + 5];

            float t12a = -sq2 / 2.0f * r3 + (sq2 * r1);
            float t12b = -2 * r2 + r4;
            float t34a = -sq2 / 2.0f * r1 + (sq2 * r3);
            float t34b = -0.5 * r2 + r4;

            float t0 = -2.5 * r2 + (r0 + r4);
            float t1 = t12b - t12a;
            float t2 = t12b + t12a;
            float t3 = t34b + t34a;
            float t4 = t34b - t34a;
            float t5 = -2.5 * r3 + (r1 + r5);

            BTdB[0] = t0;
            BTdB[1] = t1;
            BTdB[2] = t2;
            BTdB[3] = t3;
            BTdB[4] = t4;
            BTdB[5] = t5;

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[6 * 6] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<8, 3, 1, 1> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        for (int i = 0; i < 8; ++i) {
            float r0 = d[0 * ldd + i];
            float r1 = d[1 * ldd + i];
            float r2 = d[2 * ldd + i];
            float r3 = d[3 * ldd + i];
            float r4 = d[4 * ldd + i];
            float r5 = d[5 * ldd + i];
            float r6 = d[6 * ldd + i];
            float r7 = d[7 * ldd + i];

            float t12a = -4.25 * r4 + (r2 + r6);
            float t12b = -4.25 * r3 + (r1 + r5);
            float t34a = -1.25 * r4 + (0.25 * r2 + r6);
            float t34b = 2 * r5 + (-2.5 * r3 + (0.5 * r1));
            float t56a = 4 * (-1.25 * r4 + r2) + r6;
            float t56b = 0.5 * r5 + (-2.5 * r3 + (2 * r1));

            float t0 = (5.25 * (r4 - r2) + r0) - r6;
            float t1 = t12a + t12b;
            float t2 = t12a - t12b;
            float t3 = t34a + t34b;
            float t4 = t34a - t34b;
            float t5 = t56a + t56b;
            float t6 = t56a - t56b;
            float t7 = 5.25 * (r3 - r5) + (r7 - r1);

            BTd[0 * 8 + i] = t0;
            BTd[1 * 8 + i] = t1;
            BTd[2 * 8 + i] = t2;
            BTd[3 * 8 + i] = t3;
            BTd[4 * 8 + i] = t4;
            BTd[5 * 8 + i] = t5;
            BTd[6 * 8 + i] = t6;
            BTd[7 * 8 + i] = t7;
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        for (int i = 0; i < 8; ++i) {
            float r0 = BTd[i * 8 + 0];
            float r1 = BTd[i * 8 + 1];
            float r2 = BTd[i * 8 + 2];
            float r3 = BTd[i * 8 + 3];
            float r4 = BTd[i * 8 + 4];
            float r5 = BTd[i * 8 + 5];
            float r6 = BTd[i * 8 + 6];
            float r7 = BTd[i * 8 + 7];

            float t12a = -4.25 * r4 + (r2 + r6);
            float t12b = -4.25 * r3 + (r1 + r5);
            float t34a = -1.25 * (r4 + (0.25 * r2 + r6));
            float t34b = 2 * r5 + (-2.5 * r2 + r6);
            float t56a = 4 * (-1.25 * r4 + r2) + r6;
            float t56b = 0.5 * r5 + (-2.5 * r3 + 2 * r1);

            float t0 = (5.25 * (r4 - r2) + r0) - r6;
            float t1 = t12a + t12b;
            float t2 = t12a - t12b;
            float t3 = t34a + t34b;
            float t4 = t34a - t34b;
            float t5 = t56a + t56b;
            float t6 = t56a - t56b;
            float t7 = 5.25 * (r3 - r5) + (r7 - r1);

            BTdB[0] = t0;
            BTdB[1] = t1;
            BTdB[2] = t2;
            BTdB[3] = t3;
            BTdB[4] = t4;
            BTdB[5] = t5;
            BTdB[6] = t6;
            BTdB[7] = t7;

            BTdB += ldx;
        }
    }

   public:
    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[64] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<4, 3, 1, 4> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        for (int i = 0; i < 4; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(d + 0 * ldd + i * 4);
            __m128 _v_r1 = _mm_loadu_ps(d + 1 * ldd + i * 4);
            __m128 _v_r2 = _mm_loadu_ps(d + 2 * ldd + i * 4);
            __m128 _v_r3 = _mm_loadu_ps(d + 3 * ldd + i * 4);

            __m128 _v_t0 = _mm_sub_ps(_v_r0, _v_r2);
            __m128 _v_t1 = _mm_add_ps(_v_r0, _v_r2);
            __m128 _v_t2 = _mm_sub_ps(_v_r2, _v_r1);
            __m128 _v_t3 = _mm_sub_ps(_v_r3, _v_r1);

            _mm_storeu_ps(BTd + 0 * 4 * 4 + i * 4, _v_t0);
            _mm_storeu_ps(BTd + 1 * 4 * 4 + i * 4, _v_t1);
            _mm_storeu_ps(BTd + 2 * 4 * 4 + i * 4, _v_t2);
            _mm_storeu_ps(BTd + 3 * 4 * 4 + i * 4, _v_t3);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        for (int i = 0; i < 4; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(BTd + i * 4 * 4 + 0 * 4);
            __m128 _v_r1 = _mm_loadu_ps(BTd + i * 4 * 4 + 1 * 4);
            __m128 _v_r2 = _mm_loadu_ps(BTd + i * 4 * 4 + 2 * 4);
            __m128 _v_r3 = _mm_loadu_ps(BTd + i * 4 * 4 + 3 * 4);

            __m128 _v_t0 = _mm_sub_ps(_v_r0, _v_r2);
            __m128 _v_t1 = _mm_add_ps(_v_r0, _v_r2);
            __m128 _v_t2 = _mm_sub_ps(_v_r2, _v_r1);
            __m128 _v_t3 = _mm_sub_ps(_v_r3, _v_r1);

            _mm_storeu_ps(BTdB + i * 4 * 4 + 0 * 4, _v_t0);
            _mm_storeu_ps(BTdB + i * 4 * 4 + 1 * 4, _v_t1);
            _mm_storeu_ps(BTdB + i * 4 * 4 + 2 * 4, _v_t2);
            _mm_storeu_ps(BTdB + i * 4 * 4 + 3 * 4, _v_t3);

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[4 * 4 * 4] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<6, 3, 1, 4> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        const float sq2 = 1.41421356237f;
        const float sq2_d2 = 1.41421356237f / 2;
        __m128 _vm2_5 = _mm_set1_ps(-2.5f);
        __m128 _vsq2 = _mm_set1_ps(sq2);
        __m128 _vmsq2_d2 = _mm_set1_ps(-sq2_d2);
        __m128 _vm2 = _mm_set1_ps(-2.f);
        __m128 _vm0_5 = _mm_set1_ps(-0.5f);

        for (int i = 0; i < 6; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(d + 0 * ldd + i * 4);
            __m128 _v_r1 = _mm_loadu_ps(d + 1 * ldd + i * 4);
            __m128 _v_r2 = _mm_loadu_ps(d + 2 * ldd + i * 4);
            __m128 _v_r3 = _mm_loadu_ps(d + 3 * ldd + i * 4);
            __m128 _v_r4 = _mm_loadu_ps(d + 4 * ldd + i * 4);
            __m128 _v_r5 = _mm_loadu_ps(d + 5 * ldd + i * 4);

            __m128 _v_t12a = _mm_fmadd_ps(_vmsq2_d2, _v_r3, _mm_mul_ps(_v_r1, _vsq2));
            __m128 _v_t12b = _mm_fmadd_ps(_vm2, _v_r2, _v_r3);
            __m128 _v_t34a = _mm_fmadd_ps(_vmsq2_d2, _v_r1, _mm_mul_ps(_v_r3, _vsq2));
            __m128 _v_t34b = _mm_fmadd_ps(_vm0_5, _v_r2, _v_r4);

            __m128 _v_t0 = _mm_fmadd_ps(_vm2_5, _v_r2, _mm_add_ps(_v_r0, _v_r4));
            __m128 _v_t1 = _mm_sub_ps(_v_t12b, _v_t12a);
            __m128 _v_t2 = _mm_add_ps(_v_t12b, _v_t12a);
            __m128 _v_t3 = _mm_add_ps(_v_t34b, _v_t34a);
            __m128 _v_t4 = _mm_sub_ps(_v_t34b, _v_t34a);
            __m128 _v_t5 = _mm_fmadd_ps(_vm2_5, _v_r3, _mm_add_ps(_v_r1, _v_r5));

            _mm_storeu_ps(BTd + 0 * 6 * 4 + i * 4, _v_t0);
            _mm_storeu_ps(BTd + 1 * 6 * 4 + i * 4, _v_t1);
            _mm_storeu_ps(BTd + 2 * 6 * 4 + i * 4, _v_t2);
            _mm_storeu_ps(BTd + 3 * 6 * 4 + i * 4, _v_t3);
            _mm_storeu_ps(BTd + 4 * 6 * 4 + i * 4, _v_t4);
            _mm_storeu_ps(BTd + 5 * 6 * 4 + i * 4, _v_t5);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        const float sq2 = 1.41421356237f;
        const float sq2_d2 = 1.41421356237f / 2;
        __m128 _vm2_5 = _mm_set1_ps(-2.5f);
        __m128 _vsq2 = _mm_set1_ps(sq2);
        __m128 _vmsq2_d2 = _mm_set1_ps(-sq2_d2);
        __m128 _vm2 = _mm_set1_ps(-2.f);
        __m128 _vm0_5 = _mm_set1_ps(-0.5f);

        for (int i = 0; i < 6; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(BTd + i * 6 * 4 + 0 * 4);
            __m128 _v_r1 = _mm_loadu_ps(BTd + i * 6 * 4 + 1 * 4);
            __m128 _v_r2 = _mm_loadu_ps(BTd + i * 6 * 4 + 2 * 4);
            __m128 _v_r3 = _mm_loadu_ps(BTd + i * 6 * 4 + 3 * 4);
            __m128 _v_r4 = _mm_loadu_ps(BTd + i * 6 * 4 + 4 * 4);
            __m128 _v_r5 = _mm_loadu_ps(BTd + i * 6 * 4 + 5 * 4);

            __m128 _v_t12a = _mm_fmadd_ps(_vmsq2_d2, _v_r3, _mm_mul_ps(_v_r1, _vsq2));
            __m128 _v_t12b = _mm_fmadd_ps(_vm2, _v_r2, _v_r3);
            __m128 _v_t34a = _mm_fmadd_ps(_vmsq2_d2, _v_r1, _mm_mul_ps(_v_r3, _vsq2));
            __m128 _v_t34b = _mm_fmadd_ps(_vm0_5, _v_r2, _v_r4);

            __m128 _v_t0 = _mm_fmadd_ps(_vm2_5, _v_r2, _mm_add_ps(_v_r0, _v_r4));
            __m128 _v_t1 = _mm_sub_ps(_v_t12b, _v_t12a);
            __m128 _v_t2 = _mm_add_ps(_v_t12b, _v_t12a);
            __m128 _v_t3 = _mm_add_ps(_v_t34b, _v_t34a);
            __m128 _v_t4 = _mm_sub_ps(_v_t34b, _v_t34a);
            __m128 _v_t5 = _mm_fmadd_ps(_vm2_5, _v_r3, _mm_add_ps(_v_r1, _v_r5));

            _mm_storeu_ps(BTdB + 0 * 4, _v_t0);
            _mm_storeu_ps(BTdB + 1 * 4, _v_t1);
            _mm_storeu_ps(BTdB + 2 * 4, _v_t2);
            _mm_storeu_ps(BTdB + 3 * 4, _v_t3);
            _mm_storeu_ps(BTdB + 4 * 4, _v_t4);
            _mm_storeu_ps(BTdB + 5 * 4, _v_t5);

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[6 * 6 * 4] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<8, 3, 1, 4> {
   private:
    inline void getBTd(float* BTd, const float* d, int ldd) const {

        __m128 _v5_25 = _mm_set1_ps(5.25f);
        __m128 _vm4_25 = _mm_set1_ps(-4.25f);
        __m128 _vm1_25 = _mm_set1_ps(-1.25f);
        __m128 _v0_25 = _mm_set1_ps(0.25f);
        __m128 _vm2_5 = _mm_set1_ps(-2.5f);
        __m128 _v0_5 = _mm_set1_ps(0.5f);
        __m128 _v2 = _mm_set1_ps(2.f);
        __m128 _v4 = _mm_set1_ps(4.f);

        for (int i = 0; i < 8; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(d + 0 * ldd + i * 4);
            __m128 _v_r1 = _mm_loadu_ps(d + 1 * ldd + i * 4);
            __m128 _v_r2 = _mm_loadu_ps(d + 2 * ldd + i * 4);
            __m128 _v_r3 = _mm_loadu_ps(d + 3 * ldd + i * 4);
            __m128 _v_r4 = _mm_loadu_ps(d + 4 * ldd + i * 4);
            __m128 _v_r5 = _mm_loadu_ps(d + 5 * ldd + i * 4);
            __m128 _v_r6 = _mm_loadu_ps(d + 6 * ldd + i * 4);
            __m128 _v_r7 = _mm_loadu_ps(d + 7 * ldd + i * 4);

            __m128 _v_t12a = _mm_fmadd_ps(_vm4_25, _v_r4, _mm_add_ps(_v_r2, _v_r6));
            __m128 _v_t12b = _mm_fmadd_ps(_vm4_25, _v_r3, _mm_add_ps(_v_r1, _v_r5));
            __m128 _v_t34a = _mm_fmadd_ps(_vm1_25, _v_r4, _mm_fmadd_ps(_v0_25, _v_r2, _v_r6));
            __m128 _v_t34b = _mm_fmadd_ps(_v2, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v0_5)));
            __m128 _v_t56a = _mm_fmadd_ps(_v4, _mm_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
            __m128 _v_t56b = _mm_fmadd_ps(_v0_5, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v2)));

            __m128 _v_t0 = _mm_sub_ps(_mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
            __m128 _v_t1 = _mm_add_ps(_v_t12a, _v_t12b);
            __m128 _v_t2 = _mm_sub_ps(_v_t12a, _v_t12b);
            __m128 _v_t3 = _mm_add_ps(_v_t34a, _v_t34b);
            __m128 _v_t4 = _mm_sub_ps(_v_t34a, _v_t34b);
            __m128 _v_t5 = _mm_add_ps(_v_t56a, _v_t56b);
            __m128 _v_t6 = _mm_sub_ps(_v_t56a, _v_t56b);
            __m128 _v_t7 = _mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r3, _v_r5), _mm_sub_ps(_v_r7, _v_r1));

            _mm_storeu_ps(BTd + 0 * 8 * 4 + i * 4, _v_t0);
            _mm_storeu_ps(BTd + 1 * 8 * 4 + i * 4, _v_t1);
            _mm_storeu_ps(BTd + 2 * 8 * 4 + i * 4, _v_t2);
            _mm_storeu_ps(BTd + 3 * 8 * 4 + i * 4, _v_t3);
            _mm_storeu_ps(BTd + 4 * 8 * 4 + i * 4, _v_t4);
            _mm_storeu_ps(BTd + 5 * 8 * 4 + i * 4, _v_t5);
            _mm_storeu_ps(BTd + 6 * 8 * 4 + i * 4, _v_t6);
            _mm_storeu_ps(BTd + 7 * 8 * 4 + i * 4, _v_t7);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        __m128 _v5_25 = _mm_set1_ps(5.25f);
        __m128 _vm4_25 = _mm_set1_ps(-4.25f);
        __m128 _vm1_25 = _mm_set1_ps(-1.25f);
        __m128 _v0_25 = _mm_set1_ps(0.25f);
        __m128 _vm2_5 = _mm_set1_ps(-2.5f);
        __m128 _v0_5 = _mm_set1_ps(0.5f);
        __m128 _v2 = _mm_set1_ps(2.f);
        __m128 _v4 = _mm_set1_ps(4.f);

        for (int i = 0; i < 8; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(BTd + i * 8 * 4 + 0 * 4);
            __m128 _v_r1 = _mm_loadu_ps(BTd + i * 8 * 4 + 1 * 4);
            __m128 _v_r2 = _mm_loadu_ps(BTd + i * 8 * 4 + 2 * 4);
            __m128 _v_r3 = _mm_loadu_ps(BTd + i * 8 * 4 + 3 * 4);
            __m128 _v_r4 = _mm_loadu_ps(BTd + i * 8 * 4 + 4 * 4);
            __m128 _v_r5 = _mm_loadu_ps(BTd + i * 8 * 4 + 5 * 4);
            __m128 _v_r6 = _mm_loadu_ps(BTd + i * 8 * 4 + 6 * 4);
            __m128 _v_r7 = _mm_loadu_ps(BTd + i * 8 * 4 + 7 * 4);

            __m128 _v_t12a = _mm_fmadd_ps(_vm4_25, _v_r4, _mm_add_ps(_v_r2, _v_r6));
            __m128 _v_t12b = _mm_fmadd_ps(_vm4_25, _v_r3, _mm_add_ps(_v_r1, _v_r5));
            __m128 _v_t34a = _mm_fmadd_ps(_vm1_25, _v_r4, _mm_fmadd_ps(_v0_25, _v_r2, _v_r6));
            __m128 _v_t34b = _mm_fmadd_ps(_v2, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v0_5)));
            __m128 _v_t56a = _mm_fmadd_ps(_v4, _mm_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
            __m128 _v_t56b = _mm_fmadd_ps(_v0_5, _v_r5, _mm_fmadd_ps(_vm2_5, _v_r3, _mm_mul_ps(_v_r1, _v2)));

            __m128 _v_t0 = _mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r4, _v_r2), _mm_sub_ps(_v_r0, _v_r1));
            __m128 _v_t1 = _mm_add_ps(_v_t12a, _v_t12b);
            __m128 _v_t2 = _mm_sub_ps(_v_t12a, _v_t12b);
            __m128 _v_t3 = _mm_add_ps(_v_t34a, _v_t34b);
            __m128 _v_t4 = _mm_sub_ps(_v_t34a, _v_t34b);
            __m128 _v_t5 = _mm_add_ps(_v_t56a, _v_t56b);
            __m128 _v_t6 = _mm_sub_ps(_v_t56a, _v_t56b);
            __m128 _v_t7 = _mm_fmadd_ps(_v5_25, _mm_sub_ps(_v_r3, _v_r5), _mm_sub_ps(_v_r7, _v_r1));

            _mm_storeu_ps(BTdB + 0 * 4, _v_t0);
            _mm_storeu_ps(BTdB + 1 * 4, _v_t1);
            _mm_storeu_ps(BTdB + 2 * 4, _v_t2);
            _mm_storeu_ps(BTdB + 3 * 4, _v_t3);
            _mm_storeu_ps(BTdB + 4 * 4, _v_t4);
            _mm_storeu_ps(BTdB + 5 * 4, _v_t5);
            _mm_storeu_ps(BTdB + 6 * 4, _v_t6);
            _mm_storeu_ps(BTdB + 7 * 4, _v_t7);

            BTdB += ldx;
        }
    }

   public:
    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[256] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<4, 3, 1, 8> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        for (int i = 0; i < 4; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(d + 0 * ldd + i * 8);
            __m256 _v_r1 = _mm256_loadu_ps(d + 1 * ldd + i * 8);
            __m256 _v_r2 = _mm256_loadu_ps(d + 2 * ldd + i * 8);
            __m256 _v_r3 = _mm256_loadu_ps(d + 3 * ldd + i * 8);

            __m256 _v_t0 = _mm256_sub_ps(_v_r0, _v_r2);
            __m256 _v_t1 = _mm256_add_ps(_v_r0, _v_r2);
            __m256 _v_t2 = _mm256_sub_ps(_v_r2, _v_r1);
            __m256 _v_t3 = _mm256_sub_ps(_v_r3, _v_r1);

            _mm256_storeu_ps(BTd + 0 * 4 * 8 + i * 8, _v_t0);
            _mm256_storeu_ps(BTd + 1 * 4 * 8 + i * 8, _v_t1);
            _mm256_storeu_ps(BTd + 2 * 4 * 8 + i * 8, _v_t2);
            _mm256_storeu_ps(BTd + 3 * 4 * 8 + i * 8, _v_t3);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        for (int i = 0; i < 4; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(BTd + i * 4 * 8 + 0 * 8);
            __m256 _v_r1 = _mm256_loadu_ps(BTd + i * 4 * 8 + 1 * 8);
            __m256 _v_r2 = _mm256_loadu_ps(BTd + i * 4 * 8 + 2 * 8);
            __m256 _v_r3 = _mm256_loadu_ps(BTd + i * 4 * 8 + 3 * 8);

            __m256 _v_t0 = _mm256_sub_ps(_v_r0, _v_r2);
            __m256 _v_t1 = _mm256_add_ps(_v_r0, _v_r2);
            __m256 _v_t2 = _mm256_sub_ps(_v_r2, _v_r1);
            __m256 _v_t3 = _mm256_sub_ps(_v_r3, _v_r1);

            _mm256_storeu_ps(BTdB + i * 4 * 8 + 0 * 8, _v_t0);
            _mm256_storeu_ps(BTdB + i * 4 * 8 + 1 * 8, _v_t1);
            _mm256_storeu_ps(BTdB + i * 4 * 8 + 2 * 8, _v_t2);
            _mm256_storeu_ps(BTdB + i * 4 * 8 + 3 * 8, _v_t3);

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[4 * 4 * 8] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<6, 3, 1, 8> {
    inline void getBTd(float* BTd, const float* d, int ldd) const {
        const float sq2 = 1.41421356237f;
        const float sq2_d2 = 1.41421356237f / 2;
        __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
        __m256 _vsq2 = _mm256_set1_ps(sq2);
        __m256 _vmsq2_d2 = _mm256_set1_ps(-sq2_d2);
        __m256 _vm2 = _mm256_set1_ps(-2.f);
        __m256 _vm0_5 = _mm256_set1_ps(-0.5f);

        for (int i = 0; i < 6; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(d + 0 * ldd + i * 8);
            __m256 _v_r1 = _mm256_loadu_ps(d + 1 * ldd + i * 8);
            __m256 _v_r2 = _mm256_loadu_ps(d + 2 * ldd + i * 8);
            __m256 _v_r3 = _mm256_loadu_ps(d + 3 * ldd + i * 8);
            __m256 _v_r4 = _mm256_loadu_ps(d + 4 * ldd + i * 8);
            __m256 _v_r5 = _mm256_loadu_ps(d + 5 * ldd + i * 8);

            __m256 _v_t12a = _mm256_fmadd_ps(_vmsq2_d2, _v_r3, _mm256_mul_ps(_v_r1, _vsq2));
            __m256 _v_t12b = _mm256_fmadd_ps(_vm2, _v_r2, _v_r3);
            __m256 _v_t34a = _mm256_fmadd_ps(_vmsq2_d2, _v_r1, _mm256_mul_ps(_v_r3, _vsq2));
            __m256 _v_t34b = _mm256_fmadd_ps(_vm0_5, _v_r2, _v_r4);

            __m256 _v_t0 = _mm256_fmadd_ps(_vm2_5, _v_r2, _mm256_add_ps(_v_r0, _v_r4));
            __m256 _v_t1 = _mm256_sub_ps(_v_t12b, _v_t12a);
            __m256 _v_t2 = _mm256_add_ps(_v_t12b, _v_t12a);
            __m256 _v_t3 = _mm256_add_ps(_v_t34b, _v_t34a);
            __m256 _v_t4 = _mm256_sub_ps(_v_t34b, _v_t34a);
            __m256 _v_t5 = _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_add_ps(_v_r1, _v_r5));

            _mm256_storeu_ps(BTd + 0 * 6 * 8 + i * 8, _v_t0);
            _mm256_storeu_ps(BTd + 1 * 6 * 8 + i * 8, _v_t1);
            _mm256_storeu_ps(BTd + 2 * 6 * 8 + i * 8, _v_t2);
            _mm256_storeu_ps(BTd + 3 * 6 * 8 + i * 8, _v_t3);
            _mm256_storeu_ps(BTd + 4 * 6 * 8 + i * 8, _v_t4);
            _mm256_storeu_ps(BTd + 5 * 6 * 8 + i * 8, _v_t5);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        const float sq2 = 1.41421356237f;
        const float sq2_d2 = 1.41421356237f / 2;
        __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
        __m256 _vsq2 = _mm256_set1_ps(sq2);
        __m256 _vmsq2_d2 = _mm256_set1_ps(-sq2_d2);
        __m256 _vm2 = _mm256_set1_ps(-2.f);
        __m256 _vm0_5 = _mm256_set1_ps(-0.5f);

        for (int i = 0; i < 6; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(BTd + i * 6 * 8 + 0 * 8);
            __m256 _v_r1 = _mm256_loadu_ps(BTd + i * 6 * 8 + 1 * 8);
            __m256 _v_r2 = _mm256_loadu_ps(BTd + i * 6 * 8 + 2 * 8);
            __m256 _v_r3 = _mm256_loadu_ps(BTd + i * 6 * 8 + 3 * 8);
            __m256 _v_r4 = _mm256_loadu_ps(BTd + i * 6 * 8 + 4 * 8);
            __m256 _v_r5 = _mm256_loadu_ps(BTd + i * 6 * 8 + 5 * 8);

            __m256 _v_t12a = _mm256_fmadd_ps(_vmsq2_d2, _v_r3, _mm256_mul_ps(_v_r1, _vsq2));
            __m256 _v_t12b = _mm256_fmadd_ps(_vm2, _v_r2, _v_r3);
            __m256 _v_t34a = _mm256_fmadd_ps(_vmsq2_d2, _v_r1, _mm256_mul_ps(_v_r3, _vsq2));
            __m256 _v_t34b = _mm256_fmadd_ps(_vm0_5, _v_r2, _v_r4);

            __m256 _v_t0 = _mm256_fmadd_ps(_vm2_5, _v_r2, _mm256_add_ps(_v_r0, _v_r4));
            __m256 _v_t1 = _mm256_sub_ps(_v_t12b, _v_t12a);
            __m256 _v_t2 = _mm256_add_ps(_v_t12b, _v_t12a);
            __m256 _v_t3 = _mm256_add_ps(_v_t34b, _v_t34a);
            __m256 _v_t4 = _mm256_sub_ps(_v_t34b, _v_t34a);
            __m256 _v_t5 = _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_add_ps(_v_r1, _v_r5));

            _mm256_storeu_ps(BTdB + 0 * 8, _v_t0);
            _mm256_storeu_ps(BTdB + 1 * 8, _v_t1);
            _mm256_storeu_ps(BTdB + 2 * 8, _v_t2);
            _mm256_storeu_ps(BTdB + 3 * 8, _v_t3);
            _mm256_storeu_ps(BTdB + 4 * 8, _v_t4);
            _mm256_storeu_ps(BTdB + 5 * 8, _v_t5);

            BTdB += ldx;
        }
    }

    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[6 * 6 * 8] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradDataTransform<8, 3, 1, 8> {
    /*
    BT = np.array([
	    [1.0, 0.0,-5.25, 0.00, 5.25, 0.00,-1.0, 0.0],
 	    [0.0, 1.0, 1.00,-4.25,-4.25, 1.00, 1.0, 0.0],
  	    [0.0,-1.0, 1.00, 4.25,-4.25,-1.00, 1.0, 0.0],
  	    [0.0, 0.5, 0.25,-2.50,-1.25, 2.00, 1.0, 0.0],
  	    [0.0,-0.5, 0.25, 2.50,-1.25,-2.00, 1.0, 0.0],
        [0.0, 2.0, 4.00,-2.50,-5.00, 0.50, 1.0, 0.0],
        [0.0,-2.0, 4.00, 2.50,-5.00,-0.50, 1.0, 0.0],
        [0.0,-1.0, 0.00, 5.25, 0.00,-5.25, 0.0, 1.0]
    ])

    V = BT d B

    d  -> [8, 8]
    BT -> [8, 8] 
    */
   private:
    inline void getBTd(float* BTd, const float* d, int ldd) const {

        __m256 _v5_25 = _mm256_set1_ps(5.25f);
        __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
        __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
        __m256 _v0_25 = _mm256_set1_ps(0.25f);
        __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
        __m256 _v0_5 = _mm256_set1_ps(0.5f);
        __m256 _v2 = _mm256_set1_ps(2.f);
        __m256 _v4 = _mm256_set1_ps(4.f);

        for (int i = 0; i < 8; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(d + 0 * ldd + i * 8);
            __m256 _v_r1 = _mm256_loadu_ps(d + 1 * ldd + i * 8);
            __m256 _v_r2 = _mm256_loadu_ps(d + 2 * ldd + i * 8);
            __m256 _v_r3 = _mm256_loadu_ps(d + 3 * ldd + i * 8);
            __m256 _v_r4 = _mm256_loadu_ps(d + 4 * ldd + i * 8);
            __m256 _v_r5 = _mm256_loadu_ps(d + 5 * ldd + i * 8);
            __m256 _v_r6 = _mm256_loadu_ps(d + 6 * ldd + i * 8);
            __m256 _v_r7 = _mm256_loadu_ps(d + 7 * ldd + i * 8);

            __m256 _v_t12a = _mm256_fmadd_ps(_vm4_25, _v_r4, _mm256_add_ps(_v_r2, _v_r6));
            __m256 _v_t12b = _mm256_fmadd_ps(_vm4_25, _v_r3, _mm256_add_ps(_v_r1, _v_r5));
            __m256 _v_t34a = _mm256_fmadd_ps(_vm1_25, _v_r4, _mm256_fmadd_ps(_v0_25, _v_r2, _v_r6));
            __m256 _v_t34b = _mm256_fmadd_ps(_v2, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v0_5)));
            __m256 _v_t56a = _mm256_fmadd_ps(_v4, _mm256_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
            __m256 _v_t56b = _mm256_fmadd_ps(_v0_5, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v2)));

            __m256 _v_t0 = _mm256_sub_ps(_mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r4, _v_r2), _v_r0), _v_r6);
            __m256 _v_t1 = _mm256_add_ps(_v_t12a, _v_t12b);
            __m256 _v_t2 = _mm256_sub_ps(_v_t12a, _v_t12b);
            __m256 _v_t3 = _mm256_add_ps(_v_t34a, _v_t34b);
            __m256 _v_t4 = _mm256_sub_ps(_v_t34a, _v_t34b);
            __m256 _v_t5 = _mm256_add_ps(_v_t56a, _v_t56b);
            __m256 _v_t6 = _mm256_sub_ps(_v_t56a, _v_t56b);
            __m256 _v_t7 = _mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r3, _v_r5), _mm256_sub_ps(_v_r7, _v_r1));

            _mm256_storeu_ps(BTd + 0 * 8 * 8 + i * 8, _v_t0);
            _mm256_storeu_ps(BTd + 1 * 8 * 8 + i * 8, _v_t1);
            _mm256_storeu_ps(BTd + 2 * 8 * 8 + i * 8, _v_t2);
            _mm256_storeu_ps(BTd + 3 * 8 * 8 + i * 8, _v_t3);
            _mm256_storeu_ps(BTd + 4 * 8 * 8 + i * 8, _v_t4);
            _mm256_storeu_ps(BTd + 5 * 8 * 8 + i * 8, _v_t5);
            _mm256_storeu_ps(BTd + 6 * 8 * 8 + i * 8, _v_t6);
            _mm256_storeu_ps(BTd + 7 * 8 * 8 + i * 8, _v_t7);
        }
    }

    inline void getBTdB(float* BTdB, int ldx, const float* BTd) const {
        __m256 _v5_25 = _mm256_set1_ps(5.25f);
        __m256 _vm4_25 = _mm256_set1_ps(-4.25f);
        __m256 _vm1_25 = _mm256_set1_ps(-1.25f);
        __m256 _v0_25 = _mm256_set1_ps(0.25f);
        __m256 _vm2_5 = _mm256_set1_ps(-2.5f);
        __m256 _v0_5 = _mm256_set1_ps(0.5f);
        __m256 _v2 = _mm256_set1_ps(2.f);
        __m256 _v4 = _mm256_set1_ps(4.f);

        for (int i = 0; i < 8; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(BTd + i * 8 * 8 + 0 * 8);
            __m256 _v_r1 = _mm256_loadu_ps(BTd + i * 8 * 8 + 1 * 8);
            __m256 _v_r2 = _mm256_loadu_ps(BTd + i * 8 * 8 + 2 * 8);
            __m256 _v_r3 = _mm256_loadu_ps(BTd + i * 8 * 8 + 3 * 8);
            __m256 _v_r4 = _mm256_loadu_ps(BTd + i * 8 * 8 + 4 * 8);
            __m256 _v_r5 = _mm256_loadu_ps(BTd + i * 8 * 8 + 5 * 8);
            __m256 _v_r6 = _mm256_loadu_ps(BTd + i * 8 * 8 + 6 * 8);
            __m256 _v_r7 = _mm256_loadu_ps(BTd + i * 8 * 8 + 7 * 8);

            __m256 _v_t12a = _mm256_fmadd_ps(_vm4_25, _v_r4, _mm256_add_ps(_v_r2, _v_r6));
            __m256 _v_t12b = _mm256_fmadd_ps(_vm4_25, _v_r3, _mm256_add_ps(_v_r1, _v_r5));
            __m256 _v_t34a = _mm256_fmadd_ps(_vm1_25, _v_r4, _mm256_fmadd_ps(_v0_25, _v_r2, _v_r6));
            __m256 _v_t34b = _mm256_fmadd_ps(_v2, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v0_5)));
            __m256 _v_t56a = _mm256_fmadd_ps(_v4, _mm256_fmadd_ps(_vm1_25, _v_r4, _v_r2), _v_r6);
            __m256 _v_t56b = _mm256_fmadd_ps(_v0_5, _v_r5, _mm256_fmadd_ps(_vm2_5, _v_r3, _mm256_mul_ps(_v_r1, _v2)));

            __m256 _v_t0 = _mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r4, _v_r2), _mm256_sub_ps(_v_r0, _v_r1));
            __m256 _v_t1 = _mm256_add_ps(_v_t12a, _v_t12b);
            __m256 _v_t2 = _mm256_sub_ps(_v_t12a, _v_t12b);
            __m256 _v_t3 = _mm256_add_ps(_v_t34a, _v_t34b);
            __m256 _v_t4 = _mm256_sub_ps(_v_t34a, _v_t34b);
            __m256 _v_t5 = _mm256_add_ps(_v_t56a, _v_t56b);
            __m256 _v_t6 = _mm256_sub_ps(_v_t56a, _v_t56b);
            __m256 _v_t7 = _mm256_fmadd_ps(_v5_25, _mm256_sub_ps(_v_r3, _v_r5), _mm256_sub_ps(_v_r7, _v_r1));

            _mm256_storeu_ps(BTdB + 0 * 8, _v_t0);
            _mm256_storeu_ps(BTdB + 1 * 8, _v_t1);
            _mm256_storeu_ps(BTdB + 2 * 8, _v_t2);
            _mm256_storeu_ps(BTdB + 3 * 8, _v_t3);
            _mm256_storeu_ps(BTdB + 4 * 8, _v_t4);
            _mm256_storeu_ps(BTdB + 5 * 8, _v_t5);
            _mm256_storeu_ps(BTdB + 6 * 8, _v_t6);
            _mm256_storeu_ps(BTdB + 7 * 8, _v_t7);

            BTdB += ldx;
        }
    }

   public:
    inline void operator()(float* v, int ldv, const float* d, int ldd) const {
        float BTd[512] = {0};
        getBTd(BTd, d, ldd);
        getBTdB(v, ldv, BTd);
    }
};

template <>
struct WinogradUVTransform<8, 3, 1, 1> {
    inline void getATUV(float* ATUV, const float* UV, int lduv) const {
        for (int i = 0; i < 8; ++i) {
            float r0 = UV[0 * lduv + i];
            float r1 = UV[1 * lduv + i];
            float r2 = UV[2 * lduv + i];
            float r3 = UV[3 * lduv + i];
            float r4 = UV[4 * lduv + i];
            float r5 = UV[5 * lduv + i];
            float r6 = UV[6 * lduv + i];
            float r7 = UV[7 * lduv + i];

            float t024a = r1 + r2;
            float t135a = r1 - r2;
            float t024b = r3 + r4;
            float t135b = r3 - r4;
            float t024c = r5 + r6;
            float t135c = r5 - r6;

            float t0 = (r0 + t024a) + (32 * t024c + t024b);
            float t1 = 16 * t135c + (2 * t135b + t135a);
            float t2 = 8 * t024c + (4 * t024b + t024a);
            float t3 = 4 * t135c + (8 * t135b + t135a);
            float t4 = 2 * t024c + (16 * t024b + t024a);
            float t5 = (r7 + t135a) + (32 * t135b + t135c);

            ATUV[0 * 8 + i] = t0;
            ATUV[1 * 8 + i] = t1;
            ATUV[2 * 8 + i] = t2;
            ATUV[3 * 8 + i] = t3;
            ATUV[4 * 8 + i] = t4;
            ATUV[5 * 8 + i] = t5;
        }
    }

    inline void getATUVA(float* ATUVA, int ldx, const float* ATUV) const {
        for (int i = 0; i < 6; ++i) {
            float r0 = ATUV[i * 8 + 0];
            float r1 = ATUV[i * 8 + 1];
            float r2 = ATUV[i * 8 + 2];
            float r3 = ATUV[i * 8 + 3];
            float r4 = ATUV[i * 8 + 4];
            float r5 = ATUV[i * 8 + 5];
            float r6 = ATUV[i * 8 + 6];
            float r7 = ATUV[i * 8 + 7];

            float t024a = r1 + r2;
            float t135a = r1 - r2;
            float t024b = r3 + r4;
            float t135b = r3 - r4;
            float t024c = r5 + r6;
            float t135c = r5 - r6;

            float t0 = (r0 + t024a) + (32 * t024c + t024b);
            float t1 = 16 * t135c + (2 * t135b + t135a);
            float t2 = 8 * t024c + (4 * t024b + t024a);
            float t3 = 4 * t135c + (8 * t135b + t135a);
            float t4 = 2 * t024c + (16 * t024b + t024a);
            float t5 = (r7 + t135a) + (32 * t135b + t135c);

            ATUVA[0] = t0;
            ATUVA[1] = t1;
            ATUVA[2] = t2;
            ATUVA[3] = t3;
            ATUVA[4] = t4;
            ATUVA[5] = t5;
        }
    }

    inline void operator()(float* y, int ldy, const float* uv, int lduv) const {
        float ATUV[48] = {0};
        getATUV(ATUV, uv, lduv);
        getATUVA(y, ldy, ATUV);
    }
};

template <>
struct WinogradUVTransform<8, 3, 1, 4> {
   private:
    inline void getATUV(float* ATUV, const float* UV, int lduv) const {
        __m128 _v32 = _mm_set1_ps(32.0f);
        __m128 _v16 = _mm_set1_ps(16.0f);
        __m128 _v8 = _mm_set1_ps(8.f);
        __m128 _v4 = _mm_set1_ps(4.f);
        __m128 _v2 = _mm_set1_ps(2.f);

        for (int i = 0; i < 8; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(UV + 0 * lduv + i * 4);
            __m128 _v_r1 = _mm_loadu_ps(UV + 1 * lduv + i * 4);
            __m128 _v_r2 = _mm_loadu_ps(UV + 2 * lduv + i * 4);
            __m128 _v_r3 = _mm_loadu_ps(UV + 3 * lduv + i * 4);
            __m128 _v_r4 = _mm_loadu_ps(UV + 4 * lduv + i * 4);
            __m128 _v_r5 = _mm_loadu_ps(UV + 5 * lduv + i * 4);
            __m128 _v_r6 = _mm_loadu_ps(UV + 6 * lduv + i * 4);
            __m128 _v_r7 = _mm_loadu_ps(UV + 7 * lduv + i * 4);

            __m128 _v_t024a = _mm_add_ps(_v_r1, _v_r2);
            __m128 _v_t135a = _mm_sub_ps(_v_r1, _v_r2);
            __m128 _v_t024b = _mm_add_ps(_v_r3, _v_r4);
            __m128 _v_t135b = _mm_sub_ps(_v_r3, _v_r4);
            __m128 _v_t024c = _mm_add_ps(_v_r5, _v_r6);
            __m128 _v_t135c = _mm_sub_ps(_v_r5, _v_r6);

            __m128 _v_t0 = _mm_add_ps(_mm_add_ps(_v_r0, _v_t024a), _mm_fmadd_ps(_v32, _v_t024c, _v_t024b));
            __m128 _v_t1 = _mm_fmadd_ps(_v16, _v_t135c, _mm_fmadd_ps(_v2, _v_t135b, _v_t135a));
            __m128 _v_t2 = _mm_fmadd_ps(_v8, _v_t024c, _mm_fmadd_ps(_v4, _v_t024b, _v_t024a));
            __m128 _v_t3 = _mm_fmadd_ps(_v4, _v_t135c, _mm_fmadd_ps(_v8, _v_t135b, _v_t135a));
            __m128 _v_t4 = _mm_fmadd_ps(_v2, _v_t024c, _mm_fmadd_ps(_v16, _v_t024b, _v_t024a));
            __m128 _v_t5 = _mm_add_ps(_mm_add_ps(_v_r7, _v_t135a), _mm_fmadd_ps(_v32, _v_t135b, _v_t135c));

            _mm_storeu_ps(ATUV + 0 * 8 * 4 + i * 4, _v_t0);
            _mm_storeu_ps(ATUV + 1 * 8 * 4 + i * 4, _v_t1);
            _mm_storeu_ps(ATUV + 2 * 8 * 4 + i * 4, _v_t2);
            _mm_storeu_ps(ATUV + 3 * 8 * 4 + i * 4, _v_t3);
            _mm_storeu_ps(ATUV + 4 * 8 * 4 + i * 4, _v_t4);
            _mm_storeu_ps(ATUV + 5 * 8 * 4 + i * 4, _v_t5);
        }
    }

    inline void getATUVA(float* ATUVA, int ldx, const float* ATUV) const {
        __m128 _v32 = _mm_set1_ps(32.0f);
        __m128 _v16 = _mm_set1_ps(16.0f);
        __m128 _v8 = _mm_set1_ps(8.f);
        __m128 _v4 = _mm_set1_ps(4.f);
        __m128 _v2 = _mm_set1_ps(2.f);

        for (int i = 0; i < 6; ++i) {
            __m128 _v_r0 = _mm_loadu_ps(ATUV + i * 8 * 4 + 0 * 4);
            __m128 _v_r1 = _mm_loadu_ps(ATUV + i * 8 * 4 + 1 * 4);
            __m128 _v_r2 = _mm_loadu_ps(ATUV + i * 8 * 4 + 2 * 4);
            __m128 _v_r3 = _mm_loadu_ps(ATUV + i * 8 * 4 + 3 * 4);
            __m128 _v_r4 = _mm_loadu_ps(ATUV + i * 8 * 4 + 4 * 4);
            __m128 _v_r5 = _mm_loadu_ps(ATUV + i * 8 * 4 + 5 * 4);
            __m128 _v_r6 = _mm_loadu_ps(ATUV + i * 8 * 4 + 6 * 4);
            __m128 _v_r7 = _mm_loadu_ps(ATUV + i * 8 * 4 + 7 * 4);

            __m128 _v_t024a = _mm_add_ps(_v_r1, _v_r2);
            __m128 _v_t135a = _mm_sub_ps(_v_r1, _v_r2);
            __m128 _v_t024b = _mm_add_ps(_v_r3, _v_r4);
            __m128 _v_t135b = _mm_sub_ps(_v_r3, _v_r4);
            __m128 _v_t024c = _mm_add_ps(_v_r5, _v_r6);
            __m128 _v_t135c = _mm_sub_ps(_v_r5, _v_r6);

            __m128 _v_t0 = _mm_add_ps(_mm_add_ps(_v_r0, _v_t024a), _mm_fmadd_ps(_v32, _v_t024c, _v_t024b));
            __m128 _v_t1 = _mm_fmadd_ps(_v16, _v_t135c, _mm_fmadd_ps(_v2, _v_t135b, _v_t135a));
            __m128 _v_t2 = _mm_fmadd_ps(_v8, _v_t024c, _mm_fmadd_ps(_v4, _v_t024b, _v_t024a));
            __m128 _v_t3 = _mm_fmadd_ps(_v4, _v_t135c, _mm_fmadd_ps(_v8, _v_t135b, _v_t135a));
            __m128 _v_t4 = _mm_fmadd_ps(_v2, _v_t024c, _mm_fmadd_ps(_v16, _v_t024b, _v_t024a));
            __m128 _v_t5 = _mm_add_ps(_mm_add_ps(_v_r7, _v_t135a), _mm_fmadd_ps(_v32, _v_t135b, _v_t135c));

            _mm_storeu_ps(ATUVA + 0 * 4, _v_t0);
            _mm_storeu_ps(ATUVA + 1 * 4, _v_t1);
            _mm_storeu_ps(ATUVA + 2 * 4, _v_t2);
            _mm_storeu_ps(ATUVA + 3 * 4, _v_t3);
            _mm_storeu_ps(ATUVA + 4 * 4, _v_t4);
            _mm_storeu_ps(ATUVA + 5 * 4, _v_t5);
            ATUVA += ldx;
        }
    }

   public:
    inline void operator()(float* y, int ldy, const float* uv, int lduv) const {
        float ATUV[192] = {0};
        getATUV(ATUV, uv, lduv);
        getATUVA(y, ldy, ATUV);
    }
};

template <>
struct WinogradUVTransform<8, 3, 1, 8> {
    /*
AT = np.array([
	[1.0, 1.0,  1.0,  1.0,  1.0, 32.0, 32.0, 0.0],
 	[0.0, 1.0, -1.0,  2.0, -2.0, 16.0,-16.0, 0.0],
  	[0.0, 1.0,  1.0,  4.0,  4.0,  8.0,  8.0, 0.0],
    [0.0, 1.0, -1.0,  8.0, -8.0,  4.0, -4.0, 0.0],
    [0.0, 1.0,  1.0, 16.0, 16.0,  2.0,  2.0, 0.0],
    [0.0, 1.0, -1.0, 32.0,-32.0,  1.0, -1.0, 1.0]
])
*/
   private:
    inline void getATUV(float* ATUV, const float* UV, int lduv) const {
        __m256 _v32 = _mm256_set1_ps(32.0f);
        __m256 _v16 = _mm256_set1_ps(16.0f);
        __m256 _v8 = _mm256_set1_ps(8.f);
        __m256 _v4 = _mm256_set1_ps(4.f);
        __m256 _v2 = _mm256_set1_ps(2.f);

        for (int i = 0; i < 8; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(UV + 0 * lduv + i * 8);
            __m256 _v_r1 = _mm256_loadu_ps(UV + 1 * lduv + i * 8);
            __m256 _v_r2 = _mm256_loadu_ps(UV + 2 * lduv + i * 8);
            __m256 _v_r3 = _mm256_loadu_ps(UV + 3 * lduv + i * 8);
            __m256 _v_r4 = _mm256_loadu_ps(UV + 4 * lduv + i * 8);
            __m256 _v_r5 = _mm256_loadu_ps(UV + 5 * lduv + i * 8);
            __m256 _v_r6 = _mm256_loadu_ps(UV + 6 * lduv + i * 8);
            __m256 _v_r7 = _mm256_loadu_ps(UV + 7 * lduv + i * 8);

            __m256 _v_t024a = _mm256_add_ps(_v_r1, _v_r2);
            __m256 _v_t135a = _mm256_sub_ps(_v_r1, _v_r2);
            __m256 _v_t024b = _mm256_add_ps(_v_r3, _v_r4);
            __m256 _v_t135b = _mm256_sub_ps(_v_r3, _v_r4);
            __m256 _v_t024c = _mm256_add_ps(_v_r5, _v_r6);
            __m256 _v_t135c = _mm256_sub_ps(_v_r5, _v_r6);

            __m256 _v_t0 = _mm256_add_ps(_mm256_add_ps(_v_r0, _v_t024a), _mm256_fmadd_ps(_v32, _v_t024c, _v_t024b));
            __m256 _v_t1 = _mm256_fmadd_ps(_v16, _v_t135c, _mm256_fmadd_ps(_v2, _v_t135b, _v_t135a));
            __m256 _v_t2 = _mm256_fmadd_ps(_v8, _v_t024c, _mm256_fmadd_ps(_v4, _v_t024b, _v_t024a));
            __m256 _v_t3 = _mm256_fmadd_ps(_v4, _v_t135c, _mm256_fmadd_ps(_v8, _v_t135b, _v_t135a));
            __m256 _v_t4 = _mm256_fmadd_ps(_v2, _v_t024c, _mm256_fmadd_ps(_v16, _v_t024b, _v_t024a));
            __m256 _v_t5 = _mm256_add_ps(_mm256_add_ps(_v_r7, _v_t135a), _mm256_fmadd_ps(_v32, _v_t135b, _v_t135c));

            _mm256_storeu_ps(ATUV + 0 * 8 * 8 + i * 8, _v_t0);
            _mm256_storeu_ps(ATUV + 1 * 8 * 8 + i * 8, _v_t1);
            _mm256_storeu_ps(ATUV + 2 * 8 * 8 + i * 8, _v_t2);
            _mm256_storeu_ps(ATUV + 3 * 8 * 8 + i * 8, _v_t3);
            _mm256_storeu_ps(ATUV + 4 * 8 * 8 + i * 8, _v_t4);
            _mm256_storeu_ps(ATUV + 5 * 8 * 8 + i * 8, _v_t5);
        }
    }

    inline void getATUVA(float* ATUVA, int ldx, const float* ATUV) const {
        __m256 _v32 = _mm256_set1_ps(32.0f);
        __m256 _v16 = _mm256_set1_ps(16.0f);
        __m256 _v8 = _mm256_set1_ps(8.f);
        __m256 _v4 = _mm256_set1_ps(4.f);
        __m256 _v2 = _mm256_set1_ps(2.f);

        for (int i = 0; i < 6; ++i) {
            __m256 _v_r0 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 0 * 8);
            __m256 _v_r1 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 1 * 8);
            __m256 _v_r2 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 2 * 8);
            __m256 _v_r3 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 3 * 8);
            __m256 _v_r4 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 4 * 8);
            __m256 _v_r5 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 5 * 8);
            __m256 _v_r6 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 6 * 8);
            __m256 _v_r7 = _mm256_loadu_ps(ATUV + i * 8 * 8 + 7 * 8);

            __m256 _v_t024a = _mm256_add_ps(_v_r1, _v_r2);
            __m256 _v_t135a = _mm256_sub_ps(_v_r1, _v_r2);
            __m256 _v_t024b = _mm256_add_ps(_v_r3, _v_r4);
            __m256 _v_t135b = _mm256_sub_ps(_v_r3, _v_r4);
            __m256 _v_t024c = _mm256_add_ps(_v_r5, _v_r6);
            __m256 _v_t135c = _mm256_sub_ps(_v_r5, _v_r6);

            __m256 _v_t0 = _mm256_add_ps(_mm256_add_ps(_v_r0, _v_t024a), _mm256_fmadd_ps(_v32, _v_t024c, _v_t024b));
            __m256 _v_t1 = _mm256_fmadd_ps(_v16, _v_t135c, _mm256_fmadd_ps(_v2, _v_t135b, _v_t135a));
            __m256 _v_t2 = _mm256_fmadd_ps(_v8, _v_t024c, _mm256_fmadd_ps(_v4, _v_t024b, _v_t024a));
            __m256 _v_t3 = _mm256_fmadd_ps(_v4, _v_t135c, _mm256_fmadd_ps(_v8, _v_t135b, _v_t135a));
            __m256 _v_t4 = _mm256_fmadd_ps(_v2, _v_t024c, _mm256_fmadd_ps(_v16, _v_t024b, _v_t024a));
            __m256 _v_t5 = _mm256_add_ps(_mm256_add_ps(_v_r7, _v_t135a), _mm256_fmadd_ps(_v32, _v_t135b, _v_t135c));

            _mm256_storeu_ps(ATUVA + 0 * 8, _v_t0);
            _mm256_storeu_ps(ATUVA + 1 * 8, _v_t1);
            _mm256_storeu_ps(ATUVA + 2 * 8, _v_t2);
            _mm256_storeu_ps(ATUVA + 3 * 8, _v_t3);
            _mm256_storeu_ps(ATUVA + 4 * 8, _v_t4);
            _mm256_storeu_ps(ATUVA + 5 * 8, _v_t5);
            ATUVA += ldx;
        }
    }

   public:
    inline void operator()(float* y, int ldy, const float* uv, int lduv) const {
        float ATUV[384] = {0};
        getATUV(ATUV, uv, lduv);
        getATUVA(y, ldy, ATUV);
    }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum