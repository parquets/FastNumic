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
struct WinogradWeightTransform {
    inline void operator()(float* u, const float* g) const { printf("Error! Undefined Winograd Weight Transform"); }
};

template <>
struct WinogradWeightTransform<8, 3, 1, 8> {
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

        float Gg[8][8][3] = {0};

        getGg(&Gg[0][0][0], g + 0 * kernel_stride);
        getGg(&Gg[1][0][0], g + 1 * kernel_stride);
        getGg(&Gg[2][0][0], g + 2 * kernel_stride);
        getGg(&Gg[3][0][0], g + 3 * kernel_stride);
        getGg(&Gg[4][0][0], g + 4 * kernel_stride);
        getGg(&Gg[5][0][0], g + 5 * kernel_stride);
        getGg(&Gg[6][0][0], g + 6 * kernel_stride);
        getGg(&Gg[7][0][0], g + 7 * kernel_stride);

        float GgGt[8][8][8] = {0};

        getGgGt(&GgGt[0][0][0], &Gg[0][0][0]);
        getGgGt(&GgGt[1][0][0], &Gg[1][0][0]);
        getGgGt(&GgGt[2][0][0], &Gg[2][0][0]);
        getGgGt(&GgGt[3][0][0], &Gg[3][0][0]);
        getGgGt(&GgGt[4][0][0], &Gg[4][0][0]);
        getGgGt(&GgGt[5][0][0], &Gg[5][0][0]);
        getGgGt(&GgGt[6][0][0], &Gg[6][0][0]);
        getGgGt(&GgGt[7][0][0], &Gg[7][0][0]);

        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                *u++ = GgGt[0][i][j];
                *u++ = GgGt[1][i][j];
                *u++ = GgGt[2][i][j];
                *u++ = GgGt[3][i][j];
                *u++ = GgGt[4][i][j];
                *u++ = GgGt[5][i][j];
                *u++ = GgGt[6][i][j];
                *u++ = GgGt[7][i][j];
            }
        }
    }
};

template <int TileSize, int KernelSize, int StrideSize, int NPack>
struct WinogradDataTransform {};

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

template <int TileSize, int KernelSize, int StrideSize>
struct WinogradUVTransform {};

inline static void weight_transform_i4x4_k3x3_ch4(float* u, const float* g) {

    __m128 _v_g0 = _mm_loadu_ps(g);
    __m128 _v_g1 = _mm_loadu_ps(g + 4);
    __m128 _v_g2 = _mm_loadu_ps(g + 8);

    __m128 _v_g3 = _mm_loadu_ps(g + 12);
    __m128 _v_g4 = _mm_loadu_ps(g + 16);
    __m128 _v_g5 = _mm_loadu_ps(g + 20);

    __m128 _v_g6 = _mm_loadu_ps(g + 24);
    __m128 _v_g7 = _mm_loadu_ps(g + 28);
    __m128 _v_g8 = _mm_loadu_ps(g + 32);

    __m128 _v_Gg0 = _v_g0;
    __m128 _v_Gg1 = _v_g1;
    __m128 _v_Gg2 = _v_g2;
    __m128 _v_Gg3 = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_v_g0, _v_g3), _v_g6), _mm_set1_ps(0.5));
    __m128 _v_Gg4 = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_v_g1, _v_g4), _v_g7), _mm_set1_ps(0.5));
    __m128 _v_Gg5 = _mm_mul_ps(_mm_add_ps(_mm_add_ps(_v_g2, _v_g5), _v_g8), _mm_set1_ps(0.5));

    __m128 _v_Gg6 = _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_v_g0, _v_g3), _v_g6), _mm_set1_ps(0.5));
    __m128 _v_Gg7 = _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_v_g1, _v_g4), _v_g7), _mm_set1_ps(0.5));
    __m128 _v_Gg8 = _mm_mul_ps(_mm_add_ps(_mm_sub_ps(_v_g2, _v_g5), _v_g8), _mm_set1_ps(0.5));

    __m128 _v_Gg9 = _v_g6;
    __m128 _v_Gg10 = _v_g7;
    __m128 _v_Gg11 = _v_g8;

    __m128 _v_Gg0_Add_Gg2 = _mm_add_ps(_v_Gg0, _v_Gg2);

    _mm_storeu_ps(u + 0 * 4, _v_Gg0);
    _mm_storeu_ps(u + 1 * 4, _mm_mul_ps(_mm_add_ps(_v_Gg0_Add_Gg2, _v_Gg1), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 2 * 4, _mm_mul_ps(_mm_sub_ps(_v_Gg0_Add_Gg2, _v_Gg1), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 3 * 4, _v_Gg2);

    __m128 _v_Gg3_Add_Gg5 = _mm_add_ps(_v_Gg3, _v_Gg5);
    _mm_storeu_ps(u + 4 * 4, _v_Gg3);
    _mm_storeu_ps(u + 5 * 4, _mm_mul_ps(_mm_add_ps(_v_Gg3_Add_Gg5, _v_Gg4), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 6 * 4, _mm_mul_ps(_mm_sub_ps(_v_Gg3_Add_Gg5, _v_Gg4), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 7 * 4, _v_Gg5);

    __m128 _v_Gg6_Add_Gg8 = _mm_add_ps(_v_Gg6, _v_Gg8);
    _mm_storeu_ps(u + 8 * 4, _v_Gg6);
    _mm_storeu_ps(u + 9 * 4, _mm_mul_ps(_mm_add_ps(_v_Gg6_Add_Gg8, _v_Gg7), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 10 * 4, _mm_mul_ps(_mm_sub_ps(_v_Gg6_Add_Gg8, _v_Gg7), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 11 * 4, _v_Gg8);

    __m128 _v_Gg9_Add_Gg11 = _mm_add_ps(_v_Gg9, _v_Gg11);
    _mm_storeu_ps(u + 12 * 4, _v_Gg9);
    _mm_storeu_ps(u + 13 * 4, _mm_mul_ps(_mm_add_ps(_v_Gg9_Add_Gg11, _v_Gg10), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 14 * 4, _mm_mul_ps(_mm_sub_ps(_v_Gg9_Add_Gg11, _v_Gg10), _mm_set1_ps(0.5)));
    _mm_storeu_ps(u + 15 * 4, _v_Gg11);
}

inline static void data_transform_i4x4_k3x3_ch4(float* v, const float* d) {
    __m128 _v_d0 = _mm_loadu_ps(d + 0 * 4);
    __m128 _v_d1 = _mm_loadu_ps(d + 1 * 4);
    __m128 _v_d2 = _mm_loadu_ps(d + 2 * 4);
    __m128 _v_d3 = _mm_loadu_ps(d + 3 * 4);

    __m128 _v_d8 = _mm_loadu_ps(d + 8 * 4);
    __m128 _v_d9 = _mm_loadu_ps(d + 9 * 4);
    __m128 _v_d10 = _mm_loadu_ps(d + 10 * 4);
    __m128 _v_d11 = _mm_loadu_ps(d + 11 * 4);

    __m128 _v_Btd0 = _mm_sub_ps(_v_d0, _v_d8);
    __m128 _v_Btd1 = _mm_sub_ps(_v_d1, _v_d9);
    __m128 _v_Btd2 = _mm_sub_ps(_v_d2, _v_d10);
    __m128 _v_Btd3 = _mm_sub_ps(_v_d3, _v_d11);

    _mm_storeu_ps(v + 0 * 4, _mm_sub_ps(_v_Btd0, _v_Btd2));
    _mm_storeu_ps(v + 1 * 4, _mm_add_ps(_v_Btd1, _v_Btd2));
    _mm_storeu_ps(v + 2 * 4, _mm_sub_ps(_v_Btd2, _v_Btd1));
    _mm_storeu_ps(v + 3 * 4, _mm_sub_ps(_v_Btd1, _v_Btd3));

    __m128 _v_d4 = _mm_loadu_ps(d + 4 * 4);
    __m128 _v_d5 = _mm_loadu_ps(d + 5 * 4);
    __m128 _v_d6 = _mm_loadu_ps(d + 6 * 4);
    __m128 _v_d7 = _mm_loadu_ps(d + 7 * 4);

    __m128 _v_Btd4 = _mm_add_ps(_v_d4, _v_d8);
    __m128 _v_Btd5 = _mm_add_ps(_v_d5, _v_d9);
    __m128 _v_Btd6 = _mm_add_ps(_v_d6, _v_d10);
    __m128 _v_Btd7 = _mm_add_ps(_v_d7, _v_d11);

    _mm_storeu_ps(v + 4 * 4, _mm_sub_ps(_v_Btd4, _v_Btd6));
    _mm_storeu_ps(v + 5 * 4, _mm_add_ps(_v_Btd5, _v_Btd6));
    _mm_storeu_ps(v + 6 * 4, _mm_sub_ps(_v_Btd6, _v_Btd5));
    _mm_storeu_ps(v + 7 * 4, _mm_sub_ps(_v_Btd5, _v_Btd7));

    __m128 _v_Btd8 = _mm_sub_ps(_v_d8, _v_d4);
    __m128 _v_Btd9 = _mm_sub_ps(_v_d9, _v_d5);
    __m128 _v_Btd10 = _mm_sub_ps(_v_d10, _v_d6);
    __m128 _v_Btd11 = _mm_sub_ps(_v_d11, _v_d7);

    _mm_storeu_ps(v + 8 * 4, _mm_sub_ps(_v_Btd8, _v_Btd10));
    _mm_storeu_ps(v + 9 * 4, _mm_add_ps(_v_Btd9, _v_Btd10));
    _mm_storeu_ps(v + 10 * 4, _mm_sub_ps(_v_Btd10, _v_Btd9));
    _mm_storeu_ps(v + 11 * 4, _mm_sub_ps(_v_Btd9, _v_Btd11));

    __m128 _v_d12 = _mm_loadu_ps(d + 12 * 4);
    __m128 _v_d13 = _mm_loadu_ps(d + 13 * 4);
    __m128 _v_d14 = _mm_loadu_ps(d + 14 * 4);
    __m128 _v_d15 = _mm_loadu_ps(d + 15 * 4);

    __m128 _v_Btd12 = _mm_sub_ps(_v_d4, _v_d12);
    __m128 _v_Btd13 = _mm_sub_ps(_v_d5, _v_d13);
    __m128 _v_Btd14 = _mm_sub_ps(_v_d6, _v_d14);
    __m128 _v_Btd15 = _mm_sub_ps(_v_d7, _v_d15);

    _mm_storeu_ps(v + 12 * 4, _mm_sub_ps(_v_Btd12, _v_Btd14));
    _mm_storeu_ps(v + 13 * 4, _mm_add_ps(_v_Btd13, _v_Btd14));
    _mm_storeu_ps(v + 14 * 4, _mm_sub_ps(_v_Btd14, _v_Btd13));
    _mm_storeu_ps(v + 15 * 4, _mm_sub_ps(_v_Btd13, _v_Btd15));
}

}  // namespace kernel
}  // namespace cpu
}  // namespace fastnum