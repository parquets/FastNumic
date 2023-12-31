#pragma once

#include <memory.h>
#include <stdio.h>
#include <algorithm>
#include <cmath>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

#include "vectorize_x86.hpp"
#include "stranspose_x86.hpp"

namespace fastnum {
namespace cpu {
namespace kernel {

/*
[oc, ic, x]
[0, 0, x], [0, 1, x], ... , [0, 6, x], [0, 7, x] -> [0, 0, x], [1, 0, x], ...,  [6, 0, x], [7, 0, x]
[1, 0, x], [1, 1, x], ... , [1, 6, x], [1, 7, x]    [0, 1, x], [1, 1, x], ...,  [6, 1, x], [7, 1, x]
.                                                   .
.                                                   .
.                                                   .
[6, 0, x], [6, 1, x], ... , [6, 6, x], [6, 7, x]    [0, 6, x], [1, 6, x], ...,  [6, 6, x], [7, 6, x]
[7, 0, x], [7, 1, x], ... , [7, 6, x], [7, 7, x]    [0, 7, x], [1, 7, x], ...,  [6, 7, x], [7, 7, x]

*/

static inline void conv1dWeightPackOCUnit8(float* dest, const float* weight,
                                          int in_channels, int out_channels,
                                          int kernel_size) {

    const float* weight_ptr0 = weight + 0 * in_channels * kernel_size;
    const float* weight_ptr1 = weight + 1 * in_channels * kernel_size;
    const float* weight_ptr2 = weight + 2 * in_channels * kernel_size;
    const float* weight_ptr3 = weight + 3 * in_channels * kernel_size;
    const float* weight_ptr4 = weight + 4 * in_channels * kernel_size;
    const float* weight_ptr5 = weight + 5 * in_channels * kernel_size;
    const float* weight_ptr6 = weight + 6 * in_channels * kernel_size;
    const float* weight_ptr7 = weight + 7 * in_channels * kernel_size;

    __m256i _v_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    _v_index = _mm256_mullo_epi32(_v_index, _mm256_set1_epi32(kernel_size));

    int ic = 0;
    for(ic = 0; ic < in_channels - 7; ic += 8) {
        for(int k = 0; k < kernel_size; ++k) {

            __m256 _v_k0 = _mm256_i32gather_ps(weight_ptr0 + k, _v_index, sizeof(float));
            __m256 _v_k1 = _mm256_i32gather_ps(weight_ptr1 + k, _v_index, sizeof(float));
            __m256 _v_k2 = _mm256_i32gather_ps(weight_ptr2 + k, _v_index, sizeof(float));
            __m256 _v_k3 = _mm256_i32gather_ps(weight_ptr3 + k, _v_index, sizeof(float));
            __m256 _v_k4 = _mm256_i32gather_ps(weight_ptr4 + k, _v_index, sizeof(float));
            __m256 _v_k5 = _mm256_i32gather_ps(weight_ptr5 + k, _v_index, sizeof(float));
            __m256 _v_k6 = _mm256_i32gather_ps(weight_ptr6 + k, _v_index, sizeof(float));
            __m256 _v_k7 = _mm256_i32gather_ps(weight_ptr7 + k, _v_index, sizeof(float));

            transpose_8x8(_v_k0, _v_k1, _v_k2, _v_k3, _v_k4, _v_k5, _v_k6, _v_k7);

            _mm256_storeu_ps(dest + 8 * 0, _v_k0);
            _mm256_storeu_ps(dest + 8 * 1, _v_k1);
            _mm256_storeu_ps(dest + 8 * 2, _v_k2);
            _mm256_storeu_ps(dest + 8 * 3, _v_k3);
            _mm256_storeu_ps(dest + 8 * 4, _v_k4);
            _mm256_storeu_ps(dest + 8 * 5, _v_k5);
            _mm256_storeu_ps(dest + 8 * 6, _v_k6);
            _mm256_storeu_ps(dest + 8 * 7, _v_k7);

            dest += 64;
        }

        weight_ptr0 += 8 * kernel_size;
        weight_ptr1 += 8 * kernel_size;
        weight_ptr2 += 8 * kernel_size;
        weight_ptr3 += 8 * kernel_size;
        weight_ptr4 += 8 * kernel_size;
        weight_ptr5 += 8 * kernel_size;
        weight_ptr6 += 8 * kernel_size;
        weight_ptr7 += 8 * kernel_size;
    }

    __m128i _v_index_128 = _mm_setr_epi32(0, 1, 2, 3);
    _v_index_128 = _mm_mullo_epi32(_v_index_128, _mm_set1_epi32(kernel_size));

    for(; ic < in_channels - 3; ic += 4) {
        for(int k = 0; k < kernel_size; ++k)  {

            __m128 _v_k0 = _mm_i32gather_ps(weight_ptr0 + k, _v_index_128, sizeof(float));
            __m128 _v_k1 = _mm_i32gather_ps(weight_ptr1 + k, _v_index_128, sizeof(float));
            __m128 _v_k2 = _mm_i32gather_ps(weight_ptr2 + k, _v_index_128, sizeof(float));
            __m128 _v_k3 = _mm_i32gather_ps(weight_ptr3 + k, _v_index_128, sizeof(float));
            __m128 _v_k4 = _mm_i32gather_ps(weight_ptr4 + k, _v_index_128, sizeof(float));
            __m128 _v_k5 = _mm_i32gather_ps(weight_ptr5 + k, _v_index_128, sizeof(float));
            __m128 _v_k6 = _mm_i32gather_ps(weight_ptr6 + k, _v_index_128, sizeof(float));
            __m128 _v_k7 = _mm_i32gather_ps(weight_ptr7 + k, _v_index_128, sizeof(float));

            transpose_8x4(_v_k0, _v_k1, _v_k2, _v_k3, _v_k4, _v_k5, _v_k6, _v_k7);

            _mm_storeu_ps(dest + 0 * 4, _v_k0);
            _mm_storeu_ps(dest + 1 * 4, _v_k1);
            _mm_storeu_ps(dest + 2 * 4, _v_k2);
            _mm_storeu_ps(dest + 3 * 4, _v_k3);
            _mm_storeu_ps(dest + 4 * 4, _v_k4);
            _mm_storeu_ps(dest + 5 * 4, _v_k5);
            _mm_storeu_ps(dest + 6 * 4, _v_k6);
            _mm_storeu_ps(dest + 7 * 4, _v_k7);
            dest += 32;
        }

        weight_ptr0 += 4 * kernel_size;
        weight_ptr1 += 4 * kernel_size;
        weight_ptr2 += 4 * kernel_size;
        weight_ptr3 += 4 * kernel_size;
        weight_ptr4 += 4 * kernel_size;
        weight_ptr5 += 4 * kernel_size;
        weight_ptr6 += 4 * kernel_size;
        weight_ptr7 += 4 * kernel_size;
    }

    for(; ic < in_channels; ++ic) {
        for(int k = 0; k < kernel_size; ++k) {
            for(int i = 0; i < 4; ++i) {
                dest[0] = weight_ptr0[i* kernel_size + k];
                dest[1] = weight_ptr1[i* kernel_size + k];
                dest[2] = weight_ptr2[i* kernel_size + k];
                dest[3] = weight_ptr3[i* kernel_size + k];
                dest[4] = weight_ptr4[i* kernel_size + k];
                dest[5] = weight_ptr5[i* kernel_size + k];
                dest[6] = weight_ptr6[i* kernel_size + k];
                dest[7] = weight_ptr7[i* kernel_size + k];
                dest += 8;
            }
        }
        weight_ptr0 += kernel_size;
        weight_ptr1 += kernel_size;
        weight_ptr2 += kernel_size;
        weight_ptr3 += kernel_size;
        weight_ptr4 += kernel_size;
        weight_ptr5 += kernel_size;
        weight_ptr6 += kernel_size;
        weight_ptr7 += kernel_size;
    }
}

static inline void conv1dWeightPackOCUnit4(float* dest, const float* weight,
                                          int in_channels, int out_channels,
                                          int kernel_size) {

    const float* weight_ptr0 = weight + 0 * in_channels * kernel_size;
    const float* weight_ptr1 = weight + 1 * in_channels * kernel_size;
    const float* weight_ptr2 = weight + 2 * in_channels * kernel_size;
    const float* weight_ptr3 = weight + 3 * in_channels * kernel_size;

    __m256i _v_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    _v_index = _mm256_mullo_epi32(_v_index, _mm256_set1_epi32(kernel_size));

    int ic = 0;
    for(ic = 0; ic < in_channels - 7; ic += 8) {
        for(int k = 0; k < kernel_size; ++k) {
            __m256 _v_k0 = _mm256_i32gather_ps(weight_ptr0 + k, _v_index, sizeof(float));
            __m256 _v_k1 = _mm256_i32gather_ps(weight_ptr1 + k, _v_index, sizeof(float));
            __m256 _v_k2 = _mm256_i32gather_ps(weight_ptr2 + k, _v_index, sizeof(float));
            __m256 _v_k3 = _mm256_i32gather_ps(weight_ptr3 + k, _v_index, sizeof(float));

            transpose_4x8(_v_k0, _v_k1, _v_k2, _v_k3);

            _mm256_storeu_ps(dest + 0 * 8, _v_k0);
            _mm256_storeu_ps(dest + 1 * 8, _v_k1);
            _mm256_storeu_ps(dest + 2 * 8, _v_k2);
            _mm256_storeu_ps(dest + 3 * 8, _v_k3);

            dest += 32;
        }

        weight_ptr0 += 8 * kernel_size;
        weight_ptr1 += 8 * kernel_size;
        weight_ptr2 += 8 * kernel_size;
        weight_ptr3 += 8 * kernel_size;
    }

    __m128i _v_index_128 = _mm_setr_epi32(0, 1, 2, 3);
    _v_index_128 = _mm_mullo_epi32(_v_index_128, _mm_set1_epi32(kernel_size));

    for(; ic < in_channels - 3; ic += 4) {
        for(int k = 0; k < kernel_size; ++k) {
            __m128 _v_k0 = _mm_i32gather_ps(weight_ptr0 + k, _v_index_128, sizeof(float));
            __m128 _v_k1 = _mm_i32gather_ps(weight_ptr1 + k, _v_index_128, sizeof(float));
            __m128 _v_k2 = _mm_i32gather_ps(weight_ptr2 + k, _v_index_128, sizeof(float));
            __m128 _v_k3 = _mm_i32gather_ps(weight_ptr3 + k, _v_index_128, sizeof(float));

            transpose_4x4(_v_k0, _v_k1, _v_k2, _v_k3);

            _mm_storeu_ps(dest + 0 * 4, _v_k0);
            _mm_storeu_ps(dest + 1 * 4, _v_k1);
            _mm_storeu_ps(dest + 2 * 4, _v_k2);
            _mm_storeu_ps(dest + 3 * 4, _v_k3);

            dest += 16;
        }
        weight_ptr0 += 4 * kernel_size;
        weight_ptr1 += 4 * kernel_size;
        weight_ptr2 += 4 * kernel_size;
        weight_ptr3 += 4 * kernel_size;
    }

    for(; ic < in_channels; ++ic) {
        for(int k = 0; k < kernel_size; ++k) {
            for(int i = 0; i < 4; ++i) {
                dest[0] = weight_ptr0[i* kernel_size + k];
                dest[1] = weight_ptr1[i* kernel_size + k];
                dest[2] = weight_ptr2[i* kernel_size + k];
                dest[3] = weight_ptr3[i* kernel_size + k];

                dest += 4;
            }
        }

        weight_ptr0 += kernel_size;
        weight_ptr1 += kernel_size;
        weight_ptr2 += kernel_size;
        weight_ptr3 += kernel_size;
    }
}


// static inline void conv1dWeightPackOC1(float* dest, const float* weight,
//                         int in_channels, int out_channels,
//                         int kernel_size) {
    
//     for(int oc = 0; oc < out_channels; ++oc) {
//         const float* weight_ptr0 = weight + (oc + 0) * in_channels * kernel_size;
        
//         float* dest_ptr = dest + in_channels * kernel_size;

//         int ic = 0;
    
//         for(; ic < in_channels; ++ic) {
//             for(int k = 0; k < kernel_size; ++k) {
//                 for(int i = 0; i < 4; ++i) {
//                     dest_ptr[0] = weight_ptr0[i* kernel_size + k];
//                     dest_ptr += 1;
//                 }
//             }
//             weight_ptr0 += kernel_size;
//         }
//     }
// }


static inline void conv1dOCUnit8(float* dest, const float* source, 
                                 const float* packed_weight,
                                 int in_channels, int out_channels,
                                 int in_w, int kernel_w, int stride_w,
                                 int dilation_w) {
    
    int out_w = (in_w - kernel_w) / stride_w + 1;

    for(int w=0; w < out_w; ++w) {

        __m256 _v_sum0 = _mm256_setzero_ps();
        __m256 _v_sum1 = _mm256_setzero_ps();
        __m256 _v_sum2 = _mm256_setzero_ps();
        __m256 _v_sum3 = _mm256_setzero_ps();

        int ic = 0;
        float* dest_ptr = dest + w;
        for(ic = 0; ic < in_channels - 7; ic += 8) {

            const float* source_ptr = source + ic * in_w + w * stride_w;
            const float* pkw_ptr    = packed_weight + ic * 8 * kernel_w;

            for(int k=0; k < kernel_w; ++k) {
                __m256 _v_r0 = _mm256_set1_ps(source_ptr[0 * in_w]);
                __m256 _v_r1 = _mm256_set1_ps(source_ptr[1 * in_w]);
                __m256 _v_r2 = _mm256_set1_ps(source_ptr[2 * in_w]);
                __m256 _v_r3 = _mm256_set1_ps(source_ptr[3 * in_w]); 
                __m256 _v_r4 = _mm256_set1_ps(source_ptr[4 * in_w]);
                __m256 _v_r5 = _mm256_set1_ps(source_ptr[5 * in_w]);
                __m256 _v_r6 = _mm256_set1_ps(source_ptr[6 * in_w]);
                __m256 _v_r7 = _mm256_set1_ps(source_ptr[7 * in_w]);

                __m256 _v_w0 = _mm256_loadu_ps(pkw_ptr + 0 * 8);
                __m256 _v_w1 = _mm256_loadu_ps(pkw_ptr + 1 * 8);
                __m256 _v_w2 = _mm256_loadu_ps(pkw_ptr + 2 * 8);
                __m256 _v_w3 = _mm256_loadu_ps(pkw_ptr + 3 * 8);
                __m256 _v_w4 = _mm256_loadu_ps(pkw_ptr + 4 * 8);
                __m256 _v_w5 = _mm256_loadu_ps(pkw_ptr + 5 * 8);
                __m256 _v_w6 = _mm256_loadu_ps(pkw_ptr + 6 * 8);
                __m256 _v_w7 = _mm256_loadu_ps(pkw_ptr + 7 * 8);

                _v_sum0 = _mm256_fmadd_ps(_v_r0, _v_w0, _v_sum0);
                _v_sum1 = _mm256_fmadd_ps(_v_r1, _v_w1, _v_sum1);
                _v_sum2 = _mm256_fmadd_ps(_v_r2, _v_w2, _v_sum2);
                _v_sum3 = _mm256_fmadd_ps(_v_r3, _v_w3, _v_sum3);
                _v_sum0 = _mm256_fmadd_ps(_v_r4, _v_w4, _v_sum0);
                _v_sum1 = _mm256_fmadd_ps(_v_r5, _v_w5, _v_sum1);
                _v_sum2 = _mm256_fmadd_ps(_v_r6, _v_w6, _v_sum2);
                _v_sum3 = _mm256_fmadd_ps(_v_r7, _v_w7, _v_sum3);

                pkw_ptr    += 64;
                source_ptr += dilation_w;
            }
        }

        for(; ic < in_channels - 3; ic += 4) {
            const float* source_ptr = source + ic * in_w + w * stride_w;
            const float* pkw_ptr    = packed_weight + ic * 8 * kernel_w;

            for(int k=0; k < kernel_w; ++k) {
                __m256 _v_r0 = _mm256_set1_ps(source_ptr[0 * in_w]);
                __m256 _v_r1 = _mm256_set1_ps(source_ptr[1 * in_w]);
                __m256 _v_r2 = _mm256_set1_ps(source_ptr[2 * in_w]);
                __m256 _v_r3 = _mm256_set1_ps(source_ptr[3 * in_w]); 

                __m256 _v_w0 = _mm256_loadu_ps(pkw_ptr + 0 * 8);
                __m256 _v_w1 = _mm256_loadu_ps(pkw_ptr + 1 * 8);
                __m256 _v_w2 = _mm256_loadu_ps(pkw_ptr + 2 * 8);
                __m256 _v_w3 = _mm256_loadu_ps(pkw_ptr + 3 * 8);

                _v_sum0 = _mm256_fmadd_ps(_v_r0, _v_w0, _v_sum0);
                _v_sum1 = _mm256_fmadd_ps(_v_r1, _v_w1, _v_sum1);
                _v_sum2 = _mm256_fmadd_ps(_v_r2, _v_w2, _v_sum2);
                _v_sum3 = _mm256_fmadd_ps(_v_r3, _v_w3, _v_sum3);

                pkw_ptr    += 32;
                source_ptr += dilation_w;
            }
        }

        for(; ic < in_channels; ic += 1) {
            const float* source_ptr = source + ic * in_w + w * stride_w;
            const float* pkw_ptr    = packed_weight + ic * 8 * kernel_w;

            for(int k=0; k < kernel_w; ++k) {
                __m256 _v_r0 = _mm256_set1_ps(source_ptr[0 * in_w]);
                __m256 _v_w0 = _mm256_loadu_ps(pkw_ptr + 0 * 8);
                _v_sum0 = _mm256_fmadd_ps(_v_r0, _v_w0, _v_sum0);
                pkw_ptr    += 8;
                source_ptr += dilation_w;
            }
        }

        _v_sum0 = _mm256_add_ps(_v_sum0, _v_sum1);
        _v_sum2 = _mm256_add_ps(_v_sum2, _v_sum3);
        _v_sum0 = _mm256_add_ps(_v_sum0, _v_sum2);

        float sum[8];
        _mm256_storeu_ps(sum, _v_sum0);

        dest_ptr[0 * out_w] += sum[0];
        dest_ptr[1 * out_w] += sum[1];
        dest_ptr[2 * out_w] += sum[2];
        dest_ptr[3 * out_w] += sum[3];
        dest_ptr[4 * out_w] += sum[4];
        dest_ptr[5 * out_w] += sum[5];
        dest_ptr[6 * out_w] += sum[6];
        dest_ptr[7 * out_w] += sum[7];
    }

}

}
}
}