#pragma once

#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <algorithm>
#include <float.h>

#include <immintrin.h>
#include <nmmintrin.h>
#include <smmintrin.h>

namespace fastnum {
namespace cpu {
namespace kernel {

/*
take kernel 3x3, stride 1 as an example

e0 = x00 + x01 + x02 + x10 + x11 + x12 + x20 + x21 + x22
e1 = x01 + x02 + x03 + x11 + x12 + x13 + x21 + x22 + x23
e2 = x02 + x03 + x04 + x12 + x13 + x14 + x22 + x23 + x24
e3 = x03 + x04 + x05 + x13 + x14 + x15 + x23 + x24 + x25
e4 = x04 + x05 + x06 + x14 + x15 + x16 + x24 + x25 + x26
e5 = x05 + x06 + x07 + x15 + x16 + x17 + x25 + x26 + x27
e6 = x06 + x07 + x08 + x16 + x17 + x18 + x26 + x27 + x28
e7 = x07 + x08 + x09 + x17 + x18 + x19 + x27 + x28 + x29
*/
static inline void meanPoolingHorizontal8(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {
    int pool_size = kernel_h * kernel_w;
    __m256 _v_x0 = _mm256_setzero_ps();

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw);
                _v_x0 = _mm256_add_ps(_v_x0, _v_d0);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                            4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                _v_x0 = _mm256_add_ps(_v_x0, _mm256_i32gather_ps(source + kh * source_ldh + kw, _v_index, sizeof(float)));
            }
        }
    }
    __m256 _v_scale = _mm256_set1_ps(1.0/pool_size);
    _mm256_storeu_ps(dest_value, _mm256_mul_ps(_v_x0, _v_scale));
}

static inline void meanPoolingHorizontal4(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {
    int pool_size = kernel_h * kernel_w;
    __m128 _v_x0 = _mm_setzero_ps();

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m128 _v_d0 = _mm_loadu_ps(source + kh * source_ldh + kw);
                _v_x0 = _mm_add_ps(_v_x0, _v_d0);
            }
        }
    } else {
        __m128i _v_index = _mm_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                _v_x0 = _mm_add_ps(_v_x0, _mm_i32gather_ps(source + kh * source_ldh + kw, _v_index, sizeof(float)));
            }
        }
    }

    __m128 _v_scale = _mm_set1_ps(1.0/pool_size);
    _mm_storeu_ps(dest_value, _mm_mul_ps(_v_x0, _v_scale));
}


static inline void meanPoolingHorizontal16(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {
    int pool_size = kernel_h * kernel_w;
    __m256 _v_x0 = _mm256_setzero_ps();
    __m256 _v_x1 = _mm256_setzero_ps();

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);
                _v_x0 = _mm256_add_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_add_ps(_v_x1, _v_d1);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                _v_x0 = _mm256_add_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_add_ps(_v_x1, _v_d1);
            }
        }
    }
    __m256 _v_scale = _mm256_set1_ps(1.0/pool_size);
    _mm256_storeu_ps(dest_value + 0 * 8, _mm256_mul_ps(_v_x0, _v_scale));
    _mm256_storeu_ps(dest_value + 1 * 8, _mm256_mul_ps(_v_x1, _v_scale));
}

static inline void meanPoolingHorizontal32(float* dest_value,
                                            const float* source,
                                            int source_ldh,
                                            int kernel_h,
                                            int kernel_w,
                                            int stride_w) {
    int pool_size = kernel_h * kernel_w;
    __m256 _v_x0 = _mm256_setzero_ps();
    __m256 _v_x1 = _mm256_setzero_ps();
    __m256 _v_x2 = _mm256_setzero_ps();
    __m256 _v_x3 = _mm256_setzero_ps();

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);
                __m256 _v_d2 = _mm256_loadu_ps(source + kh * source_ldh + kw + 2 * 8);
                __m256 _v_d3 = _mm256_loadu_ps(source + kh * source_ldh + kw + 3 * 8);
                _v_x0 = _mm256_add_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_add_ps(_v_x1, _v_d1);
                _v_x2 = _mm256_add_ps(_v_x2, _v_d2);
                _v_x3 = _mm256_add_ps(_v_x3, _v_d3);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d2 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 2 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d3 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 3 * 8 * stride_w, _v_index, sizeof(float));
                _v_x0 = _mm256_add_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_add_ps(_v_x1, _v_d1);
                _v_x2 = _mm256_add_ps(_v_x2, _v_d2);
                _v_x3 = _mm256_add_ps(_v_x3, _v_d3);

            }
        }
    }
    __m256 _v_scale = _mm256_set1_ps(1.0/pool_size);
    _mm256_storeu_ps(dest_value + 0 * 8, _mm256_mul_ps(_v_x0, _v_scale));
    _mm256_storeu_ps(dest_value + 1 * 8, _mm256_mul_ps(_v_x1, _v_scale));
    _mm256_storeu_ps(dest_value + 2 * 8, _mm256_mul_ps(_v_x2, _v_scale));
    _mm256_storeu_ps(dest_value + 3 * 8, _mm256_mul_ps(_v_x3, _v_scale));
}


static inline void maxPoolingHorizontal8(float* dest_value,
                                         const float* source,
                                         int source_ldh,
                                         int kernel_h,
                                         int kernel_w,
                                         int stride_w) {
    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw);
                _v_x0 = _mm256_max_ps(_v_x0, _v_d0);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                            4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                _v_x0 = _mm256_max_ps(_v_x0, _mm256_i32gather_ps(source + kh * source_ldh + kw, _v_index, sizeof(float)));
            }
        }
    }

    _mm256_storeu_ps(dest_value, _v_x0);
}

static inline void maxPoolingHorizontal4(float* dest_value,
                                         const float* source,
                                         int source_ldh,
                                         int kernel_h,
                                         int kernel_w,
                                         int stride_w) {

    __m128 _v_x0 = _mm_set1_ps(-FLT_MAX);

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m128 _v_d0 = _mm_loadu_ps(source + kh * source_ldh + kw);
                _v_x0 = _mm_max_ps(_v_x0, _v_d0);
            }
        }
    } else {
        __m128i _v_index = _mm_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                _v_x0 = _mm_max_ps(_v_x0, _mm_i32gather_ps(source + kh * source_ldh + kw, _v_index, sizeof(float)));
            }
        }
    }

    _mm_storeu_ps(dest_value, _v_x0);
}


static inline void maxPoolingHorizontal16(float* dest_value,
                                          const float* source,
                                          int source_ldh,
                                          int kernel_h,
                                          int kernel_w,
                                          int stride_w) {

    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x1 = _mm256_set1_ps(-FLT_MAX);

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);
                _v_x0 = _mm256_max_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_max_ps(_v_x1, _v_d1);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                _v_x0 = _mm256_max_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_max_ps(_v_x1, _v_d1);
            }
        }
    }
    _mm256_storeu_ps(dest_value + 0 * 8, _v_x0);
    _mm256_storeu_ps(dest_value + 1 * 8, _v_x1);
}

static inline void maxPoolingHorizontal32(float* dest_value,
                                           const float* source,
                                           int source_ldh,
                                           int kernel_h,
                                           int kernel_w,
                                           int stride_w) {
    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x1 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x2 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x3 = _mm256_set1_ps(-FLT_MAX);

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);
                __m256 _v_d2 = _mm256_loadu_ps(source + kh * source_ldh + kw + 2 * 8);
                __m256 _v_d3 = _mm256_loadu_ps(source + kh * source_ldh + kw + 3 * 8);
                _v_x0 = _mm256_max_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_max_ps(_v_x1, _v_d1);
                _v_x2 = _mm256_max_ps(_v_x2, _v_d2);
                _v_x3 = _mm256_max_ps(_v_x3, _v_d3);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d2 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 2 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d3 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 3 * 8 * stride_w, _v_index, sizeof(float));
                _v_x0 = _mm256_max_ps(_v_x0, _v_d0);
                _v_x1 = _mm256_max_ps(_v_x1, _v_d1);
                _v_x2 = _mm256_max_ps(_v_x2, _v_d2);
                _v_x3 = _mm256_max_ps(_v_x3, _v_d3);

            }
        }
    }
    _mm256_storeu_ps(dest_value + 0 * 8, _v_x0);
    _mm256_storeu_ps(dest_value + 1 * 8, _v_x1);
    _mm256_storeu_ps(dest_value + 2 * 8, _v_x2);
    _mm256_storeu_ps(dest_value + 3 * 8, _v_x3);
}

static inline void maxPoolingHorizontal8(float* dest_value,
                                         int* dest_index,
                                         const float* source,
                                         int source_ldh,
                                         int kernel_h,
                                         int kernel_w,
                                         int stride_w,
                                         int base_offset) {
    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_diff = _mm256_setr_ps(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                    4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
    __m256 _v_i0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 0 * 8));

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw);
                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                __m256 _v_ti = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw));
                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti, _v_m0);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw, _v_index, sizeof(float));
                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                __m256 _v_ti = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw));
                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti, _v_m0);
            }
        }
    }
    _mm256_storeu_ps(dest_value, _v_x0);
    _mm256_storeu_epi32(dest_index, _mm256_cvtps_epi32(_v_i0));
}

static inline void maxPoolingHorizontal16(float* dest_value,
                                         int* dest_index,
                                         const float* source,
                                         int source_ldh,
                                         int kernel_h,
                                         int kernel_w,
                                         int stride_w,
                                         int base_offset) {
    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x1 = _mm256_set1_ps(-FLT_MAX);

    __m256 _v_diff = _mm256_setr_ps(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                    4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
    __m256 _v_i0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 0 * 8));
    __m256 _v_i1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 1 * 8));

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);

                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                __m256 _v_m1 = _mm256_cmp_ps(_v_x1, _v_d1, 1);
                
                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                _v_x1 = _mm256_blendv_ps(_v_x1, _v_d1, _v_m1);
                
                __m256 _v_ti0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 0));
                __m256 _v_ti1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 8));
                
                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti0, _v_m0);
                _v_i1 = _mm256_blendv_ps(_v_i1, _v_ti1, _v_m1);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                
                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                __m256 _v_m1 = _mm256_cmp_ps(_v_x1, _v_d1, 1);
                
                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                _v_x1 = _mm256_blendv_ps(_v_x1, _v_d1, _v_m1);
                
                __m256 _v_ti0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 0 * 8 * stride_w));
                __m256 _v_ti1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 1 * 8 * stride_w));
                
                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti0, _v_m0);
                _v_i1 = _mm256_blendv_ps(_v_i1, _v_ti1, _v_m1);
            }
        }
    }
    _mm256_storeu_ps(dest_value + 0, _v_x0);
    _mm256_storeu_ps(dest_value + 8, _v_x1);
    _mm256_storeu_epi32(dest_index + 0, _mm256_cvtps_epi32(_v_i0));
    _mm256_storeu_epi32(dest_index + 8, _mm256_cvtps_epi32(_v_i1));
}


static inline void maxPoolingHorizontal32(float* dest_value,
                                         int* dest_index,
                                         const float* source,
                                         int source_ldh,
                                         int kernel_h,
                                         int kernel_w,
                                         int stride_w,
                                         int base_offset) {
    __m256 _v_x0 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x1 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x2 = _mm256_set1_ps(-FLT_MAX);
    __m256 _v_x3 = _mm256_set1_ps(-FLT_MAX);

    __m256 _v_diff = _mm256_setr_ps(0.0f * stride_w, 1.0f * stride_w, 2.0f * stride_w, 3.0f * stride_w,
                                    4.0f * stride_w, 5.0f * stride_w, 6.0f * stride_w, 7.0f * stride_w);
    
    __m256 _v_i0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 0 * 8.0f));
    __m256 _v_i1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 1 * 8.0f));
    __m256 _v_i2 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 2 * 8.0f));
    __m256 _v_i3 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + 3 * 8.0f));

    if(stride_w == 1) {
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_loadu_ps(source + kh * source_ldh + kw + 0 * 8);
                __m256 _v_d1 = _mm256_loadu_ps(source + kh * source_ldh + kw + 1 * 8);
                __m256 _v_d2 = _mm256_loadu_ps(source + kh * source_ldh + kw + 2 * 8);
                __m256 _v_d3 = _mm256_loadu_ps(source + kh * source_ldh + kw + 2 * 8);

                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                __m256 _v_m1 = _mm256_cmp_ps(_v_x1, _v_d1, 1);
                __m256 _v_m2 = _mm256_cmp_ps(_v_x2, _v_d2, 1);
                __m256 _v_m3 = _mm256_cmp_ps(_v_x3, _v_d3, 1);

                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                _v_x1 = _mm256_blendv_ps(_v_x1, _v_d1, _v_m1);
                _v_x2 = _mm256_blendv_ps(_v_x2, _v_d2, _v_m2);
                _v_x3 = _mm256_blendv_ps(_v_x3, _v_d3, _v_m3);

                __m256 _v_ti0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 0 * 8));
                __m256 _v_ti1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 1 * 8));
                __m256 _v_ti2 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 2 * 8));
                __m256 _v_ti3 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 3 * 8));

                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti0, _v_m0);
                _v_i1 = _mm256_blendv_ps(_v_i1, _v_ti1, _v_m1);
                _v_i2 = _mm256_blendv_ps(_v_i2, _v_ti2, _v_m2);
                _v_i3 = _mm256_blendv_ps(_v_i3, _v_ti3, _v_m3);
            }
        }
    } else {
        __m256i _v_index = _mm256_setr_epi32(0 * stride_w, 1 * stride_w, 2 * stride_w, 3 * stride_w,
                                             4 * stride_w, 5 * stride_w, 6 * stride_w, 7 * stride_w);
        for(int kh = 0; kh < kernel_h; ++kh) {
            for(int kw = 0; kw < kernel_w; ++kw) {
                __m256 _v_d0 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 0 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d1 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 1 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d2 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 2 * 8 * stride_w, _v_index, sizeof(float));
                __m256 _v_d3 = _mm256_i32gather_ps(source + kh * source_ldh + kw + 3 * 8 * stride_w, _v_index, sizeof(float));

                __m256 _v_m0 = _mm256_cmp_ps(_v_x0, _v_d0, 1);
                __m256 _v_m1 = _mm256_cmp_ps(_v_x1, _v_d1, 1);
                __m256 _v_m2 = _mm256_cmp_ps(_v_x2, _v_d2, 1);
                __m256 _v_m3 = _mm256_cmp_ps(_v_x3, _v_d3, 1);

                _v_x0 = _mm256_blendv_ps(_v_x0, _v_d0, _v_m0);
                _v_x1 = _mm256_blendv_ps(_v_x1, _v_d1, _v_m1);
                _v_x2 = _mm256_blendv_ps(_v_x2, _v_d2, _v_m2);
                _v_x3 = _mm256_blendv_ps(_v_x3, _v_d3, _v_m3);

                __m256 _v_ti0 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 0 * 8 * stride_w));
                __m256 _v_ti1 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 1 * 8 * stride_w));
                __m256 _v_ti2 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 2 * 8 * stride_w));
                __m256 _v_ti3 = _mm256_add_ps(_v_diff, _mm256_set1_ps(base_offset + kh * source_ldh + kw + 3 * 8 * stride_w));

                _v_i0 = _mm256_blendv_ps(_v_i0, _v_ti0, _v_m0);
                _v_i1 = _mm256_blendv_ps(_v_i1, _v_ti1, _v_m1);
                _v_i2 = _mm256_blendv_ps(_v_i2, _v_ti2, _v_m2);
                _v_i3 = _mm256_blendv_ps(_v_i3, _v_ti3, _v_m3);
            }
        }
    }
    _mm256_storeu_ps(dest_value + 0 * 8, _v_x0);
    _mm256_storeu_ps(dest_value + 1 * 8, _v_x1);
    _mm256_storeu_ps(dest_value + 2 * 8, _v_x0);
    _mm256_storeu_ps(dest_value + 3 * 8, _v_x1);

    _mm256_storeu_epi32(dest_index + 0 * 8, _mm256_cvtps_epi32(_v_i0));
    _mm256_storeu_epi32(dest_index + 1 * 8, _mm256_cvtps_epi32(_v_i1));
    _mm256_storeu_epi32(dest_index + 2 * 8, _mm256_cvtps_epi32(_v_i2));
    _mm256_storeu_epi32(dest_index + 3 * 8, _mm256_cvtps_epi32(_v_i3));
}

}
}
}