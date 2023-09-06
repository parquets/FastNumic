#include <math.h>
#include "convloution.hpp"

static inline void im2col_chw(float* dst, const float* src, 
                              int C, int H, int W, 
                              int kernel_h, int kernel_w, int stride_h,
                              int stride_w, int pad_h, int pad_w) {

    int outH = (H + 2 * pad_h - kernel_h) / stride_h + 1;
    int outW = (W + 2 * pad_w + kernel_w) / stride_w + 1;

    int c_stride = H * W;

    for (int oh = 0; oh < outH; ++oh) {
        for (int ow = 0; ow < outW; ++ow) {
            int ofset_h = oh * stride_h;
            int ofset_w = ow * stride_w;

            const float* src_ptr = src + ofset_h * W + ofset_w;

            for (int c = 0; c < C; ++c) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        *dst++ = src_ptr[kh * W + kw];
                    }
                }
                src_ptr += c_stride;
            }
        }
    }
}