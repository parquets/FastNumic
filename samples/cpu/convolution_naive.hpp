#pragma once
#include <stdio.h>


void convolution1d_naive(float* dst,
                  const float* src,
                  int src_w,
                  int in_channels,
                  int out_channels,
                  const float* weight,
                  int kernel_w,
                  int stride_w,
                  int pad_w) {

    int out_w = (src_w - kernel_w) / stride_w + 1;

    for(int oc = 0; oc < out_channels; ++oc) {
        for(int w=0; w < out_w; ++w) {
            float* dst_ptr = dst + oc * out_w + w;
            for(int ic = 0; ic < in_channels; ++ic) {
                const float* src_ptr = src + ic * src_w + w * stride_w;
                const float* weight_ptr = weight + oc * in_channels * kernel_w + ic * kernel_w;
                for(int k=0; k < kernel_w; ++k) {
                    *dst_ptr += src_ptr[k]*weight_ptr[k];
                }
            }
        }
    }
}

void convolution_naive(float* dst,
                       const float* src,
                       int src_h,
                       int src_w,
                       int in_channels,
                       int out_channels,
                       const float* weight,
                       int kernel_h,
                       int kernel_w,
                       int stride_h,
                       int stride_w,
                       int pad_h,
                       int pad_w) {
    int dst_h = (src_h - kernel_h + 2 * pad_h) / stride_h + 1;
    int dst_w = (src_w - kernel_w + 2 * pad_w) / stride_w + 1;

    int src_ch_stride = src_h * src_w;
    int dst_ch_stride = dst_h * dst_w;

    for (int oc = 0; oc < out_channels; ++oc) {
        const float* weight_oc_ptr = weight + oc * kernel_h * kernel_w * in_channels;

        for (int ic = 0; ic < in_channels; ++ic) {
            const float* src_channel_ptr = src + ic * src_ch_stride;
            const float* weight_ic_ptr = weight_oc_ptr + ic * kernel_h * kernel_w;
            float* dst_ptr = dst + dst_ch_stride * oc;

            for (int h = 0; h < src_h - (kernel_h - 1); h += stride_h) {
                for (int w = 0; w < src_w - (kernel_w - 1); w += stride_w) {
                    float t = 0;
                    // printf("h=%d, w=%d\n", h, w);
                    const float* src_kernel_ptr = src_channel_ptr + h * src_w + w;

                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            t += src_kernel_ptr[kh * src_w + kw] * weight_ic_ptr[kh * kernel_w + kw];
                        }
                    }
                    *dst_ptr++ += t;
                }
            }
        }
    }
}