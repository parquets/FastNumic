#include <math.h>
#include <memory.h>
#include <stdlib.h>

#include "convloution.hpp"
#include "cpu/winograd_block.hpp"
#include "cpu/x86/arithmetic_x86.hpp"
#include "cpu/x86/convolution_kernel_x86.hpp"

namespace fastnum {
namespace cpu {

void convolution_1x1(float* dst, const float* src, const float* weight, int src_h, int src_w, int in_channels, int out_channels) {
    int ch_stride = src_h * src_w;

    for (int oc = 0; oc < out_channels; ++oc) {
        int ic = 0;
        for (ic = 0; ic < in_channels - 5; ic += 6) {
            kernel::convolution_k1s1c6(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
        for (; ic < in_channels - 3; ic += 4) {
            kernel::convolution_k1s1c4(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
        for (; ic < in_channels; ++ic) {
            kernel::convolution_k1s1c1(dst + oc * ch_stride, src + ic * ch_stride, weight + oc * in_channels + ic, src_h, src_w);
        }
    }
}

void convolution_k3s1_wo_weight_tranform(float* dst,
                                         const float* src,
                                         const float* u8,
                                         const float* u4,
                                         int src_h,
                                         int src_w,
                                         int in_channels,
                                         int out_channels) {
    int dst_h = (src_h - 3) + 1;
    int dst_w = (src_w - 3) + 1;

    int nh6 = dst_h / 6.0;
    int nw6 = dst_w / 6.0;

    int nh2 = (dst_h % 6) / 2;
    int nw2 = (dst_w % 6) / 2;

    // printf("dst_h=%d, dst_w=%d\n", dst_h, dst_w);

    fastnum::cpu::WinogradDataTransformByChannelBlock<8, 3, 1> data_trans_t8;
    fastnum::cpu::WinogradUVTransformByChannelBlock<8, 3, 1> uv_trans_t8;

    fastnum::cpu::WinogradDataTransformByChannelBlock<4, 3, 1> data_trans_t4;
    fastnum::cpu::WinogradUVTransformByChannelBlock<4, 3, 1> uv_trans_t4;

    // printf("malloc size = %d\n", sizeof(float) * 8 * 8 * in_channels * std::max((dst_h / 6) * (dst_w / 6), 1));
    int malloc_size_v8 = 8 * 8 * in_channels * std::max((dst_h / 6) * (dst_w / 6), 1);
    int malloc_size_v4 = 4 * 4 * in_channels * std::max((dst_h / 2) * (dst_w / 2), 1);

    float* v = (float*)malloc(sizeof(float) * std::max(malloc_size_v8, malloc_size_v4));
    float* uv = (float*)malloc(sizeof(float) * 8 * 8 * in_channels);

    memset(dst, 0, sizeof(float) * dst_h * dst_w * out_channels);

    for (int i = 0; i < dst_h / 6; ++i) {
        for (int j = 0; j < dst_w / 6; ++j) {
            data_trans_t8(v + j * 8 * 8 * in_channels, src + i * 6 * src_w + 6 * j, in_channels, src_h * src_w, src_w);
            fastnum::cpu::kernel::mul(uv, u8, v + j * 8 * 8 * in_channels, 8 * 8 * in_channels);
            uv_trans_t8(dst + i * 6 * dst_w + 6 * j, dst_w, uv, in_channels);
        }

        for (int j = 0; j < (dst_w % 6) / 2; ++j) {
            data_trans_t4(v + j * 4 * 4 * in_channels, src + (i * 6 + 0) * src_w + nw6 * 6 + j * 2, in_channels, src_h * src_w, src_w);
            fastnum::cpu::kernel::mul(uv, u4, v + j * 4 * 4 * in_channels, 4 * 4 * in_channels);
            uv_trans_t4(dst + (i * 6 + 0) * dst_w + nw6 * 6 + 2 * j, dst_w, uv, in_channels);
        }

        for (int j = 0; j < (dst_w % 6) / 2; ++j) {
            data_trans_t4(v + j * 4 * 4 * in_channels, src + (i * 6 + 2) * src_w + nw6 * 6 + j * 2, in_channels, src_h * src_w, src_w);
            fastnum::cpu::kernel::mul(uv, u4, v + j * 4 * 4 * in_channels, 4 * 4 * in_channels);
            uv_trans_t4(dst + (i * 6 + 2) * dst_w + nw6 * 6 + 2 * j, dst_w, uv, in_channels);
        }

        for (int j = 0; j < (dst_w % 6) / 2; ++j) {
            data_trans_t4(v + j * 4 * 4 * in_channels, src + (i * 6 + 4) * src_w + nw6 * 6 + j * 2, in_channels, src_h * src_w, src_w);
            fastnum::cpu::kernel::mul(uv, u4, v + j * 4 * 4 * in_channels, 4 * 4 * in_channels);
            uv_trans_t4(dst + (i * 6 + 4) * dst_w + nw6 * 6 + 2 * j, dst_w, uv, in_channels);
        }
    }
    for (int i = 0; i < (dst_h % 6) / 2; ++i) {
        for (int j = 0; j < dst_w / 2; ++j) {
            data_trans_t4(v + j * 4 * 4 * in_channels, src + (nh6 * 6 + i * 2) * src_w + j * 2, in_channels, src_h * src_w, src_w);
            fastnum::cpu::kernel::mul(uv, u4, v + j * 4 * 4 * in_channels, 4 * 4 * in_channels);
            uv_trans_t4(dst + (nh6 * 6 + i * 2) * dst_w + 2 * j, dst_w, uv, in_channels);
        }
    }

    free(v);
    free(uv);
}



// static inline bool _use_wiograd(int kernel_h, int kernel_w, int stride_h, int stride_w) {
//     if (kernel_h == kernel_w && kernel_h == 3 && stride_h == stride_w && stride_h == 1)
//         return true;
//     return false;
// }

// Convolution2d::Convolution2d(int _in_channels,
//                              int _out_channels,
//                              int _kernel_h,
//                              int _kernel_w,
//                              int _stride_h,
//                              int _stride_w,
//                              int _pad_h,
//                              int _pad_w,
//                              int _use_bias)
//     : in_channels(_in_channels),
//       out_channels(_out_channels),
//       kernel_h(_kernel_h),
//       kernel_w(_kernel_w),
//       stride_h(_stride_h),
//       stride_w(_stride_w),
//       pad_h(_pad_h),
//       pad_w(_pad_w),
//       use_bias(_use_bias),
//       weight(nullptr),
//       bias(nullptr) {
//     weight = (float*)malloc(sizeof(float) * _in_channels * _out_channels * _kernel_h * _kernel_w);
//     if (_use_bias) {
//         bias = (float*)malloc(sizeof(float) * _out_channels);
//     }
//     use_wiograd = _use_wiograd(_kernel_h, _kernel_w, _stride_h, _stride_w);
// }

// void Convolution2d::setBias(const float* _b) {
//     if (use_bias) {
//         memcpy(bias, _b, sizeof(float) * out_channels);
//     }
// }

// void Convolution2d::setWeight(const float* _w) {
//     if (use_wiograd) {
//         WinogradWeightTransformByChannel<>
//     }
// }

// void Convolution2d::setConvWeightAndBias(const float* _w, const float* _b) {
//     this->setWeight(_w);
//     this->setBias(_b);
// }

}  // namespace cpu
}  // namespace fastnum