#pragma once

namespace fastnum {
namespace cpu {

void bilinear_resize(float* dest,
                     const float* source,
                     int in_channels,
                     int in_h,
                     int in_w,
                     int out_h,
                     int out_w,
                     bool center_align);

void nearest_resize(float* dest,
                    const float* source,
                    int in_channels,
                    int in_h,
                    int in_w,
                    int out_h,
                    int out_w,
                    bool center_align);

}
}