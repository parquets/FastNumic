#include "layout.hpp"
#include <math.h>
#include <memory.h>
#include <stdlib.h>

namespace fastnum {
namespace cpu {

void nchw2nhwc(float *dst, const float *src, 
               int n, int ldn, 
               int channel, int ldc, 
               int height, int ldh, int width) {

    for (int in = 0; in < n; ++in) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int ic = 0; ic < channel; ++ic) {
                    *dst++ = src[in * ldn + ic * ldc + h * ldh + w];
                }
            }
        }
    }
}

void nhwc2nchw(float *dst, const float *src, 
               int n, int ldn, 
               int height, int ldh, 
               int width, int ldw, int channel) {

    for(int in=0; in<n; ++in) {
        for(int ic=0; ic<channel; ++ic) {
            for(int h=0; h<height; ++h) {
                for(int w=0; w<width; ++w) {
                    *dst++ = src[in*ldn+h*ldh+width*ldw+ic];
                }
            }
        }
    }
}




void nchw_pack(float *dst, const float *src, int n, int channel, int height, int width) {
    
}

}  // namespace cpu
}  // namespace fastnum