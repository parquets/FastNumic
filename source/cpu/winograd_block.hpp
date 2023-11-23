#pragma once

#include <math.h>
#include <stdio.h>
#include <chrono>
#include "x86/vectorize_x86.hpp"
#include "x86/winograd_dtrans_x86.hpp"
#include "x86/winograd_wtrans_x86.hpp"
#include "x86/winograd_uvtrans_x86.hpp"
#include "x86/arithmetic_x86.hpp"

namespace fastnum {
namespace cpu {



void convK3S1NaivePack1(float* dest, const float* source, const float* weight, int h_stride, int ch_stride);
void convK3S1NaivePack4(float* dest, const float* source, const float* weight, int h_stride, int ch_stride);
void convK3S1NaivePack8(float* dest, const float* source, const float* weight, int h_stride, int ch_stride);

void winogradConvK3S1Unit4Pack1(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit4Pack4(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit4Pack8(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit4Pack16(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);


void winogradConvK3S1Unit8Pack1(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit8Pack4(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit8Pack8(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Unit8Pack16(float* dest, 
                                float* u_buffer, 
                                float* v_buffer, 
                                float* uv_buffer,
                                const float* source, 
                                int src_h, int src_w,
                                int ldh, int ldc,
                                const float* weight,
                                int ldest);

void winogradConvK3S1Pack1(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols);

void winogradConvK3S1Pack4(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols);

void winogradConvK3S1Pack8(float* dest, 
                           float* u_buffer, 
                           float* v_buffer, 
                           float* uv_buffer,
                           const float* source, 
                           const float* weight, 
                           int rows, int cols);

void winogradDataTransSliceUnit8K3S1Pack1(float* dest, 
                                          const float* source, 
                                          int in_h, int in_w, 
                                          int ldh, int ldc);

void winogradDataTransSliceUnit8K3S1Pack4(float* dest, 
                                          const float* source, 
                                          int in_h, int in_w, 
                                          int ldh, int ldc);

void winogradDataTransSliceUnit8K3S1Pack8(float* dest, 
                                          const float* source, 
                                          int in_h, int in_w, 
                                          int ldh, int ldc);

void winogradUVTransSliceUnit8K3S1Pack1(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc);

void winogradUVTransSliceUnit8K3S1Pack4(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc);

void winogradUVTransSliceUnit8K3S1Pack8(float* dest, 
                                        int ldest,
                                        const float* source_v, 
                                        const float* source_u, 
                                        int in_h, int in_w, 
                                        int ldh, int ldc);


}
}