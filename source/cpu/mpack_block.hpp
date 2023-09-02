#pragma once

#include "x86/mmpack_kernel_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

namespace kernel {
inline void mpack_hx(float* packed, const float* X, int ldx, int h, int n_pack, int max_pack) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < h; ++j)
            *packed++ = *(X + j * ldx);
        ++X;
    }
}

inline void mpack_vx(float* packed, const float* X, int ldx, int v, int n_pack, int max_pack) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < v; ++j)
            *packed++ = *(X + j);
        X += ldx;
    }
}
}  // namespace kernel

void mpack_block_h8(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_h6(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_h16(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_v8(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_v6(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_v16(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_h1(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_block_v1(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_h(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

void mpack_v(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack);

}  // namespace cpu
}  // namespace fastnum