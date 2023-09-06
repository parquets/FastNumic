#pragma once

#include "x86/mpack_kernel_x86.hpp"

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


void h_packA(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, int max_pack);
void v_packA(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, int max_pack);
void h_packB(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, int max_pack);
void v_packB(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, int max_pack);

}  // namespace cpu
}  // namespace fastnum