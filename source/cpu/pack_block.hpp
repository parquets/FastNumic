#pragma once

#include "x86/dpack_x86.hpp"
#include "x86/spack_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

namespace kernel {
inline void spack_hx(float* packed, const float* X, int ldx, int h, int n_pack) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < h; ++j)
            *packed++ = *(X + j * ldx);
        ++X;
    }
}

inline void spack_vx(float* packed, const float* X, int ldx, int v, int n_pack) {
    for (int i = 0; i < n_pack; ++i) {
        memcpy(packed, X, sizeof(float) * v);
        packed += v;
        X += ldx;
    }
}

inline void dpack_hx(double* packed, const double* X, int ldx, int h, int n_pack) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < h; ++j)
            *packed++ = *(X + j * ldx);
        ++X;
    }
}

inline void dpack_vx(double* packed, const double* X, int ldx, int v, int n_pack) {
    for (int i = 0; i < n_pack; ++i) {
        memcpy(packed, X, sizeof(double) * v);
        packed += v;
        X += ldx;
    }
}

}  // namespace kernel

void spackA_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack);
void spackA_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack);
void spackB_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack);
void spackB_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack);

void dpackA_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack);
void dpackA_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack);
void dpackB_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack);
void dpackB_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack);

}  // namespace cpu
}  // namespace fastnum