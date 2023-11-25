#pragma once

#include "x86/dpack_x86.hpp"
#include "x86/spack_x86.hpp"

#define MR 6
#define NR 16

namespace fastnum {
namespace cpu {

namespace kernel {
inline void spack_hx(float* packed, const float* X, int ldx, int h, int n_pack, float scale = 1.0f) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < h; ++j)
            *packed++ = *(X + j * ldx) * scale;
        ++X;
    }
}

inline void spack_vx(float* packed, const float* X, int ldx, int v, int n_pack, float scale = 1.0f) {
    for (int i = 0; i < n_pack; ++i) {
        // memcpy(packed, X, sizeof(float) * v);
        for(int j=0; j < v; ++j) {
            packed[j] = X[j] * scale;
        }
        packed += v;
        X += ldx;
    }
}

inline void dpack_hx(double* packed, const double* X, int ldx, int h, int n_pack, double scale = 1.0) {
    for (int i = 0; i < n_pack; ++i) {
        for (int j = 0; j < h; ++j)
            *packed++ = *(X + j * ldx) * scale;
        ++X;
    }
}

inline void dpack_vx(double* packed, const double* X, int ldx, int v, int n_pack, double scale = 1.0) {
    for (int i = 0; i < n_pack; ++i) {
        for(int j=0; j < v; ++j) {
            packed[j] = X[j] * scale;
        }
        packed += v;
        X += ldx;
    }
}

}  // namespace kernel

void spackA_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, float scale=1.0f);
void spackA_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, float scale=1.0f);
void spackB_h(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, float scale=1.0f);
void spackB_v(int xc, int yc, float* packed_data, const float* X, int ldx, int n_pack, float scale=1.0f);

void dpackA_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack, double scale=1.0);
void dpackA_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack, double scale=1.0);
void dpackB_h(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack, double scale=1.0);
void dpackB_v(int xc, int yc, double* packed_data, const double* X, int ldx, int n_pack, double scale=1.0);

}  // namespace cpu
}  // namespace fastnum