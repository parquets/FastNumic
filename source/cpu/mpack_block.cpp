#include "mpack_block.hpp"

namespace fastnum {
namespace cpu {

void mpack_block_h8(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int x = 0;
    for (x = 0; x < xc - 7; x += 8) {
        kernel::mpack_h8(data, X, ldx, yc, max_pack);
        data += (8 * yc);
        X += (8 * ldx);
    }
    if (x < xc) {
        kernel::mpack_hx(data, X, ldx, (xc - x), n_pack, max_pack);
        data += (xc - x) * ldx;
    }
}

void mpack_block_h6(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int x = 0;
    for (x = 0; x < xc - 5; x += 6) {
        kernel::mpack_h6(data, X, ldx, yc, max_pack);
        data += (6 * yc);
        X += (6 * ldx);
    }
    if (x < xc) {
        kernel::mpack_hx(data, X, ldx, (xc - x), n_pack, max_pack);
    }
}

void mpack_block_h16(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int x = 0;
    for (x = 0; x < xc - 15; x += 16) {
        kernel::mpack_h16(data, X, ldx, yc, max_pack);
        data += (16 * yc);
        X += (16 * ldx);
    }
    if (x < xc) {
        kernel::mpack_hx(data, X, ldx, (xc - x), n_pack, max_pack);
    }
}

void mpack_block_v8(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int y = 0;
    for (y = 0; y < yc - 7; y += 8) {
        kernel::mpack_v8(data, X, ldx, xc, max_pack);
        data += (8 * xc);
        X += 8;
    }
    if (y < yc) {
        kernel::mpack_vx(data, X, ldx, (yc - y), n_pack, max_pack);
    }
}

void mpack_block_v6(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int y = 0;
    for (y = 0; y < yc - 5; y += 6) {
        kernel::mpack_v6(data, X, ldx, xc, max_pack);
        data += (6 * xc);
        X += 6;
    }
    if (y < yc) {
        kernel::mpack_vx(data, X, ldx, (yc - y), n_pack, max_pack);
    }
}

void mpack_block_v16(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    int y = 0;
    for (y = 0; y < yc - 15; y += 16) {
        kernel::mpack_v16(data, X, ldx, xc, max_pack);
        data += (16 * xc);
        X += 16;
    }
    if (y < yc) {
        kernel::mpack_vx(data, X, ldx, (yc - y), n_pack, max_pack);
    }
}

void mpack_block_h1(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    for (int x = 0; x < xc; ++x) {
        kernel::mpack_h1(data, X, ldx, n_pack, max_pack);
        data += xc;
        X += ldx;
    }
}

void mpack_block_v1(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    for (int y = 0; y < yc; ++y) {
        kernel::mpack_v1(data, X, ldx, n_pack, max_pack);
        data += xc;
        X += 1;
    }
}

void mpack_h(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    if constexpr (MR == 8)
        mpack_block_h8(xc, yc, data, X, ldx, n_pack, max_pack);
    else if constexpr (MR == 6)
        mpack_block_h6(xc, yc, data, X, ldx, n_pack, max_pack);
    else if constexpr (MR == 16)
        mpack_block_h16(xc, yc, data, X, ldx, n_pack, max_pack);
}

void mpack_v(int xc, int yc, float* data, const float* X, int ldx, int n_pack, int max_pack) {
    if constexpr (NR == 8)
        mpack_block_v8(xc, yc, data, X, ldx, n_pack, max_pack);
    else if constexpr (NR == 6)
        mpack_block_v6(xc, yc, data, X, ldx, n_pack, max_pack);
    else if constexpr (NR == 16)
        mpack_block_v16(xc, yc, data, X, ldx, n_pack, max_pack);
}

}  // namespace cpu
}  // namespace fastnum