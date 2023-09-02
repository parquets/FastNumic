#include "mma_block.hpp"

namespace fastnum {
namespace cpu {

void mma_block_8x8(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {

    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;

    for (mm = 0; mm < mc - 7; mm += 8) {
        offset_b = 0;
        for (nn = 0; nn < nc - 7; nn += 8) {
            kernel::mma_8x8(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        if (nn < nc) {
            kernel::mma_8xn(nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
        offset_a += 8 * kc;
    }

    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc - 7; nn += 8) {
            kernel::mma_mx8(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        if (nn < nc) {
            kernel::mma_mxn(mc - mm, nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
    }
}

void mma_block_4x4(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {
    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;

    for (mm = 0; mm < mc - 3; mm += 4) {
        offset_b = 0;
        for (nn = 0; nn < nc - 3; nn += 4) {
            kernel::mma_4x4(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 4 * kc;
        }
        if (nn < nc) {
            kernel::mma_4xn(nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
        offset_a += 8 * kc;
    }

    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc - 3; nn += 4) {
            kernel::mma_mx4(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 8 * kc;
        }
        if (nn < nc) {
            kernel::mma_mxn(mc - mm, nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
    }
}

void mma_block_8x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {

    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;

    for (mm = 0; mm < mc - 7; mm += 8) {
        offset_b = 0;
        for (nn = 0; nn < nc; ++nn) {
            kernel::mma_8x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += 8 * kc;
    }

    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc; ++nn) {
            kernel::mma_mx1(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
    }
}

void mma_block_4x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {
    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;

    for (mm = 0; mm < mc - 3; mm += 4) {
        offset_b = 0;
        for (nn = 0; nn < nc; ++nn) {
            kernel::mma_4x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += 4 * kc;
    }

    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc; ++nn) {
            kernel::mma_mx1(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
    }
}

void mma_block_6x16(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {

    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;
    for (mm = 0; mm < mc - 5; mm += 6) {
        offset_b = 0;
        for (nn = 0; nn < nc - 15; nn += 16) {
            kernel::mma_6x16(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 16 * kc;
        }
        if (nn < nc) {
            kernel::mma_6xn(nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
        offset_a += 6 * kc;
    }

    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc - 15; nn += 16) {
            kernel::mma_mx16(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 16 * kc;
        }
        if (nn < nc) {
            kernel::mma_mxn(mc - mm, nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
    }
}

void mma_block_16x6(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {

    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;
    for (mm = 0; mm < mc - 15; mm += 16) {
        offset_b = 0;
        for (nn = 0; nn < nc - 5; nn += 6) {
            kernel::mma_16x6(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 6 * kc;
        }
        if (nn < nc) {
            kernel::mma_16xn(nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
        offset_a += 16 * kc;
    }
    if (mm < mc) {
        offset_b = 0;
        for (nn = 0; nn < nc - 5; nn += 6) {
            kernel::mma_mx6(mc - mm, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += 6 * kc;
        }
        if (nn < nc) {
            kernel::mma_mxn(mc - mm, nc - nn, packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
        }
    }
}

void mma_block_1x1(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {
    int offset_a = 0;
    int offset_b = 0;

    int mm = 0, nn = 0;
    for (mm = 0; mm < mc; ++mm) {
        offset_b = 0;
        for (nn = 0; nn < nc; ++nn) {
            kernel::mma_1x1(packA + offset_a, packB + offset_b, kc, C + mm * ldc + nn, ldc);
            offset_b += kc;
        }
        offset_a += kc;
    }
}

void mma_block(int mc, int nc, int kc, const float* packA, const float* packB, float* C, int ldc) {
    if constexpr (MR == 8 && NR == 8) {
        mma_block_8x8(mc, nc, kc, packA, packB, C, ldc);
    } else if constexpr (MR == 6 && NR == 16) {
        mma_block_6x16(mc, nc, kc, packA, packB, C, ldc);
    } else if constexpr (MR == 16 && NR == 6) {
        mma_block_16x6(mc, nc, kc, packA, packB, C, ldc);
    }
}

}  // namespace cpu
}  // namespace fastnum