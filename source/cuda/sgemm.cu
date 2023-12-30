#include "sgemm.cuh"
#include <chrono>

namespace fastnum {
namespace cuda {
namespace kernel {

// #define SMEM_LDA (128)
// #define SMEM_LDB (128)

// __global__ void sgemm_nn(int M, int N, int K, 
//                          float alpha,
//                          const float* A, int lda,
//                          const float* B, int ldb,
//                          float beta,
//                          float* C, int ldc) {

//     __shared__ float a_shared[128 * 8];
//     __shared__ float b_shared[128 * 8];

//     const float* a_ptr = A + blockIdx.y * 128 * lda;
//     const float* b_ptr = B + blockIdx.x * 128;


//     int sac = (threadIdx.x / 8) * 4; 
//     int sar = (threadIdx.x % 8);
//     int sbr = threadIdx.x / 32;
//     int sbc = threadIdx.x % 32;


//     float a_penal[8] = {0.0f};
//     float b_penal[8] = {0.0f};
//     float c_penal[8][8] = {0.0f};

//     for(int k=0; k < K; k+=8) {
// #pragma unroll
//         for(int x=0; x < 4; ++x) {
//             if (blockIdx.y * 128 + sac + x < M && sar + k < K) {
//                 a_shared[sar * 128 + sac + x] = alpha * a_ptr[(sac + x) * lda + sar];
//             } else {
//                 a_shared[sar * 128 + sac + x] = 0.0f;
//             }
//         } 
// #pragma unroll
//         for(int x=0; x < 4; ++x) {
//             if (sbr + k < K && sbc + x * 32 + blockIdx.x * 128 < N) {
//                 b_shared[sbr * 128 + sbc + x * 32]  = b_ptr[sbr * ldb + sbc + x * 32];
//             } else {
//                 b_shared[sbr * 128 + sbc + x * 32]  = 0.0f;
//             }
//         }

//         __syncthreads();

//         a_ptr += 8;
//         b_ptr += 8*ldb;

//         int apc = (threadIdx.x / 16) * 4;
//         int bpc = (threadIdx.x % 16) * 4;
// #pragma unroll
//         for(int subk = 0; subk < 8; ++subk) {
//             int skc = subk * 128;
// #pragma unroll
//             for(int x=0; x < 4; ++x) {
//                 a_penal[x] = a_shared[skc + apc + x]; 
//                 a_penal[x + 4] = a_shared[skc + apc + x + 64]; 
//             }
// #pragma unroll
//             for(int x=0; x < 4; ++x) {
//                 b_penal[x] = b_shared[skc + bpc + x];
//                 b_penal[x + 4] = b_shared[skc + bpc + x + 64];
//             }

// #pragma unroll
//             for(int y=0; y < 8; ++y) {
// #pragma unroll
//                 for(int x=0; x < 8; ++x) {
//                     c_penal[y][x] += a_penal[y] * b_penal[x];
//                 }
//             } 
//         }
//         __syncthreads();
//     }

//     int offset_y = blockIdx.y * 128 +  (threadIdx.x / 16) * 4;
//     int offset_x = blockIdx.x * 128 + (threadIdx.x % 16) * 4;
//     int c_offset = offset_y * ldc + offset_x;
//     float* c_ptr = C + c_offset;

// #pragma unroll
//     for(int y=0; y < 4; ++y) {
//         for(int x=0; x < 4; ++x) {
//             if(offset_y + y + 64 < M) {
//                 if (offset_x + x + 64 < N) {
//                     c_ptr[y * ldc + x]             = beta * c_ptr[y * ldc + x]             + c_penal[y][x];
//                     c_ptr[y * ldc + x + 64]        = beta * c_ptr[y * ldc + x + 64]        + c_penal[y][x+4];
//                     c_ptr[(y + 64) * ldc + x]      = beta * c_ptr[(y + 64) * ldc + x]      + c_penal[y+4][x];
//                     c_ptr[(y + 64) * ldc + x + 64] = beta * c_ptr[(y + 64) * ldc + x + 64] + c_penal[y+4][x + 4];
//                 } else {
//                     if (offset_x + x < N) {
//                         c_ptr[y * ldc + x]        = beta * c_ptr[y * ldc + x]        + c_penal[y][x];
//                         c_ptr[(y + 64) * ldc + x] = beta * c_ptr[(y + 64) * ldc + x] + c_penal[y+4][x];
//                     }
//                 }
//             } else {
//                 if (offset_y + y < M) {
//                     if (offset_x + x + 64 < N) {
//                         c_ptr[y * ldc + x]      = beta * c_ptr[y * ldc + x]      + c_penal[y][x];
//                         c_ptr[y * ldc + x + 64] = beta * c_ptr[y * ldc + x + 64] + c_penal[y][x+4];
//                     } else {
//                         if (offset_x + x < N) {
//                             c_ptr[y * ldc + x] = beta * c_ptr[y * ldc + x] + c_penal[y][x];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// } 

#define SMEM_LDA (132)
#define SMEM_LDB (128)
#define SMEM_LDC (64)

// remove original guard
__device__ __forceinline__ void ldg32_nc_0(float &reg, const void *ptr) {
    asm volatile("{.reg .pred p;\n"
               "mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 &&                 \
    __CUDA_ARCH__ >= 750
               "ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
               "ld.global.nc.f32 %0, [%1];}\n"
#endif
               : "=f"(reg)
               : "l"(ptr));
}

__device__ __forceinline__ uint32_t smem_u32addr(const void *smem_ptr) {
  uint32_t addr;
  asm("{.reg .u64 u64addr;\n"
      " cvta.to.shared.u64 u64addr, %1;\n"
      " cvt.u32.u64 %0, u64addr;}\n"
      : "=r"(addr)
      : "l"(smem_ptr));

  return addr;
}

__device__ __forceinline__ void lds128(float &reg0, float &reg1, float &reg2,
                                       float &reg3, const uint32_t &addr) {
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
               : "r"(addr));
}

__device__ __forceinline__ void sts128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const uint32_t &addr) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void stg128(const float &reg0, const float &reg1,
                                       const float &reg2, const float &reg3,
                                       const float *addr) {
  asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\n"
               :
               : "l"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3));
}

__device__ __forceinline__ void stg32(const float &reg, const void *ptr) {
  asm volatile("{.reg .pred p;\n"
               " st.global.f32 [%0], %1;}\n"
               :
               : "l"(ptr), "f"(reg));
}

__device__ __forceinline__ void sts32(const float &reg, const uint32_t &addr) {
  asm volatile("st.shared.f32 [%0], %1;\n" : : "r"(addr), "f"(reg));
}

// MY_MMult = [
// 1024 16561.20 7.247925e-05 
// 2048 18817.16 1.525879e-04 
// 3072 18516.94 2.288818e-04 
// 4096 18292.37 4.425049e-04 
// ];
/**
 * version 12 相对于  version  11, 增加 subk 计算中的 ping-pong
 */
__global__ __launch_bounds__(256, 2) void sgemm_128x128x8(int m, int n, int k,
                                                          const float *a,
                                                          const float *b,
                                                          float *c) {

  __shared__ __align__(
      16 * 1024) char smem[24 * 1024]; // 16KB shared memory for buffer

  float *ashare = reinterpret_cast<float *>(smem);
  float *bshare =
      reinterpret_cast<float *>(smem + 16 * 1024); // 8k shared mem for B
  float sum[8][8] = {0};
  float panelA[2][8] = {0}, panelB[2][8] = {0};

  int from_a = (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8;
  int from_b = (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32;

  float a_ldg_reg[4], b_ldg_reg[4];

  uint32_t a_sts_addr = smem_u32addr(ashare + (threadIdx.x % 8) * SMEM_LDA +
                                     (threadIdx.x / 8) * 4);
  uint32_t b_sts_addr =
      smem_u32addr(bshare + (threadIdx.x / 32) * SMEM_LDB + (threadIdx.x % 32));

  uint32_t aptr_base = smem_u32addr(ashare + (threadIdx.x / 16) * 4);
  uint32_t bptr_base = smem_u32addr(bshare + (threadIdx.x % 16) * 4);

  {
// load first
// load gmem to smem for ashare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(a_ldg_reg[i],
                 (const char *)(a + from_a) + i * k * sizeof(float));
    }
    sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3], a_sts_addr);

// load gmem to smem for bshare
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      ldg32_nc_0(b_ldg_reg[i],
                 (const char *)(b + from_b) + i * 32 * sizeof(float));
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sts32(b_ldg_reg[i], b_sts_addr + i * 32 * sizeof(float));
    }
    __syncthreads();
    // add offset and flip flag
    from_a += 8;
    from_b += 8 * n;

    a_sts_addr ^= 0x2000;
    b_sts_addr ^= 0x1000;
  }

  // load fisrt panel
  lds128(panelA[0][0], panelA[0][1], panelA[0][2], panelA[0][3], aptr_base);
  lds128(panelA[0][4], panelA[0][5], panelA[0][6], panelA[0][7],
         aptr_base + 64 * sizeof(float));

  lds128(panelB[0][0], panelB[0][1], panelB[0][2], panelB[0][3], bptr_base);
  lds128(panelB[0][4], panelB[0][5], panelB[0][6], panelB[0][7],
         bptr_base + 64 * sizeof(float));

  for (int loop = 0; loop < k; loop += 8) {
// calc
#pragma unroll
    for (int subk = 0; subk < 8; ++subk) {

      if (7 == subk && loop < k - 8) {
        // if have more, load next
        sts128(a_ldg_reg[0], a_ldg_reg[1], a_ldg_reg[2], a_ldg_reg[3],
               a_sts_addr);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
          sts32(b_ldg_reg[i], b_sts_addr + i * 32 * sizeof(float));
        }
        __syncthreads();
        from_a += 8;
        from_b += 8 * n;

        aptr_base ^= 0x2000;
        bptr_base ^= 0x1000;
        a_sts_addr ^= 0x2000;
        b_sts_addr ^= 0x1000;
      }

      const int pp = (subk + 1) % 2; // ping-pong index
      lds128(panelA[pp][0], panelA[pp][1], panelA[pp][2], panelA[pp][3],
             aptr_base + ((subk + 1) % 8) * SMEM_LDA * sizeof(float));
      lds128(panelA[pp][4], panelA[pp][5], panelA[pp][6], panelA[pp][7],
             aptr_base + (((subk + 1) % 8) * SMEM_LDA + 64) * sizeof(float));

      lds128(panelB[pp][0], panelB[pp][1], panelB[pp][2], panelB[pp][3],
             bptr_base + ((subk + 1) % 8) * SMEM_LDB * sizeof(float));
      lds128(panelB[pp][4], panelB[pp][5], panelB[pp][6], panelB[pp][7],
             bptr_base + (((subk + 1) % 8) * SMEM_LDB + 64) * sizeof(float));

      if (0 == subk && loop < k - 8) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ldg32_nc_0(a_ldg_reg[i],
                     (const char *)(a + from_a) + i * k * sizeof(float));
        }
        // load gmem to smem for bshare
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          ldg32_nc_0(b_ldg_reg[i],
                     (const char *)(b + from_b) + i * 32 * sizeof(float));
        }
      }

#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[i][j] += panelA[subk % 2][i] * panelB[subk % 2][j];
        }
      }
    }
  }

  int write_offset = (blockIdx.y * 128 + (threadIdx.x / 16) * 4) * n +
                     blockIdx.x * 128 + (threadIdx.x % 16) * 4;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    stg128(sum[i][0], sum[i][1], sum[i][2], sum[i][3],
           c + write_offset + i * n);
    stg128(sum[i][4], sum[i][5], sum[i][6], sum[i][7],
           c + write_offset + i * n + 64);
    stg128(sum[i + 4][0], sum[i + 4][1], sum[i + 4][2], sum[i + 4][3],
           c + write_offset + (i + 64) * n);
    stg128(sum[i + 4][4], sum[i + 4][5], sum[i + 4][6], sum[i + 4][7],
           c + write_offset + (i + 64) * n + 64);
  }
}

#undef SMEM_LDA
#undef SMEM_LDB


__global__ void sgemm_nt(int M, int N, int K, 
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) {
    
}

__global__ void sgemm_tn(int M, int N, int K, 
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {

}

__global__ void sgemm_tt(int M, int N, int K, 
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {
    
}


} // namespace kernel;


void sgemm_nn_cuda_wrap(int M, int N, int K, 
                        float alpha, 
                        const float* A, int lda, 
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) {
    
    constexpr int MB_MN = 128;

    int grid_size_m = (M + MB_MN - 1) / MB_MN;
    int grid_size_n = (N + MB_MN - 1) / MB_MN;
    dim3 gridDim(grid_size_m, grid_size_n);
    
    kernel::sgemm_128x128x8<<<gridDim, 256>>> (M, N, K, A, B, C);

}

} // namespace cuda
} // namespace fastnum