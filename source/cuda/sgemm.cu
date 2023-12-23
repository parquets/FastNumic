#include "sgemm.cuh"
#include <chrono>

namespace fastnum {
namespace cuda {
namespace kernel {

// #define SMEM_LDA (128)
// #define SMEM_LDB (128)

__global__ void sgemm_nn(int M, int N, int K, 
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {

    __shared__ float a_shared[128 * 8];
    __shared__ float b_shared[128 * 8];

    const float* a_ptr = A + blockIdx.y * 128 * lda;
    const float* b_ptr = B + blockIdx.x * 128;


    int sac = (threadIdx.x / 8) * 4; 
    int sar = (threadIdx.x % 8);
    int sbr = threadIdx.x / 32;
    int sbc = threadIdx.x % 32;


    float a_penal[8] = {0.0f};
    float b_penal[8] = {0.0f};
    float c_penal[8][8] = {0.0f};

    for(int k=0; k < K - 7; k+=8) {
#pragma unroll
        for(int x=0; x < 4; ++x) {
            a_shared[sar * 128 + sac + x] = alpha * a_ptr[(sac + x) * lda + sar];
        } 
#pragma unroll
        for(int x=0; x < 4; ++x) {
            b_shared[sbr * 128 + sbc + x * 32]  = b_ptr[sbr * ldb + sbc + x * 32];
        }

        __syncthreads();

        a_ptr += 8;
        b_ptr += 8*ldb;

        int apc = (threadIdx.x / 16) * 4;
        int bpc = (threadIdx.x % 16) * 4;
#pragma unroll
        for(int subk = 0; subk < 8; ++subk) {
            int skc = subk * 128;
#pragma unroll
            for(int x=0; x < 4; ++x) {
                a_penal[x] = a_shared[skc + apc + x]; 
                a_penal[x + 4] = a_shared[skc + apc + x + 64]; 
            }
#pragma unroll
            for(int x=0; x < 4; ++x) {
                b_penal[x] = b_shared[skc + bpc + x];
                b_penal[x + 4] = b_shared[skc + bpc + x + 64];
            }

#pragma unroll
            for(int y=0; y < 8; ++y) {
#pragma unroll
                for(int x=0; x < 8; ++x) {
                    c_penal[y][x] += a_penal[y] * b_penal[x];
                }
            } 
        }
        __syncthreads();
    }


    float* c_ptr = C + (blockIdx.y * 128 +  (threadIdx.x / 16) * 4) * ldc +
                    blockIdx.x * 128 + (threadIdx.x % 16) * 4;

#pragma unroll
    for(int y=0; y < 4; ++y) {
        for(int x=0; x < 4; ++x) {
            c_ptr[y * ldc + x]             = beta * c_ptr[y * ldc + x]             + c_penal[y][x];
            c_ptr[y * ldc + x + 64]        = beta * c_ptr[y * ldc + x + 64]        + c_penal[y][x+4];
            c_ptr[(y + 64) * ldc + x]      = beta * c_ptr[(y + 64) * ldc + x]      + c_penal[y+4][x];
            c_ptr[(y + 64) * ldc + x + 64] = beta * c_ptr[(y + 64) * ldc + x + 64] + c_penal[y+4][x + 4];
        }
    }
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
    
    kernel::sgemm_nn<<<gridDim, 256>>> (M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

}

} // namespace cuda
} // namespace fastnum