#include "sgemm.cuh"

namespace fastnum {
namespace cuda {
namespace kernel {


// threadDim.x = threadDim.y = MB_X * STRIDE_X
template<int MB_X, int STRIDE_X>
__global__ void sgemm_nn(int M, int N, int K, 
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {

    __shared__ float shared_A[MB_X][MB_X];
    __shared__ float shared_B[MB_X][MB_X];

    int offset_x = blockIdx.x * blockDim.x;
    int offset_y = blockIdx.y * blockDim.y;

    for(int i=0; i<STRIDE_X; ++i) {
        for(int j=0; j<STRIDE_X; ++j) {
            shared_A[threadIdx.y * STRIDE_X + i][threadIdx.x * STRIDE_X + j] = alpha * (A + (offset_y + threadIdx.y * STRIDE_X + i) * lda 
                                                                                 + (offset_x + threadIdx.x * STRIDE_X + j));
            shared_B[threadIdx.y * STRIDE_X + i][threadIdx.x * STRIDE_X + j] = B + (offset_y + threadIdx.x * STRIDE_X + j) * ldb 
                                                                                 + (offset_x + threadIdx.y * STRIDE_X + i);
        }
    }

    __syncthreads();

    float reg_result[STRIDE_X][STRIDE_X] = {0};

    int block_offset_x = threadIdx.x / STRIDE_X;
    int block_offset_y = threadIdx.y / STRIDE_X;

    for(int x = 0; x<STRIDE_X; ++x) {
        for(int y=0; y < STRIDE_X; ++y) {
            for(int i=0; i<MB_X; ++i) {
                reg_result[x][y] += shared_A[threadIdx.x + block_offset_x + x][i]*shared_B[threadIdx.y + block_offset_y + y][i]; 
            }
        }
    }

    for(int y = 0; y < STRIDE_X; ++y) {
        for(int x=0; x < STRIDE_X; ++x) {
            C[(offset_y+block_offset_y+y)*ldc + (offset_x+block_offset_x+x)] += reg_result[y][x];
        }
    }
} 

} // namespace kernel;
} // namespace cuda
} // namespace fastnum