#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "transpose.cuh"

namespace fastnum {
namespace cuda {
namespace kernel {


__global__ void transpose(int M, int N, const float* A, int lda, float* B, int ldb) {

    __shared__ float local[BLOCK][BLOCK + 1];
    
    int bx = blockIdx.x * BLOCK;
    int by = blockIdx.y * BLOCK;
        
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    
    const float* blockA = A + by * lda + bx;
    float* blockB = B + bx * ldb + by;
    
    if (bx + tx < N && by + ty < M) {
        local[ty][tx] = blockA[ty * lda + tx];
    }
    
    __syncthreads();
    
    
    if (by + tx < M && bx + ty < N) {
        blockB[ty * ldb + tx] = local[tx][ty];
    }
    
}
    
}

void transpose_cuda_wrap(int M, int N, const float* A, int lda, float* B, int ldb) {
    dim3 bdim(BLOCK, BLOCK);
    dim3 gdim((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    kernel::transpose <<<gdim, bdim>>> (M, N, A, lda, B, ldb);
}

}
}