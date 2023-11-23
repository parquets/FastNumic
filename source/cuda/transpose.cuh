#pragma once

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace fastnum {
namespace cuda {

const int BLOCK = 32;

void transpose_cuda_wrap(int M, int N, const float* A, int lda, float* B, int ldb);


}
}