cmake_minimum_required(VERSION 3.12)

set(FastNumCPUSource
    mma_block.cpp
    pack_block.cpp
    pooling_block.cpp
    # winograd_block.cpp
)

add_library(FastNumCPU STATIC ${FastNumCPUSource})
