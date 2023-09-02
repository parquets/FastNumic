cmake_minimum_required(VERSION 3.12)

set(FastNumCPUSource
    mma_block.cpp
    mpack_block.cpp
)

add_library(FastNumCPU STATIC ${FastNumCPUSource})
