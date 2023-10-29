cmake_minimum_required(VERSION 3.12)

add_executable(gemm_test gemm_test.cpp)
add_executable(gemv_test gemv_test.cpp)


target_link_libraries(gemm_test PRIVATE FastNum)
target_link_libraries(gemv_test PRIVATE FastNum)

target_include_directories(gemm_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_include_directories(gemv_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)