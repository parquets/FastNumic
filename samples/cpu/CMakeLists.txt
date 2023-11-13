cmake_minimum_required(VERSION 3.12)

add_executable(sgemm_test sgemm_test.cpp)
add_executable(dgemm_test dgemm_test.cpp)
add_executable(gemv_test gemv_test.cpp)
add_executable(transpose_test transpose_test.cpp)

target_link_libraries(sgemm_test PRIVATE FastNum)
target_link_libraries(dgemm_test PRIVATE FastNum)
target_link_libraries(gemv_test PRIVATE FastNum)
target_link_libraries(transpose_test PRIVATE FastNum)

target_include_directories(sgemm_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_include_directories(dgemm_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_include_directories(gemv_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)
target_include_directories(transpose_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/include/fastnum)