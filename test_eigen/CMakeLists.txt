cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
project(test_eigen)
set(EIGEN_INCLUDE_DIRS /usr/include/eigen3)

include_directories(
        ${EIGEN_INCLUDE_DIRS}
)
add_executable(about_array src/about_array.cc)
add_executable(fast src/fast.cc)
#target_compile_definitions(fast PRIVATE -O2)