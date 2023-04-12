project(hello)
cuda_add_library(cu_lib simple.cu)
add_executable(hello main.cc)
target_link_libraries(hello cu_lib)