project(test_cpp_template)
set(CMAKE_CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.0)
add_executable(multi_template src/multi_template.cc)
add_executable(check_member src/check_member.cc)