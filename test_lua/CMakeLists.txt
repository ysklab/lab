cmake_minimum_required(VERSION 3.5)
project(test_lua)

set(CMAKE_CXX_STANDARD 17)
find_package(glog REQUIRED)
include_directories(/usr/include/lua5.2)
add_executable(test_lua main.cpp)
target_link_libraries(test_lua glog::glog lua5.2)
