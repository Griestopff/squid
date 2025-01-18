cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(quantpng LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(./libs/lodepng)
add_subdirectory(./libs/quantize)

add_executable(quantpng main.cpp)

target_link_libraries(quantpng PRIVATE lodepng quantize)
