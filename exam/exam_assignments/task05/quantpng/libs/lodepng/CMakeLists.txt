cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(lodepng LANGUAGES CXX)

set(PNG_SOURCES 
lodepng.cpp
lodepng.h 
)

add_library(lodepng STATIC ${PNG_SOURCES})

target_include_directories(lodepng PUBLIC ./)
