cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(quantize LANGUAGES C)

set(PNG_SOURCES 
blur.c
blur.h 
kmeans.c
kmeans.h 
mediancut.c
mediancut.h 
nearest.c
nearest.h 
pam.c
pam.h
libimagequant.c
libimagequant.h
mempool.c 
mempool.h
)

add_library(quantize STATIC ${PNG_SOURCES})

target_include_directories(quantize PUBLIC ./)
