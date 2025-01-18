cmake_minimum_required(VERSION 3.10)  # Minimale Version von CMake
project(SQUID)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Speedup with -O3 -ffast-math -march=native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Bibliotheksverzeichnis für FFTW3, falls nicht standard
link_directories(/usr/lib/x86_64-linux-gnu)

find_package(TIFF REQUIRED)
find_package(OpenCV REQUIRED)
# Findet die FFTW3-Bibliothek
# Manuell setzen der Include- und Bibliothekspfade
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
# Finde FFTW3
#find_package(FFTW3 REQUIRED)
# Finden Sie OpenMP
find_package(OpenMP REQUIRED)


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TIFF_INCLUDE_DIR})

add_executable(squid ${CMAKE_SOURCE_DIR}/src/main.cpp
        src/hyperstack/HyperstackImage.cpp
        include/HyperstackImage.h
        src/hyperstack/Image3D.cpp
        include/Image3D.h
        src/hyperstack/ImageMetaData.cpp
        include/ImageMetaData.h
        src/psf/PSF.cpp
        include/PSF.h
        include/BasePSFGeneratorAlgorithm.h
        include/SimpleGaussianPSFGeneratorAlgorithm.h
        include/PSFGenerator.h
        include/DeconvolutionAlgorithm.h
        include/RLDeconvolutionAlgorithm.h
        src/deconvolution/DeconvolutionConfig.cpp
        include/DeconvolutionConfig.h
        include/BaseDeconvolutionAlgorithm.h
        src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
        src/hyperstack/HyperstackIO.cpp
        src/hyperstack/HyperstackMetaData.cpp
        include/UtlFFT.h
        src/utilities/UtlFFT.cpp
        include/UtlGrid.h
        src/utilities/UtlGrid.cpp
        include/UtlImage.h
        src/utilities/UtlImage.cpp
        include/UtlIO.h
        src/utilities/UtlIO.cpp
        include/Channel.h
        src/deconvolution/BaseDeconvolutionAlgorithm.cpp
        include/GaussianPSFGeneratorAlgorithm.h
        src/psf/algorithms/GaussianPSFGeneratorAlgorithm.cpp
        include/RLTVDeconvolutionAlgorithm.h
        src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
)
# Link your executable with the required libraries
target_link_libraries(squid
        fftw3
        fftw3_threads
        -lpthread
        -lgomp
        stdc++fs
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        ${TIFF_LIBRARIES}
)

# Testziel für Catch2 erstellen
add_executable(tests
        tests/UtlFFT_tests.cpp
        src/utilities/UtlFFT.cpp
        tests/UtlImage_tests.cpp
        src/utilities/UtlImage.cpp

)
include_directories(${CMAKE_SOURCE_DIR}/lib/catch2/include)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tests
        fftw3
        fftw3_threads
        -lpthread
        -lgomp
        stdc++fs
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        ${TIFF_LIBRARIES}
)
# Catch2 und deine Bibliothek verlinken
# Testziel aktivieren
enable_testing()
add_test(NAME UnitTests COMMAND tests)