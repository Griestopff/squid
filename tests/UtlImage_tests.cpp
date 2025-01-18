#include <../lib/catch2/include/catch.hpp>
#include "../include/UtlImage.h"
#include <fftw3.h>
#include <limits>
#include <vector>
#include <iostream>


TEST_CASE("UtlImage::isValidForFloat", "[UtlImage]") {
    double float_min = std::numeric_limits<float>::min();
    double float_max = 3.4e38;

    SECTION("All values within the valid range") {
        size_t size = 10;
        std::vector<fftw_complex> fftwData(size);

        // Initialize all values within the valid range
        for (size_t i = 0; i < size; ++i) {
            fftwData[i][0] = 0.0; // Real part
            fftwData[i][1] = 0.0; // Imaginary part
        }

        REQUIRE(UtlImage::isValidForFloat(fftwData.data(), size) == true);
    }

    SECTION("One value out of range (real part too large)") {
        size_t size = 10;
        std::vector<fftw_complex> fftwData(size);

        // Initialize all values within the valid range
        for (size_t i = 0; i < size; ++i) {
            fftwData[i][0] = static_cast<double>(0.0); // Real part
            fftwData[i][1] = static_cast<double>(0.0); // Imaginary part
        }

        fftwData[5][0] = 1e39;

        REQUIRE(UtlImage::isValidForFloat(fftwData.data(), size) == false);
    }

    SECTION("One value out of range (imaginary part too small)") {
        size_t size = 10;
        std::vector<fftw_complex> fftwData(size);

        // Initialize all values within the valid range
        for (size_t i = 0; i < size; ++i) {
            fftwData[i][0] = 0.0; // Real part
            fftwData[i][1] = 0.0; // Imaginary part
        }
        // Set one value out of the valid range
        fftwData[3][1] = 1e-40;

        REQUIRE(UtlImage::isValidForFloat(fftwData.data(), size) == false);
    }

    SECTION("Boundary values (valid)") {
        size_t size = 2;
        std::vector<fftw_complex> fftwData(size);

        // Set values exactly at the boundary limits
        fftwData[0][0] = float_min;
        fftwData[0][1] = float_max;
        fftwData[1][0] = float_max;
        fftwData[1][1] = float_min;

        REQUIRE(UtlImage::isValidForFloat(fftwData.data(), size) == true);
    }

    SECTION("Empty array") {
        size_t size = 0;
        fftw_complex* fftwData = nullptr;

        // An empty array is always valid
        REQUIRE(UtlImage::isValidForFloat(fftwData, size) == true);
    }
}
