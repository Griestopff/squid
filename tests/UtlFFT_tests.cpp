#define CATCH_CONFIG_MAIN
#include <../lib/catch2/include/catch.hpp>
#include "../include/UtlFFT.h"  // Header file defining the UtlFFT class and its functions

TEST_CASE("Elementwise complex matrix multiplication with UtlFFT::complexMultiplication") {
    fftw_complex input[4], input2[4], output[4];

    for (int i = 0; i < 4; i++) {
        input[i][0] = i + 1;  // real
        input[i][1] = i;      // img
        input2[i][0] = i + 1; // real
        input2[i][1] = i;     // img
    }
    UtlFFT::complexMultiplication(input, input2, output, 4);

    REQUIRE(output[0][0] == Approx(1 * 1 - 0 * 0).epsilon(1e-6));
    REQUIRE(output[0][1] == Approx(1 * 0 + 1 * 0).epsilon(1e-6));

    REQUIRE(output[1][0] == Approx(2 * 2 - 1 * 1).epsilon(1e-6));
    REQUIRE(output[1][1] == Approx(2 * 1 + 2 * 1).epsilon(1e-6));

    REQUIRE(output[2][0] == Approx(3 * 3 - 2 * 2).epsilon(1e-6));
    REQUIRE(output[2][1] == Approx(3 * 2 + 3 * 2).epsilon(1e-6));

    REQUIRE(output[3][0] == Approx(4 * 4 - 3 * 3).epsilon(1e-6));
    REQUIRE(output[3][1] == Approx(4 * 3 + 4 * 3).epsilon(1e-6));
}

TEST_CASE("Elementwise complex division with UtlFFT::complexDivision") {
    constexpr int size = 3;
    constexpr double epsilon = 1e-6;

    fftw_complex a[size] = {{4.0, 2.0}, {1.0, 3.0}, {0.0, 0.0}};
    fftw_complex b[size] = {{2.0, 1.0}, {4.0, 0.0}, {0.0, 0.0}};
    fftw_complex result[size];

    fftw_complex expected[size] = {
            {2.0, 0.0},    // (4 + 2i) / (2 + 1i) = 2 + 0i
            {0.25, 0.75},  // (1 + 3i) / (4 + 0i) = 0.25 + 0.75i
            {0.0, 0.0}     // (0 + 0i) / (epsilon + 0i) = 0 + 0i
    };

    UtlFFT::complexDivision(a, b, result, size, epsilon);

    for (int i = 0; i < size; ++i) {
        REQUIRE(result[i][0] == Approx(expected[i][0]).margin(1e-6));
        REQUIRE(result[i][1] == Approx(expected[i][1]).margin(1e-6));
    }
}

TEST_CASE("UtlFFT::complexMultiplicationWithConjugate performs correct computation") {
    int size = 3;
    fftw_complex a[3] = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    fftw_complex b[3] = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
    fftw_complex result[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

    fftw_complex expected[3] = {
            {1.0 * 7.0 - 2.0 * (-8.0), 1.0 * (-8.0) + 2.0 * 7.0},
            {3.0 * 9.0 - 4.0 * (-10.0), 3.0 * (-10.0) + 4.0 * 9.0},
            {5.0 * 11.0 - 6.0 * (-12.0), 5.0 * (-12.0) + 6.0 * 11.0}
    };

    UtlFFT::complexMultiplicationWithConjugate(a, b, result, size);

    for (int i = 0; i < size; ++i) {
        REQUIRE(result[i][0] == Approx(expected[i][0]).epsilon(1e-5));
        REQUIRE(result[i][1] == Approx(expected[i][1]).epsilon(1e-5));
    }
}

TEST_CASE("UtlFFT::complexMultiplicationWithConjugate handles null pointers gracefully") {
    fftw_complex* a = nullptr;
    fftw_complex* b = nullptr;
    fftw_complex* result = nullptr;

    REQUIRE_NOTHROW(UtlFFT::complexMultiplicationWithConjugate(a, b, result, 3));
}

TEST_CASE("UtlFFT::complexDivision handles division by zero gracefully") {
    constexpr int size = 1;
    constexpr double epsilon = 1e-6;

    fftw_complex a[size] = {{1.0, 1.0}};
    fftw_complex b[size] = {{0.0, 0.0}};
    fftw_complex result[size];

    fftw_complex expected[size] = {{0.0, 0.0}};

    UtlFFT::complexDivision(a, b, result, size, epsilon);

    REQUIRE(result[0][0] == Approx(expected[0][0]).margin(1e-6));
    REQUIRE(result[0][1] == Approx(expected[0][1]).margin(1e-6));
}

void initializeComplexArray(fftw_complex* array, int size, double realValue, double imagValue) {
    for (int i = 0; i < size; ++i) {
        array[i][0] = realValue;
        array[i][1] = imagValue;
    }
}

TEST_CASE("UtlFFT::gradientX calculates the x-direction gradient correctly") {
    constexpr int width = 4, height = 3, depth = 2;
    constexpr int size = width * height * depth;
    fftw_complex image[size], gradX[size];

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                image[index][0] = x;
                image[index][1] = 0.0;
            }
        }
    }

    UtlFFT::gradientX(image, gradX, width, height, depth);

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width - 1; ++x) {
                int index = z * height * width + y * width + x;
                REQUIRE(gradX[index][0] == Approx(-1.0));
                REQUIRE(gradX[index][1] == Approx(0.0));
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            int lastIndex = z * height * width + y * width + (width - 1);
            REQUIRE(gradX[lastIndex][0] == Approx(0.0).margin(1e-12));
            REQUIRE(gradX[lastIndex][1] == Approx(0.0).margin(1e-12));
        }
    }
}

TEST_CASE("UtlFFT::gradientY calculates the y-direction gradient correctly") {
    constexpr int width = 4, height = 3, depth = 2;
    constexpr int size = width * height * depth;
    fftw_complex image[size], gradY[size];

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                image[index][0] = y;
                image[index][1] = 0.0;
            }
        }
    }

    UtlFFT::gradientY(image, gradY, width, height, depth);

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height - 1; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                REQUIRE(gradY[index][0] == Approx(-1.0));
                REQUIRE(gradY[index][1] == Approx(0.0));
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x < width; ++x) {
            int lastIndex = z * height * width + (height - 1) * width + x;
            REQUIRE(gradY[lastIndex][0] == Approx(0.0).margin(1e-12));
            REQUIRE(gradY[lastIndex][1] == Approx(0.0).margin(1e-12));
        }
    }
}

TEST_CASE("UtlFFT::gradientZ calculates the z-direction gradient correctly") {
    constexpr int width = 4, height = 3, depth = 2;
    constexpr int size = width * height * depth;
    fftw_complex image[size], gradZ[size];

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                image[index][0] = z;
                image[index][1] = 0.0;
            }
        }
    }

    UtlFFT::gradientZ(image, gradZ, width, height, depth);

    for (int z = 0; z < depth - 1; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                REQUIRE(gradZ[index][0] == Approx(-1.0));
                REQUIRE(gradZ[index][1] == Approx(0.0));
            }
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int lastIndex = (depth - 1) * height * width + y * width + x;
            REQUIRE(gradZ[lastIndex][0] == Approx(0.0).margin(1e-12));
            REQUIRE(gradZ[lastIndex][1] == Approx(0.0).margin(1e-12));
        }
    }
}

TEST_CASE("UtlFFT::computeTV calculates total variation correctly") {
    constexpr int width = 4, height = 3, depth = 2;
    constexpr int size = width * height * depth;
    fftw_complex gx[size], gy[size], gz[size], tv[size];
    double lambda = 0.5;

    initializeComplexArray(gx, size, 1.0, 0.0);
    initializeComplexArray(gy, size, 1.0, 0.0);
    initializeComplexArray(gz, size, 1.0, 0.0);

    UtlFFT::computeTV(lambda, gx, gy, gz, tv, width, height, depth);

    for (int i = 0; i < size; ++i) {
        REQUIRE(tv[i][0] == Approx(1.0 / ((1.0 + 1.0 + 1.0) * lambda + 1.0)));
        REQUIRE(tv[i][1] == Approx(0.0));
    }
}

TEST_CASE("UtlFFT::computeTV handles zero gracefully") {
    constexpr int width = 4, height = 3, depth = 2;
    constexpr int size = width * height * depth;
    fftw_complex gx[size], gy[size], gz[size], tv[size];
    double lambda = 0.0;

    initializeComplexArray(gx, size, 0.0, 0.0);
    initializeComplexArray(gy, size, 0.0, 0.0);
    initializeComplexArray(gz, size, 0.0, 0.0);

    UtlFFT::computeTV(lambda, gx,gy, gz, tv, width, height, depth);

    for (int i = 0; i < size; ++i) {
        REQUIRE(tv[i][0] == Approx(1.0 / ((1.0 + 1.0 + 1.0) * lambda + 1.0)));
        REQUIRE(tv[i][1] == Approx(0.0));
    }
}