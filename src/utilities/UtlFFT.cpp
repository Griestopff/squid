#include "UtlFFT.h"
#include "Image3D.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "omp.h"

// Perform point-wise complex multiplication
void UtlFFT::complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) {
    // Ensure that input pointers are not null
    if (!a || !b || !result) {
        std::cerr << "Error: Null pointer passed to complexMultiplication." << std::endl;
        return;
    }

    // Parallelize the loop with OpenMP
/*#pragma omp parallel
    {
        // Get the thread number and total number of threads
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Calculate the chunk size for each thread
        int chunk_size = size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? size : start + chunk_size;

        // Perform complex multiplication for the assigned chunk
        for (int i = start; i < end; ++i) {
            double real_a = a[i][0];
            double imag_a = a[i][1];
            double real_b = b[i][0];
            double imag_b = b[i][1];

            // Perform the complex multiplication and store the result
            result[i][0] = real_a * real_b - imag_a * imag_b;
            result[i][1] = real_a * imag_b + imag_a * real_b;
        }
    }*/
// Perform complex multiplication for each element in a sequential manner
#pragma omp parallel for simd shared(a, b, result)
    for (int i = 0; i < size; ++i) {
        double real_a = a[i][0];
        double imag_a = a[i][1];
        double real_b = b[i][0];
        double imag_b = b[i][1];

        // Perform the complex multiplication and store the result
        result[i][0] = real_a * real_b - imag_a * imag_b;
        result[i][1] = real_a * imag_b + imag_a * real_b;
    }
}

// Perform point-wise complex multiplication with conjugation
void UtlFFT::complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size) {
    // Ensure that input pointers are not null
    if (!a || !b || !result) {
        //std::cerr << "Error: Null pointer passed to complexMultiplicationWithConjugate." << std::endl;
        return;
    }

    // Parallelize the loop with OpenMP
#pragma omp parallel shared(a, b, result)
    {
        // Get the thread number and total number of threads
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Calculate the chunk size for each thread
        int chunk_size = size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? size : start + chunk_size;

        // Perform complex multiplication with conjugation for the assigned chunk
        for (int i = start; i < end; ++i) {
            double real_a = a[i][0];
            double imag_a = a[i][1];
            double real_b = b[i][0];
            double imag_b = -b[i][1];  // Conjugate the imaginary part

            // Perform the complex multiplication with conjugate and store the result
            result[i][0] = real_a * real_b - imag_a * imag_b; // Real part
            result[i][1] = real_a * imag_b + imag_a * real_b; // Imaginary part
        }
    }
}
// Perform point-wise complex division (min->epsilon)
void UtlFFT::complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) {
    // Ensure that input pointers are not null
    if (!a || !b || !result) {
        std::cerr << "Error: Null pointer passed to complexDivision." << std::endl;
        return;
    }

    // Parallelize the loop with OpenMP
#pragma omp parallel shared(a, b, result)
    {
        // Get the thread number and total number of threads
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Calculate the chunk size for each thread
        int chunk_size = size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? size : start + chunk_size;

        // Perform complex division for the assigned chunk
        for (int i = start; i < end; ++i) {
            double real_a = a[i][0];
            double imag_a = a[i][1];
            double real_b = b[i][0];
            double imag_b = b[i][1];

            double denominator = real_b * real_b + imag_b * imag_b;

            if (denominator < epsilon) {
                result[i][0] = 0.0;
                result[i][1] = 0.0;
            } else {
                result[i][0] = (real_a * real_b + imag_a * imag_b) / denominator;
                result[i][1] = (imag_a * real_b - real_a * imag_b) / denominator;
            }
        }
    }
}

// Perform point-wise complex division with stabilization (min->epsilon)
void UtlFFT::complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon) {
    // Ensure that input pointers are not null
    if (!a || !b || !result) {
        std::cerr << "Error: Null pointer passed to complexDivisionStabilized." << std::endl;
        return;
    }

    // Parallelize the loop with OpenMP
#pragma omp parallel shared(a, b, result)
    {
        // Get the thread number and total number of threads
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        // Calculate the chunk size for each thread
        int chunk_size = size / num_threads;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads - 1) ? size : start + chunk_size;

        // Perform stabilized complex division for the assigned chunk
        for (int i = start; i < end; ++i) {
            double real_a = a[i][0];  // Realteil von a
            double imag_a = a[i][1];  // Imaginärteil von a
            double real_b = b[i][0];  // Realteil von b
            double imag_b = b[i][1];  // Imaginärteil von b

            // Berechnung des Betrags von b, mit Stabilisierung durch epsilon
            double mag = std::max(epsilon, real_b * real_b + imag_b * imag_b);

            // Durchführung der stabilisierten Division
            result[i][0] = (real_a * real_b + imag_a * imag_b) / mag;  // Realteil des Ergebnisses
            result[i][1] = (imag_a * real_b - real_a * imag_b) / mag;  // Imaginärteil des Ergebnisses
        }
    }
}


// Perform quadrant shift on FFTW complex array
void UtlFFT::octantFourierShift(fftw_complex* data, int width, int height, int depth) {
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    // Parallelize the nested loops using OpenMP with collapsing to reduce overhead
#pragma omp parallel for collapse(3) shared(data, width, height, depth)
    for (int z = 0; z < halfDepth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate the indices for the swap
                int idx1 = z * height * width + y * width + x;
                int idx2 = ((z + halfDepth) % depth) * height * width + ((y + halfHeight) % height) * width + ((x + halfWidth) % width);

                // Perform the swap only if the indices are different
                if (idx1 != idx2) {
                    // Swap real parts
                    double temp_real = data[idx1][0];
                    data[idx1][0] = data[idx2][0];
                    data[idx2][0] = temp_real;

                    // Swap imaginary parts
                    double temp_imag = data[idx1][1];
                    data[idx1][1] = data[idx2][1];
                    data[idx2][1] = temp_imag;
                }
            }
        }
    }
}



// Convert OpenCV Mat vector to FFTW complex array
void UtlFFT::convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth) {
    for (int z = 0; z < depth; ++z) {
        CV_Assert(input[z].type() == CV_32F);  // Ensure input is of type float
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                output[z * height * width + y * width + x][0] = static_cast<double>(input[z].at<float>(y, x));
                output[z * height * width + y * width + x][1] = 0.0;
            }
        }
    }
}
// Pad the PSF to the size of the image
void UtlFFT::padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth) {
    UtlFFT::octantFourierShift(psf, psf_width, psf_height, psf_depth);
    for (int i = 0; i < width * height * depth; ++i) {
        padded_psf[i][0] = 0.0;
        padded_psf[i][1] = 0.0;
    }

    if(psf_depth > depth){
        std::cerr << "[ERROR] PSF has more layers than image" << std::endl;
    }

    int x_offset = (width - psf_width) / 2;
    int y_offset = (height - psf_height) / 2;
    int z_offset = (depth - psf_depth) / 2;

    for (int z = 0; z < psf_depth; ++z) {
        for (int y = 0; y < psf_height; ++y) {
            for (int x = 0; x < psf_width; ++x) {
                int padded_index = ((z + z_offset) * height + (y + y_offset)) * width + (x + x_offset);
                int psf_index = (z * psf_height + y) * psf_width + x;

                padded_psf[padded_index][0] = psf[psf_index][0];
                padded_psf[padded_index][1] = psf[psf_index][1];
            }
        }
    }
    UtlFFT::octantFourierShift(padded_psf, width, height, depth);
}

void UtlFFT::convertFFTWComplexToCVMatVector(const fftw_complex* input, std::vector<cv::Mat>& output, int width, int height, int depth) {
    std::vector<cv::Mat> tempOutput;
    for (int z = 0; z < depth; ++z) {
        cv::Mat result(height, width, CV_32F);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int index = z * height * width + y * width + x;
                result.at<float>(y, x) =
                        input[index][0] / (width * height * depth); // Normalize

            }
        }
        tempOutput.push_back(result);
    }

    output = tempOutput;
}




// Berechnet den Gradienten in x-Richtung eines 3D-Bildes.
void UtlFFT::gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth) {
    // Parallelize the loops using OpenMP for better performance.
//#pragma omp parallel for collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width - 1; ++x) {
                // Calculate the linear index for the 3D array.
                int index = z * height * width + y * width + x;
                int nextIndex = index + 1;

                // Derivative in x-direction: gradX = image - next value in x.
                gradX[index][0] = image[index][0] - image[nextIndex][0]; // Real part
                gradX[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
            }

            // Handle the boundary condition at the last x position.
            int lastIndex = z * height * width + y * width + (width - 1);
            gradX[lastIndex][0] = 0.0; // Set the derivative to zero or another suitable boundary condition.
            gradX[lastIndex][1] = 0.0;
        }
    }
}

// Berechnet den Gradienten in y-Richtung eines 3D-Bildes.
void UtlFFT::gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth) {
    // Parallelize the loops using OpenMP for better performance.
//#pragma omp parallel for collapse(3)
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height - 1; ++y) {  // y < height - 1 to avoid overflow
            for (int x = 0; x < width; ++x) {
                // Calculate the linear index for the 3D array.
                int index = z * height * width + y * width + x;
                int nextIndex = index + width;

                // Derivative in y-direction: gradY = image - next value in y.
                gradY[index][0] = image[index][0] - image[nextIndex][0]; // Real part
                gradY[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
            }
        }

        // Handle the boundary condition at the last y position (y == height - 1).
        for (int x = 0; x < width; ++x) {
            int lastIndex = z * height * width + (height - 1) * width + x;
            gradY[lastIndex][0] = 0.0; // Set the derivative to zero or another suitable boundary condition.
            gradY[lastIndex][1] = 0.0;
        }
    }
}

// Berechnet den Gradienten in z-Richtung eines 3D-Bildes.
void UtlFFT::gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth) {
    // Parallelize the loops using OpenMP for better performance.
//#pragma omp parallel for collapse(3)
    for (int z = 0; z < depth - 1; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Calculate the linear index for the 3D array.
                int index = z * height * width + y * width + x;
                int nextIndex = index + height * width;

                // Derivative in z-direction: gradZ = image - next value in z.
                gradZ[index][0] = image[index][0] - image[nextIndex][0]; // Real part
                gradZ[index][1] = image[index][1] - image[nextIndex][1]; // Imaginary part
            }

            // Handle the boundary condition at the last z position.
            int lastIndex = (depth - 1) * height * width + y * width + (width - 1);
            gradZ[lastIndex][0] = 0.0; // Set the derivative to zero or another suitable boundary condition.
            gradZ[lastIndex][1] = 0.0;
        }
    }
}
// Berechnet die Total-Variation-Filterung basierend auf den Gradienten und einem Regularisierungsparameter lambda.
void UtlFFT::computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth) {
    int nxy = width * height;

    // Parallelize the loops using OpenMP for better performance.
//#pragma omp parallel for collapse(2)
    for (int z = 0; z < depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            // Calculate the linear index for the 3D array.
            int index = z * nxy + i;

            // Retrieve the gradient components.
            double dx = gx[index][0]; // Assume that the gradient data is stored in the real part.
            double dy = gy[index][0];
            double dz = gz[index][0];

            // Compute the TV value using the provided formula.
            tv[index][0] = static_cast<float>(1.0 / ((dx + dy + dz) * lambda + 1.0));
            tv[index][1] = 0.0; // Assuming the output is real-valued, set the imaginary part to zero.
        }
    }
}

// Normalisiert die Vektorkomponenten eines 3D-Feldes.
void UtlFFT::normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon) {
    int nxy = width * height;

    // Parallelize the loops using OpenMP for better performance.
//#pragma omp parallel for collapse(2)
    for (int z = 0; z < depth; ++z) {
        for (int i = 0; i < nxy; ++i) {
            int index = z * nxy + i;

            // Berechne die Norm (Länge) des Vektors.
            double norm = std::sqrt(
                    gradX[index][0] * gradX[index][0] + gradX[index][1] * gradX[index][1] +
                    gradY[index][0] * gradY[index][0] + gradY[index][1] * gradY[index][1] +
                    gradZ[index][0] * gradZ[index][0] + gradZ[index][1] * gradZ[index][1]
            );

            // Vermeide Division durch zu kleine Werte, indem die Norm durch max(epsilon, norm) ersetzt wird.
            norm = std::max(norm, epsilon);

            // Normalisiere die Komponenten.
            gradX[index][0] /= norm;
            gradX[index][1] /= norm;
            gradY[index][0] /= norm;
            gradY[index][1] /= norm;
            gradZ[index][0] /= norm;
            gradZ[index][1] /= norm;
        }
    }
}

// Normalizing after the inverse FFT helps maintain correct scaling of the result,
// especially in iterative algorithms where accumulated scaling errors could otherwise
// lead to instability. However, in this case, the normalization/rescaling has no significant effect.
// maybe TODO as optional configuration, rescaledInverse yes/no
void UtlFFT::rescaledInverse(fftw_complex* data, double cubeVolume) {
    for (int i = 0; i < cubeVolume; ++i) {
        data[i][0] /= cubeVolume; // Realteil normalisieren
        data[i][1] /= cubeVolume; // Imaginärteil normalisieren
    }
}