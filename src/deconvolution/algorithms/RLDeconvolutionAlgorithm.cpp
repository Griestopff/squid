#include "RLDeconvolutionAlgorithm.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fftw3.h>
#include "UtlFFT.h"
#include "UtlGrid.h"
#include "UtlImage.h"
#include <omp.h>

void RLDeconvolutionAlgorithm::configure(const DeconvolutionConfig& config) {
    // Algorithm specific
    this->iterations = config.iterations;

    // General
    this->epsilon = config.epsilon;

    // Grid
    this->borderType = config.borderType;
    this->psfSafetyBorder = config.psfSafetyBorder;
    this->cubeSize = config.cubeSize;

    // Output
    std::cout << "[CONFIGURATION] Richardson-Lucy algorithm" << std::endl;
    std::cout << "[CONFIGURATION] iterations: " << this->iterations << std::endl;
    std::cout << "[CONFIGURATION] epsilon: " << this->epsilon << std::endl;
    std::cout << "[CONFIGURATION] borderType: " << this->borderType << std::endl;
    std::cout << "[CONFIGURATION] psfSafetyBorder: " << this->psfSafetyBorder << std::endl;
    std::cout << "[CONFIGURATION] cubeSize: " << this->cubeSize << std::endl;

}

void RLDeconvolutionAlgorithm::algorithm(Hyperstack &data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) {

        // Allocate memory for intermediate FFTW arrays
        fftw_complex *c = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        std::memcpy(f, g, sizeof(fftw_complex) * this->cubeVolume);

        for (int n = 0; n < this->iterations; ++n) {
            std::cout << "\r[STATUS] Channel: " << channel_num + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " Iteration: " << n + 1 << "/"
                      << this->iterations << " ";

            // a) First transformation:
            // Fn = FFT(fn)
            fftw_execute_dft(this->forwardPlan, f, f);

            // Fn' = Fn * H
            UtlFFT::complexMultiplication(f, H, c, this->cubeVolume);

            // fn' = IFFT(Fn')
            fftw_execute_dft(this->backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // b) Calculation of the Correction Factor:
            // c = g / fn'
            // c = max(c, ε)
            UtlFFT::complexDivision(g, c, c, this->cubeVolume, this->epsilon);

            // c) Second transformation:
            // C = FFT(c)
            fftw_execute_dft(this->forwardPlan, c, c);

            // C' = C * conj(H)
            UtlFFT::complexMultiplicationWithConjugate(c, H, c, this->cubeVolume);

            // c' = IFFT(C')
            fftw_execute_dft(backwardPlan, c, c);
            UtlFFT::octantFourierShift(c, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            // d) Update the estimated image:
            // fn = IFFT(Fn)
            fftw_execute_dft(this->backwardPlan, f, f);

            // fn+1 = fn * c
            UtlFFT::complexMultiplication(f, c, f, this->cubeVolume);

            // Uncomment the following lines for debugging
            // UtlFFT::normalizeImage(f, size, this->epsilon);
            // UtlFFT::saveInterimImages(f, imageWidth, imageHeight, imageDepth, gridNum, channel_z, i);
            // Check size of number
            if (!(UtlImage::isValidForFloat(f, this->cubeVolume))) {
                std::cout << "[WARNING] Value fftwPlanMem fftcomplex(double) is smaller than float" << std::endl;
            }
            std::flush(std::cout);
        }
    fftw_free(c);

}
