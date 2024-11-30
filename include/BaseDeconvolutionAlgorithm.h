#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "PSF.h"
#include <fftw3.h>

class BaseDeconvolutionAlgorithm {
public:
    virtual ~BaseDeconvolutionAlgorithm(){cleanup();}
    virtual void configure(const DeconvolutionConfig& config) = 0;
    virtual void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) = 0;

    Hyperstack deconvolve(Hyperstack& data, PSF& psf);
    bool preprocess(Channel& channel, PSF& psf);
    bool postprocess(double epsilon);
    void cleanup();

protected:
    // Configuration
    double epsilon;
    int borderType;
    int psfSafetyBorder;
    int cubeSize;

    // Image handling and fftw
    std::vector<cv::Mat> mergedVolume;
    std::vector<std::vector<cv::Mat>> gridImages;
    fftw_plan forwardPlan, backwardPlan = nullptr;
    fftw_complex *paddedH, *fftwPlanMem = nullptr;

    // Image info
    int originalImageWidth, originalImageHeight, originalImageDepth;

    // Grid/cube arrangement
    int totalGridNum = 1;
    int cubeVolume, cubeWidth, cubeHeight, cubeDepth, cubePadding;
};
