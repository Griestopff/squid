#pragma once

#include <string>
#include <opencv2/core/base.hpp>
#include <vector>

class DeconvolutionConfig {
public:
    int iterations = 10;
    double epsilon = 1e-6;
    double lambda = 0.015;
    int borderType = cv::BORDER_REFLECT;
    int psfSafetyBorder = 10;
    int cubeSize = 0;

    void loadFromJSON(const std::string &directoryPath);
};



