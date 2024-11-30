#include "HyperstackImage.h"
#include "UtlFFT.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fftw3.h>


Hyperstack::Hyperstack() = default;
Hyperstack::Hyperstack(const Hyperstack &rhs) : channels{rhs.channels}, metaData{rhs.metaData} {}
Hyperstack::~Hyperstack() = default;

bool Hyperstack::showSlice(int channel, int z) {
    if (!this->channels[channel].image.slices.empty()) {
        if(z < 0 || z > this->metaData.totalImages){
            std::cerr << "Slice " << std::to_string(z) << " out of Range in Channel " << std::to_string(channel) << std::endl;
            return false;
        }else {
            cv::Mat& slice = channels[channel].image.slices[z];

            if (slice.type() != CV_32F) {
                std::cerr << "Expected CV_32F data type." << std::endl;
                return false;
            }
            if (slice.empty()) {
                std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
                return false;
            }

            // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
            cv::Mat img8bit;
            double minVal, maxVal;
            cv::minMaxLoc(slice, &minVal, &maxVal);

            // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
            // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
            slice.convertTo(img8bit, CV_8U, 255.0 / maxVal);
            cv::imshow("Slice " + std::to_string(z) + " in Channel " + std::to_string(channel), img8bit);
            cv::waitKey();

            std::cout << "Slice " + std::to_string(z) + " in Channel " + std::to_string(channel) + " shown" << std::endl;
            return true;
        }
    }else{
        std::cerr<< "Layer " <<  std::to_string(z) << " cannot shown" << std::endl;
        return false;
    }
}
bool Hyperstack::showChannel(int channel){
    for (const auto &mat : this->channels[channel].image.slices) {
            if (mat.type() != CV_32F) {
                std::cerr << "Expected CV_32F data type." << std::endl;
                continue;
            }
            if (mat.empty()) {
                std::cerr << "Layer data is empty or could not be retrieved." << std::endl;
                continue;            }

            // Erstellen eines neuen leeren Bildes für das 8-Bit-Format
        cv::Mat img8bit;
        double minVal, maxVal;
        cv::minMaxLoc(mat, &minVal, &maxVal);

        // Konvertieren des 32-Bit-Fließkommabildes in ein 8-Bit-Bild
        // Die Pixelwerte werden von [0.0, 1.0] auf [0, 255] skaliert
        mat.convertTo(img8bit, CV_8U, 255.0 / maxVal);
        cv::imshow("Channel " + std::to_string(channel), img8bit);
        cv::waitKey();
    }

    return true;
}

float Hyperstack::getPixel(int channel, int x, int y, int z) {
    return this->channels[channel].image.slices[z].at<float>(x,y);
}




