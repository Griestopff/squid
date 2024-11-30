#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../lib/CLI/CLI11.hpp"
#include "../lib/nlohmann/json.hpp"

#include "HyperstackImage.h"
#include "PSF.h"
#include "PSFGenerator.h"
#include "GaussianPSFGeneratorAlgorithm.h"
#include "DeconvolutionAlgorithm.h"
#include "RLDeconvolutionAlgorithm.h"
#include "RLTVDeconvolutionAlgorithm.h"

using json = nlohmann::json;


int main(int argc, char** argv) {
    std::cout << "[Start SQUID]" << std::endl;

    // Arguments
    std::string image_path;
    std::string psf_path = "gauss";
    std::string algorithm = "rl";
    int iterations = 10; //RL and RLTV
    double lambda = 0.01; //RLTV
    double sigmax = 5.0; //gaussian PSF
    double sigmay = 5.0; //gaussian PSF
    double sigmaz = 5.0; //gaussian PSF
    int psfx = 20; //gaussian PSF width
    int psfy = 20; //gaussian PSF heigth
    int psfz = 40; //gaussian PSF depth/layers
    double epsilon = 1e-6; // complex divison
    bool time = false; //show time
    bool sep = false; //save layer separate as TIF dir
    bool savePsf = false; //save PSF
    bool showExampleLayers = false; //show random example layer of image and PSF
    bool printInfo = false; //show metadata of image
    int cubeSize = 0; //sub-image size (edge)
    int psfSafetyBorder = 10; //padding around PSF
    int borderType = cv::BORDER_REFLECT; //extension type of image


    CLI::App app{"SQUID - Super Quick Image Deconvolution"};
    // Define a group for CLI arguments
    CLI::Option_group *cli_group = app.add_option_group("CLI", "Commandline options");

    cli_group->add_option("-i,--image", image_path, "Input image Path")->required();
    cli_group->add_option("-p,--psf", psf_path, "Input PSF path or 'synthetic'")->required();
    cli_group->add_option("-a,--algorithm", algorithm, "Algorithm selection ('rl'/'rltv')");

    cli_group->add_option("--psfx", psfx, "PSF width for synthetic PSF  [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfy", psfy, "PSF heigth for synthetic PSF  [20]")->check(CLI::PositiveNumber);
    cli_group->add_option("--psfz", psfz, "PSF depth for synthetic PSF  [30]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmax", sigmax, "SigmaX for synthetic PSF [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmay", sigmay, "SigmaY for synthetic PSF  [5]")->check(CLI::PositiveNumber);
    cli_group->add_option("--sigmaz", sigmaz, "SigmaZ for synthetic PSF  [5]")->check(CLI::PositiveNumber);

    cli_group->add_option("--epsilon", epsilon, "Epsilon [1e-6] (for Complex Division)")->check(CLI::PositiveNumber);
    cli_group->add_option("--iterations", iterations, "Iterations [10] (for 'rl' and 'rltv')")->check(
            CLI::PositiveNumber);
    cli_group->add_option("--lambda", lambda, "Lambda regularization parameter [1e-2] (for 'rif' and 'rltv')");

    cli_group->add_option("--borderType", borderType,
                          "Border for extended image [2](0-constant, 1-replicate, 2-reflecting)")->check(
            CLI::PositiveNumber);
    cli_group->add_option("--psfSafetyBorder", psfSafetyBorder, "Padding around PSF [10]")->check(CLI::PositiveNumber);
    cli_group->add_option("--cubeSize", cubeSize,
                          "CubeSize/EdgeLength for sub-images of grid [0] (0-auto fit to PSF)")->check(
            CLI::PositiveNumber);

    cli_group->add_flag("--savepsf", time, "Save used PSF");
    cli_group->add_flag("--time", time, "Show duration active");
    cli_group->add_flag("--seperate", sep, "Save as TIF directory, each layer as single file");
    cli_group->add_flag("--info", printInfo, "Prints info about input Image");
    cli_group->add_flag("--showExampleLayers", showExampleLayers, "Shows a layer of loaded image and PSF)");

    // Define a group for configuration file
    CLI::Option_group *config_group = app.add_option_group("Config", "Configuration file");
    std::string config_file_path;
    config_group->add_option("-c,--config", config_file_path, "Path to configuration file")->required();

    // Exclude CLI arguments if configuration file is set
    cli_group->excludes(config_group);
    config_group->excludes(cli_group);

    CLI11_PARSE(app, argc, argv);

    json config;

    if (!config_file_path.empty()) {
        // Read configuration file
        std::ifstream config_file(config_file_path);
        std::cout << "[STATUS] " << config_file_path << " successfully read" << std::endl;
        if (!config_file.is_open()) {
            std::cerr << "[ERROR] failed opening of configuration file:" << config_file_path << std::endl;
            return EXIT_FAILURE;
        }
        config_file >> config;
        // Values from configuration file passed to arguments
        image_path = config["image_path"].get<std::string>();
        psf_path = config["psf_path"].get<std::string>();

        // Required in configuration file
        algorithm = config["algorithm"].get<std::string>();
        sigmax = config["sigmax"].get<double>();
        sigmay = config["sigmay"].get<double>();
        sigmaz = config["sigmaz"].get<double>();
        psfx = config["psfx"].get<int>();
        psfy = config["psfy"].get<int>();
        psfz = config["psfz"].get<int>();
        epsilon = config["epsilon"].get<double>();
        iterations = config["iterations"].get<int>();
        lambda = config["lambda"].get<double>();
        psfSafetyBorder = config["psfSafetyBorder"].get<int>();
        cubeSize = config["cubeSize"].get<int>();
        borderType = config["borderType"].get<int>();
        sep = config["seperate"].get<bool>();
        time = config["time"].get<bool>();
        savePsf = config["savePsf"].get<bool>();
        showExampleLayers = config["showExampleLayers"].get<bool>();
        printInfo = config["info"].get<bool>();

        //###PROGRAMM START###//
        PSF psf;
        if (psf_path == "gauss") {
             PSFGenerator<GaussianPSFGeneratorAlgorithm, double &, double &, double &, int &, int &, int &> gaussianGenerator(
                        sigmax, sigmay, sigmaz, psfx, psfy, psfz);
             psf = gaussianGenerator.generate();
        } else {
            if (psf_path.ends_with(".tif") || psf_path.ends_with(".tiff") || psf_path.ends_with(".TIF") || psf_path.ends_with(".TIFF") || psf_path.ends_with(".ometif") || psf_path.ends_with(".ometiff")) {
                psf.readFromTifFile(psf_path.c_str());
                std::cout << "[INFO] Read PSF as FILE" << std::endl;
            } else{
                psf.readFromTifDir(psf_path.c_str());
            }
        }

        Hyperstack hyperstack;
        if (image_path.ends_with(".tif") || image_path.ends_with(".tiff") || image_path.ends_with(".TIF") || image_path.ends_with(".TIFF") || image_path.ends_with(".ometif") || image_path.ends_with(".ometiff")) {
            hyperstack.readFromTifFile(image_path.c_str());
        } else {
            hyperstack.readFromTifDir(image_path.c_str());
        }
        if (savePsf) {
            psf.saveAsTifFile("../result/psf.tif");
        }
        if (printInfo) {
            hyperstack.printMetadata();
        }
        if (showExampleLayers) {
            hyperstack.showChannel(0);
        }
        hyperstack.saveAsTifFile("../result/input_hyperstack.tif");
        hyperstack.saveAsTifDir("../result/input_hyperstack");

        DeconvolutionConfig deconvConfig;
        deconvConfig.iterations = iterations;
        deconvConfig.epsilon = epsilon;
        deconvConfig.lambda = lambda;
        deconvConfig.borderType = borderType;
        deconvConfig.psfSafetyBorder = psfSafetyBorder;
        deconvConfig.cubeSize = cubeSize;

        Hyperstack deconvHyperstack;

        // Starttime
        auto start = std::chrono::high_resolution_clock::now();

        if (algorithm == "rl") {
            DeconvolutionAlgorithm<RLDeconvolutionAlgorithm> rlAlgorithm(deconvConfig);
            deconvHyperstack = rlAlgorithm.deconvolve(hyperstack, psf);
        } else if (algorithm == "rltv") {
            DeconvolutionAlgorithm<RLTVDeconvolutionAlgorithm> rltvAlgorithm(deconvConfig);
            deconvHyperstack = rltvAlgorithm.deconvolve(hyperstack, psf);
        } else {
            std::cerr
                    << "[ERROR] Please choose a available --algorithm: RichardsonLucy[rl], RichardsonLucyTotalVariation[rltv]"
                    << std::endl;
            return EXIT_FAILURE;
        }

        if (time) {
            // Endtime
            auto end = std::chrono::high_resolution_clock::now();
            // Calculation of the duration of the programm
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "[INFO] Algorithm duration: " << duration.count() << " ms" << std::endl;
        }
        if (showExampleLayers) {
            deconvHyperstack.showChannel(0);
        }

        deconvHyperstack.saveAsTifFile("../result/deconv.tif");
        if (sep) {
            deconvHyperstack.saveAsTifDir("../result/deconv");
        }

        //###PROGRAMM END###//
        std::cout << "[End DeconvTool]" << std::endl;
        return EXIT_SUCCESS;
    }
}