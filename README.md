<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>SQUID - Super Quick Image Deconvolution v1.0.0</h1>
</div>

SQUID is a compact high-performance 3D image deconvolution tool designed for processing large microscopy datasets with customized Point Spread Functions (PSFs) and algorithms like Richardson-Lucy deconvolution (RL) and RL with Total Variation (RLTV). The tool allows a quick and easy application. It supports various configurations for generating or loading PSFs and provides flexible options for enhancing image quality through deconvolution.

## Features

- **Support for Custom PSF**: Load an external PSF from a file or generate a synthetic Gaussian PSF with adjustable parameters.
- **Multiple Deconvolution Algorithms**: Choose between Richardson-Lucy and RL with Total Variation.
- **Configuration via CLI or JSON file**: Set up options either directly through the command line or with a JSON configuration file.
- **Grid Processing**: The image is split up into several cubes, which will processed parallel.
- **Extended Image Options**: Configure image border handling, PSF padding, and metadata viewing.
- **Output Options**: Save results as single or multi-layer TIFF files, with optional PSF saving.

## Dependencies

- **OpenCV**: For image manipulation and processing.
- **CLI11**: For command-line parsing.
- **nlohmann/json**: For JSON configuration file parsing.
- **C++ Standard Libraries**: `iostream`, `chrono`, and `fstream`.

## Usage

### Compilation

To compile the project, make sure all dependencies are installed and linked correctly. Then, use your preferred build system or compiler.

### Running the Tool

#### Basic Usage

```
./SQUID -i <input_image_path> -p <psf_path> -c <config_file_path> [OPTIONS]
```

#### Command-Line Options

| Option                 | Description                                                                                     | Default       |
|------------------------|-------------------------------------------------------------------------------------------------|---------------|
| `-i, --image`          | Path to input image (required).                                                                 |               |
| `-p, --psf`            | Path to PSF file or 'gauss' for synthetic PSF generation.                                       | "gauss"       |
| `-a, --algorithm`      | Deconvolution algorithm: 'rl' (Richardson-Lucy) or 'rltv' (RL with Total Variation).            | "rl"          |
| `--psfx`, `--psfy`, `--psfz` | PSF dimensions (width, height, depth) for synthetic PSF.                           | 20, 20, 40    |
| `--sigmax`, `--sigmay`, `--sigmaz` | Gaussian sigma values for synthetic PSF (x, y, z).                        | 5.0, 5.0, 5.0 |
| `--epsilon`            | Epsilon value for complex division.                                                             | 1e-6          |
| `--iterations`         | Number of iterations for the deconvolution algorithm.                                           | 10            |
| `--lambda`             | Regularization parameter for RLTV.                                                              | 0.01          |
| `--borderType`         | Border extension type (0: constant, 1: replicate, 2: reflect).                                  | 2             |
| `--psfSafetyBorder`    | Padding around the PSF.                                                                        | 10            |
| `--cubeSize`           | Edge length for sub-image grid (0 for auto-fit to PSF).                                        | 0             |
| `--savepsf`            | Save the generated PSF as a TIFF file.                                                          | false         |
| `--time`               | Display algorithm duration.                                                                     | false         |
| `--separate`           | Save each layer as a separate TIFF file in a directory.                                        | false         |
| `--info`               | Display metadata of the input image.                                                            | false         |
| `--showExampleLayers`  | Show example layers of the loaded image and PSF.                                                | false         |

### Configuration via JSON File

Configuration can also be loaded from a JSON file, with parameters specified as follows:

```json
{
    "image_path": "path/to/image.tif",
    "psf_path": "path/to/psf.tif",
    "algorithm": "rl",
    "sigmax": 5.0,
    "sigmay": 5.0,
    "sigmaz": 5.0,
    "psfx": 20,
    "psfy": 20,
    "psfz": 40,
    "epsilon": 1e-6,
    "iterations": 10,
    "lambda": 0.01,
    "psfSafetyBorder": 10,
    "cubeSize": 0,
    "borderType": 2,
    "separate": false,
    "time": true,
    "savePsf": true,
    "showExampleLayers": false,
    "info": true
}
```

To use a configuration file, specify it with the `-c` or `--config` option. Command-line arguments override any settings specified in the configuration file.

### Example Command

To deconvolve an image with a synthetic Gaussian PSF and display timing information:

```bash
./SQUID -i example_image.tif -p gauss -a rl --psfx 30 --psfy 30 --psfz 50 --time --info
```

## Output

- **Deconvolved Image**: Saved as `deconv.tif` in the `../result/` directory.
- **Optional PSF Output**: If `--savepsf` is set, the PSF is saved as `psf.tif` in the same directory.
- **Optional Layer Output**: If `--separate` is enabled, each layer of the deconvolved image is saved as a separate TIFF in a directory.

## License

Please refer to the LICENSE file in the repository for license information.