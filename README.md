<p align="center">
<img src="https://github.com/Steffenhir/GraXpert/blob/main/img/GraXpert_LOGO_Hauptvariante.png" width="500"/>
</p>

GraXpert is an astronomical image processing program for extracting and removing gradients, as well as denoising and deconvolving your astrophotos. We provide several methods, including traditional interpolation methods such as Radial Basis Functions (RBF), Splines, and Kriging, which require the user to manually select sample points in the background of the image for gradient removal. Our newest additions are advanced AI models that automate these processes, including gradient removal, denoising, and deconvolution, without requiring any user input.

Original                     |  Gradients removed with AI
:-------------------------:|:-------------------------:
![Original](https://github.com/Steffenhir/GraXpert/blob/main/img/NGC7000_original.jpg)   |  ![Gradients removed](https://github.com/Steffenhir/GraXpert/blob/main/img/NGC7000_processed.jpg)
![Original](https://github.com/Steffenhir/GraXpert/blob/main/img/LDN1235_original.jpg)   |  ![Gradients removed](https://github.com/Steffenhir/GraXpert/blob/main/img/LDN1235_processed.jpg)

# Features
- Background gradient extraction using traditional methods (RBF, Splines, Kriging) or AI
- AI-powered denoising
- AI-powered deconvolution with two modes:
  - Object-only deconvolution for enhancing nebulosity and galaxies
  - Stars-only deconvolution for improving stellar sharpness
- Support for multiple image formats including FITS, TIFF, XISF, PNG, JPEG, and BMP
- GPU acceleration support for AI operations
- Batch processing capabilities

**Homepage:** [https://www.graxpert.com](https://www.graxpert.com)  
**Download:** [https://github.com/Steffenhir/GraXpert/releases/latest](https://github.com/Steffenhir/GraXpert/releases/latest)

# Installation
You can download the latest official release of GraXpert [here](https://github.com/Steffenhir/GraXpert/releases/latest). Select the correct version for your operating system. For macOS, we provide different versions
for Intel processors (x86_64) and for apple silicon (arm64).

**Windows:** After downloading the .exe file, you should be able to start it directly. \
**Linux:** Before you can start GraXpert, you have to make it executable by running ```chmod u+x ./GraXpert-linux``` \
**macOS:** After opening the .dmg file, simply drag the GraXpert icon into the applications folder. GraXpert can now be started from the applications folder.

# Command-Line Usage
GraXpert comes with a graphical user interface but also supports command-line operations for batch processing. Available commands include:

## Background Extraction
```
GraXpert-win64.exe my_image.fits -cli --command background-extraction [options]
```

## Denoising
```
GraXpert-win64.exe my_image.fits -cli --command denoising [options]
```

## Deconvolution
```
# For object deconvolution:
GraXpert-win64.exe my_image.fits -cli --command deconv-obj [options]

# For stellar deconvolution:
GraXpert-win64.exe my_image.fits -cli --command deconv-stellar [options]
```

Common options for all commands:
- `-output [filename]`: Specify output filename
- `-gpu [true/false]`: Enable/disable GPU acceleration
- `-ai_version [version]`: Specify AI model version
- `-batch_size [size]`: Number of image tiles to process in parallel (1-32, default: 4)

Additional options by command:
- Background Extraction:
  - `-correction [Subtraction/Division]`: Background correction method
  - `-smoothing [0.0-1.0]`: Smoothing strength
  - `-bg`: Also save the background model

- Denoising:
  - `-strength [0.0-1.0]`: Denoising strength

- Deconvolution:
  - `-strength [0.0-1.0]`: Deconvolution strength
  - `-psfsize [value]`: PSF size for deconvolution

# Installation for Developers
This guide will help you get started with development of GraXpert on Windows, Linux, and macOS. Follow these steps to clone the repository, create a virtual environment with Python, install the required packages, and run GraXpert from the source code.

## Clone the repository
Open your terminal or command prompt and use git to clone the GraXpert repository:
```
git clone https://github.com/Steffenhir/GraXpert
cd GraXpert
```

## Setting up a Virtual Environment
We recommend using a virtual environment to isolate the project's dependencies. Ensure you have Python>=3.11 installed on your system before proceeding. Here's how to set up a virtual environment with Python:
Windows:
```
# Create a new virtual environment with Python 3.11
python -m venv graxpert-env

# Activate the virtual environment
graxpert-env\Scripts\activate
```

Linux and macOS:
```
# Create a new virtual environment with Python 3.11
python3 -m venv graxpert-env

# Activate the virtual environment
source graxpert-env/bin/activate
```

## Install required packages
All the requirements can be found in the requirements.txt file. You can install them with:

Windows and Linux:
```
pip install -r requirements.txt
pip install onnxruntime-gpu # if you have an AMD gpu change to onnxruntime-rocm
```

macOS:
```
pip3 install -r requirements.txt
pip3 install onnxruntime
```

For macOS, we have to install tkinter separately.
We use the version provided by brew because it is newer
and solves issues with macOS Sonoma. Please use the version matching with your Python version.
```
brew install python-tk@3.11
```

## Running GraXpert
Once you have set up the virtual environment and installed the required packages, you can start GraXpert:

```
python -m graxpert.main
```


