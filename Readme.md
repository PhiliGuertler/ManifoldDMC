# Masters-Thesis Philipp Gürtler

## Introduction
This Repository's main project `VisualDMC` implements a post-process for the Dual Marching Cubes algorithm, that resolves non-manifold geometry.
The code relies on the `C++17` standard and `CUDA 11.1`.

## Setup
To setup the projects, _CMake 3.18_ or newer is required (For CUDA Reasons).
Currently, this project only supports Windows (or rather, was only tested on a Windows machine).
Depending on the installed Graphics Card, `code/CMakeLists.txt` might have to be altered to match the card's architecture.
A list of values for each major architecture is included right at the point of its usage.
The main project is `VisualDMC`.

### Setup using CMake-Gui on Windows
- Run CMake-Gui and select the `code`-Directory as source. Choose a `build` folder, for example by creating a folder in the `code`-Directory. Click `Configure`, then `Generate`. After a successful generation, click `Open Project`. Right-Click the project that should be started and select `Set as startup project`, then build it.
- Extract the archive `Data-and-Results.zip` for example files. By default, the application loads the file `1-Edge.bin` from this extracted folder.

## References
The implementation of the Dual Marching Cubes algorithm is taken from https://github.com/rogrosso/tmc.
It has been modified heavily to be integrated into a graphical user interface.