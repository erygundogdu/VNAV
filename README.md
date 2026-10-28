# Camera motion estimation

A small C++ project for estimating camera motion from a sequence of images using OpenCV. It matches ORB features between frames, estimates the essential matrix with RANSAC, and recovers the camera pose. Positions are saved to `trajectory.csv`.

## Setup

You need a C++17 compiler, CMake 3.16 or newer, and the OpenCV 4 C++ libraries.

On Ubuntu / Debian:

```sh
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

On macOS, install the Xcode command line tools, then:

```sh
brew install cmake opencv
```

On Windows, install Visual Studio with the **Desktop development with C++** workload, CMake, and OpenCV 4 for your compiler and architecture.

## Build

Run these commands from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

CMake finds OpenCV automatically. If it cannot find your installation, set `OpenCV_DIR` to the folder containing `OpenCVConfig.cmake`:

```sh
cmake -S . -B build -DOpenCV_DIR="/path/to/opencv/cmake/directory"
```

Use a fresh build directory when switching computers or compilers.

## Run

Put your PNG frames directly in `data/`:

```text
data/
  000001.png
  000002.png
  000003.png
```

Frames are read in filename order, so use timestamps or zero-padded names. Run from the repository root:

```sh
./build/camera_motion
```

With Visual Studio, the executable is usually at `build/Release/camera_motion.exe`.

The program writes `trajectory.csv` in the current directory, with columns `frame,x,y,z`. The first position is the origin. Frames where pose recovery fails have `nan` coordinates; the reason is printed in the terminal.

The calibration in `main.cpp` is for the left camera in EuRoC MAV. Change the camera matrix and distortion coefficients if you use another camera. Translation uses a fixed step scale of 1.0, so the coordinates are **not in meters**.

Image sequences and generated CSV files are ignored by Git and are not included when you clone the repo.

## VS Code

Install the recommended **C/C++** and **CMake Tools** extensions. Run **CMake: Configure**, choose your compiler, then **CMake: Build**. The included settings let IntelliSense use the OpenCV paths found by CMake.

## Local debug demo

If you have `debug.cpp` and `airport.jpg`, you can also build the ORB matching demo:

```sh
cmake -S . -B build -DBUILD_DEBUG_DEMO=ON
cmake --build build --config Release --parallel
./build/debug_demo
```

It opens a window showing matches between the image and a translated copy. The demo is off by default, and its source, binaries, and image assets are ignored by Git.
