# Camera motion estimation

C++17 / OpenCV project with two programs:

- `camera_motion`: estimates camera motion from PNG frames in `data/` and writes `trajectory.csv`.
- `debug_demo` (optional, local only): displays ORB matches between `airport.jpg` and a translated copy. Requires a graphical desktop. Its source and image assets are ignored by Git.

## Dependencies

Install a C++17 compiler, CMake 3.16 or newer, a build tool (Make, Ninja, or Visual Studio), and the **OpenCV 4 C++ development libraries**. The Python `opencv-python` package is not sufficient.

Ubuntu / Debian:

```sh
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

macOS with Homebrew and the Xcode command line tools installed:

```sh
brew install cmake opencv
```

On Windows, install Visual Studio with Desktop development with C++, CMake, and an OpenCV 4 development distribution matching your compiler and architecture. Set `OpenCV_DIR` as described below if needed.

## Build

From the cloned repository directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
```

CMake discovers OpenCV and supplies the compiler's include paths and link libraries automatically. No manual `g++` command is needed; CMake uses the installed compiler through the selected build tool.

If OpenCV is installed in a custom location, point to the **directory containing `OpenCVConfig.cmake`**:

```sh
cmake -S . -B build -DOpenCV_DIR="/path/to/opencv/cmake/directory"
```

Alternatively, provide the installation prefix with `-DCMAKE_PREFIX_PATH="/path/to/opencv/install"`. These paths are local configuration and should not be committed. When moving between computers or changing compilers, start with a fresh build directory.

The GUI demo and its `highgui` dependency are disabled by default. To enable it when you have local copies of `debug.cpp` and `airport.jpg`:

```sh
cmake -S . -B build -DBUILD_DEBUG_DEMO=ON
cmake --build build --config Release --parallel
```

## Run

Run from the repository root so relative image paths resolve correctly. If you enabled the local GUI demo:

```sh
./build/debug_demo
```

For camera motion, first place your PNG sequence in `data/` (not included), then run:

```sh
./build/camera_motion
```

Frames are sorted lexicographically; use zero-padded names such as `000001.png`. Update the camera intrinsics and distortion in `main.cpp` for your dataset. Translation uses a fixed step scale of 1.0, so the output is not a calibrated metric trajectory.

With Visual Studio or other multi-configuration generators, executables are under `build/Release/` instead (for example, `./build/Release/camera_motion.exe`).

## VS Code

Install the recommended C/C++ and CMake Tools extensions. Run **CMake: Configure**, select an installed compiler if prompted, then **CMake: Build**. CMake Tools provides IntelliSense with the detected compiler and OpenCV paths. Make and Ninja builds also generate `build/compile_commands.json`.

## Version control

Commit the main/helper source files, `CMakeLists.txt`, this README, `.gitignore`, and the shared `.vscode` configuration. Airport image assets (`airport.jpg*`), debug experiments (`debug*`), build directories, local datasets, generated trajectories, and old standalone binaries are ignored. If any ignored files were already tracked, `.gitignore` will not untrack them automatically.
