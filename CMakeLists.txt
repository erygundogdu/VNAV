cmake_minimum_required(VERSION 3.16)
project(camera_motion LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_DEBUG_DEMO "Build the local OpenCV GUI matching demo (requires debug.cpp)" OFF)

set(opencv_components core imgproc imgcodecs features2d calib3d)
if(BUILD_DEBUG_DEMO)
    list(APPEND opencv_components highgui)
endif()
find_package(OpenCV 4 REQUIRED COMPONENTS ${opencv_components})

add_library(camera_motion_helpers STATIC helper.cpp)
target_compile_features(camera_motion_helpers PUBLIC cxx_std_17)
target_include_directories(camera_motion_helpers PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(camera_motion_helpers SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_motion_helpers PUBLIC ${OpenCV_LIBS})

add_executable(camera_motion main.cpp)
target_link_libraries(camera_motion PRIVATE camera_motion_helpers)

if(BUILD_DEBUG_DEMO)
    add_executable(debug_demo debug.cpp)
    target_compile_features(debug_demo PRIVATE cxx_std_17)
    target_include_directories(debug_demo SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(debug_demo PRIVATE ${OpenCV_LIBS})
endif()
