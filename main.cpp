#include "helper.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>

int main() {
    const cv::Mat K = (cv::Mat_<double>(3, 3) <<
        458.654, 0.0, 367.215,
        0.0, 457.296, 248.375,
        0.0, 0.0, 1.0);

    const cv::Mat distortion = (cv::Mat_<double>(1, 4) <<
        -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05);

    std::vector<cv::String> imagePaths;
    cv::glob("data/*.png", imagePaths);
    std::sort(imagePaths.begin(), imagePaths.end());

    if (imagePaths.empty()) {
        std::cerr << "No PNG frames found in data/\n";
        return 1;
    }

    cv::Mat ImagePrev = cv::imread(imagePaths[0], cv::IMREAD_GRAYSCALE);

    if (ImagePrev.empty()) {
        std::cerr << "Could not load: " << imagePaths[0] << '\n';
        return 1;
    }

    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    std::ofstream csv("trajectory.csv");

    if (!csv) {
        std::cerr << "Could not open trajectory.csv\n";
        return 1;
    }

    csv << "frame,x,y,z\n";
    csv << "0,0,0,0\n";
    size_t skipped = 0;

    for (size_t i = 1; i < imagePaths.size(); ++i) {
        cv::Mat image = cv::imread(imagePaths[i], cv::IMREAD_GRAYSCALE);

        if (image.empty()) {
            std::cerr << "Could not load: " << imagePaths[i] << '\n';
            return 1;
        }

        try {
            const CameraMotion motion_t =
                estimateCameraMotion(ImagePrev, image, K, 1.0, distortion);

            T = motion_t.C * T;
            cv::Mat cameraPose = T.inv();

            cv::Point3d position(
                cameraPose.at<double>(0, 3),
                cameraPose.at<double>(1, 3),
                cameraPose.at<double>(2, 3)
            );

            csv << i << ',' << position.x << ','
                << position.y << ',' << position.z << '\n';

            ImagePrev = image;
        } catch (const std::exception& error) {
            std::cerr << "Frame " << i << " skipped: "
                      << error.what() << '\n';
            csv << i << ",nan,nan,nan\n";
            ++skipped;
        }
    }

    std::cout << "Saved " << imagePaths.size()
              << " frames to trajectory.csv; "
              << skipped << " positions could not be recovered.\n";

    return 0;
}