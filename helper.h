#ifndef HELPER_H
#define HELPER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct MatchedPoints {
    std::vector<cv::Point2f> previous;
    std::vector<cv::Point2f> current;
};

struct CameraMotion {
    cv::Mat R;
    cv::Mat t;
    cv::Mat C;
    cv::Mat inlierMask;
    int inlierCount = 0;
};

cv::Mat readImage(const std::string& str);

cv::Mat RotranslateImage(
    cv::Mat& Image,
    double R = 0.0,
    double t_x = 50.0,
    double t_y = 50.0
);

MatchedPoints orbMatch(
    const cv::Mat& previousGray,
    const cv::Mat& currentGray,
    int maxFeatures = 2000
);

cv::Mat ransacEssentialMatrix(
    const MatchedPoints& points,
    const cv::Mat& K,
    cv::Mat& inlierMask,
    double thresholdPixels = 1.0
);

CameraMotion recoverCameraMotion(
    const cv::Mat& E,
    const MatchedPoints& points,
    const cv::Mat& K,
    const cv::Mat& ransacInliers,
    double stepScale = 1.0
);

CameraMotion estimateCameraMotion(
    const cv::Mat& previousImage,
    const cv::Mat& currentImage,
    const cv::Mat& K,
    double stepScale = 1.0,
    const cv::Mat& distortion = cv::Mat()
);

#endif