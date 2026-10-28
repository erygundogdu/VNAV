#include "helper.h"
#include <cmath>
#include <stdexcept>
#include <utility>

cv::Mat readImage(const std::string& str) {
    return cv::imread(str);
}

cv::Mat RotranslateImage(
    cv::Mat& Image,
    double R,
    double t_x,
    double t_y
) {
    cv::Mat ImageT;
    cv::Point2f center(Image.cols / 2.0f, Image.rows / 2.0f);
    cv::Mat M = cv::getRotationMatrix2D(center, R, 1.0);

    M.at<double>(0, 2) += t_x;
    M.at<double>(1, 2) += t_y;

    cv::warpAffine(Image, ImageT, M, Image.size());

    return ImageT;
}

MatchedPoints orbMatch(
    const cv::Mat& previousGray,
    const cv::Mat& currentGray,
    int maxFeatures
) {
    if (previousGray.empty() || currentGray.empty()) {
        throw std::invalid_argument(
            "orbMatch needs two nonempty images."
        );
    }

    auto orb = cv::ORB::create(maxFeatures);
    std::vector<cv::KeyPoint> previousKeypoints, currentKeypoints;
    cv::Mat previousDescriptors, currentDescriptors;

    orb->detectAndCompute(
        previousGray, cv::noArray(),
        previousKeypoints, previousDescriptors
    );

    orb->detectAndCompute(
        currentGray, cv::noArray(),
        currentKeypoints, currentDescriptors
    );

    MatchedPoints points;

    if (previousDescriptors.empty() || currentDescriptors.empty()) {
        return points;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(previousDescriptors, currentDescriptors, matches);

    points.previous.reserve(matches.size());
    points.current.reserve(matches.size());

    for (const cv::DMatch& match : matches) {
        points.previous.push_back(previousKeypoints[match.queryIdx].pt);
        points.current.push_back(currentKeypoints[match.trainIdx].pt);
    }

    return points;
}

cv::Mat ransacEssentialMatrix(
    const MatchedPoints& points,
    const cv::Mat& K,
    cv::Mat& inlierMask,
    double thresholdPixels
) {
    if (points.previous.size() != points.current.size() ||
        points.previous.size() < 8) {
        throw std::invalid_argument(
            "Need at least eight matched point pairs."
        );
    }

    if (K.rows != 3 || K.cols != 3) {
        throw std::invalid_argument(
            "K must be the calibrated 3x3 intrinsic matrix."
        );
    }

    cv::Mat E = cv::findEssentialMat(
        points.previous, points.current, K,
        cv::RANSAC, 0.999, thresholdPixels, inlierMask
    );

    if (E.rows != 3 || E.cols != 3 ||
        cv::countNonZero(inlierMask) < 8) {
        throw std::runtime_error(
            "Could not estimate a supported essential matrix."
        );
    }

    return E;
}

CameraMotion recoverCameraMotion(
    const cv::Mat& E,
    const MatchedPoints& points,
    const cv::Mat& K,
    const cv::Mat& ransacInliers,
    double stepScale
) {
    if (!std::isfinite(stepScale) || stepScale <= 0.0) {
        throw std::invalid_argument(
            "stepScale must be finite and positive."
        );
    }

    CameraMotion motion;
    motion.inlierMask = ransacInliers.clone();

    motion.inlierCount = cv::recoverPose(
        E, points.previous, points.current, K,
        motion.R, motion.t, motion.inlierMask
    );

    if (motion.inlierCount < 8) {
        throw std::runtime_error(
            "Only " + std::to_string(motion.inlierCount) +
            " pose inliers (need at least 8); check matches and parallax."
        );
    }

    motion.t *= stepScale;
    motion.C = cv::Mat::eye(4, 4, CV_64F);

    motion.R.copyTo(motion.C(cv::Rect(0, 0, 3, 3)));
    motion.t.copyTo(motion.C(cv::Rect(3, 0, 1, 3)));

    return motion;
}

CameraMotion estimateCameraMotion(
    const cv::Mat& previousImage,
    const cv::Mat& currentImage,
    const cv::Mat& K,
    double stepScale,
    const cv::Mat& distortion
) {
    MatchedPoints points = orbMatch(previousImage, currentImage);

    if (!distortion.empty() && !points.previous.empty()) {
        std::vector<cv::Point2f> previousUndistorted;
        std::vector<cv::Point2f> currentUndistorted;

        cv::undistortPoints(
            points.previous, previousUndistorted,
            K, distortion, cv::noArray(), K
        );

        cv::undistortPoints(
            points.current, currentUndistorted,
            K, distortion, cv::noArray(), K
        );

        points.previous = std::move(previousUndistorted);
        points.current = std::move(currentUndistorted);
    }

    cv::Mat inlierMask;
    cv::Mat E = ransacEssentialMatrix(points, K, inlierMask);

    return recoverCameraMotion(
        E, points, K, inlierMask, stepScale
    );
}