#ifndef IMU_H
#define IMU_H

#include <opencv2/core.hpp>
#include <cstdint>
#include <string>
#include <vector>

struct ImuSample {
    std::int64_t timestampNs;
    cv::Vec3d angularVelocity; // rad/s, IMU frame
    cv::Vec3d specificForce;   // m/s^2, includes gravity, IMU frame
};

struct ImuInitialState {
    std::int64_t timestampNs = 0;
    cv::Matx33d worldFromImu = cv::Matx33d::eye();
    cv::Vec3d velocityWorld{0, 0, 0};
    cv::Vec3d gravityWorld{0, 0, -9.81};
    cv::Vec3d gyroBias{0, 0, 0};
    cv::Vec3d accelBias{0, 0, 0};
    cv::Vec3d cameraPositionInImu{0, 0, 0}; // calibrated camera lever arm, meters
};

std::vector<ImuSample> readImuCsv(const std::string& path);
ImuInitialState readImuInitialState(const std::string& path);
std::int64_t imageTimestampNs(const std::string& path);

// Open-loop inertial integration: metric units, but errors accumulate without
// visual feedback or bias estimation. Initial state must be supplied explicitly.
class ImuScaleEstimator {
public:
    ImuScaleEstimator(const std::vector<ImuSample>& samples,
                      const std::vector<std::int64_t>& imageTimes,
                      const ImuInitialState& initial);
    // Net camera displacement between the actual pair of matched images.
    double stepScale(std::size_t previousFrame, std::size_t currentFrame) const;
private:
    std::vector<cv::Vec3d> cameraPositions_;
};

#endif
