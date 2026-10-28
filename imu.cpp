#include "imu.h"
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

cv::Matx33d rotation(const cv::Vec3d& angle) {
    cv::Matx33d result;
    cv::Rodrigues(angle, result);
    return result;
}

cv::Vec3d readVector(const cv::FileStorage& file, const char* key) {
    std::vector<double> values;
    file[key] >> values;
    if (values.size() != 3) throw std::runtime_error(std::string("Expected 3 values for ") + key);
    cv::Vec3d v(values[0], values[1], values[2]);
    if (!finite(v)) throw std::runtime_error(std::string("Nonfinite ") + key);
    return v;
}
}

std::int64_t imageTimestampNs(const std::string& path) {
    const std::string stem = std::filesystem::path(path).stem().string();
    if (stem.empty() || stem.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Image filenames must be integer nanosecond timestamps: " + path);
    return std::stoll(stem);
}

std::vector<ImuSample> readImuCsv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open IMU CSV: " + path);
    std::vector<ImuSample> samples;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        if (line.empty() || line[0] == '#' || line == "\r") continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        ImuSample sample;
        std::string extra;
        if (!(row >> sample.timestampNs
                  >> sample.angularVelocity[0] >> sample.angularVelocity[1] >> sample.angularVelocity[2]
                  >> sample.specificForce[0] >> sample.specificForce[1] >> sample.specificForce[2]) ||
            (row >> extra) || !finite(sample.angularVelocity) || !finite(sample.specificForce) ||
            (!samples.empty() && sample.timestampNs <= samples.back().timestampNs))
            throw std::runtime_error("Invalid IMU CSV row " + std::to_string(lineNumber));
        samples.push_back(sample);
    }
    if (samples.size() < 2) throw std::runtime_error("Need at least two IMU samples");
    return samples;
}

ImuInitialState readImuInitialState(const std::string& path) {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) throw std::runtime_error("Cannot open IMU initial state: " + path);
    ImuInitialState state;
    std::string timestamp;
    file["timestamp_ns"] >> timestamp; // string avoids losing nanosecond precision
    if (timestamp.empty() || timestamp.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("timestamp_ns must be a quoted integer string");
    state.timestampNs = std::stoll(timestamp);
    state.velocityWorld = readVector(file, "velocity_world");
    state.gravityWorld = readVector(file, "gravity_world");
    state.gyroBias = readVector(file, "gyro_bias");
    state.accelBias = readVector(file, "accel_bias");
    state.cameraPositionInImu = readVector(file, "camera_position_in_imu");
    std::vector<double> r;
    file["world_from_imu"] >> r;
    if (r.size() != 9) throw std::runtime_error("world_from_imu needs 9 row-major values");
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(r[i])) throw std::runtime_error("Invalid initial rotation");
        state.worldFromImu.val[i] = r[i];
    }
    return state;
}

ImuScaleEstimator::ImuScaleEstimator(const std::vector<ImuSample>& samples,
                                   const std::vector<std::int64_t>& times,
                                   const ImuInitialState& initial) {
    if (samples.size() < 2 || times.empty()) throw std::invalid_argument("Missing IMU samples or image times");
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!finite(samples[i].angularVelocity) || !finite(samples[i].specificForce) ||
            (i && samples[i].timestampNs <= samples[i-1].timestampNs))
            throw std::invalid_argument("IMU samples must be finite and strictly time-ordered");
    }
    for (std::size_t i = 1; i < times.size(); ++i)
        if (times[i] <= times[i-1]) throw std::invalid_argument("Image times must strictly increase");
    if (initial.timestampNs > times.front() || initial.timestampNs < samples.front().timestampNs ||
        times.back() > samples.back().timestampNs)
        throw std::invalid_argument("IMU coverage must include initialization and all image times");
    cv::Matx33d R = initial.worldFromImu;
    if (!cv::checkRange(cv::Mat(R)) || cv::norm(R.t()*R - cv::Matx33d::eye()) > 1e-6 ||
        std::abs(cv::determinant(R) - 1.0) > 1e-6 || !finite(initial.velocityWorld) ||
        !finite(initial.gravityWorld) || !finite(initial.gyroBias) ||
        !finite(initial.accelBias) || !finite(initial.cameraPositionInImu))
        throw std::invalid_argument("Invalid IMU initial state or rotation");

    cv::Vec3d position(0, 0, 0), velocity = initial.velocityWorld;
    std::int64_t t = initial.timestampNs;
    std::size_t k = 0;
    for (const auto imageTime : times) {
        while (t < imageTime) {
            while (k + 1 < samples.size() && samples[k+1].timestampNs <= t) ++k;
            if (k + 1 >= samples.size()) throw std::runtime_error("IMU coverage ended");
            const auto& a = samples[k];
            const auto& b = samples[k+1];
            const double span = static_cast<double>(b.timestampNs - a.timestampNs);
            if (span > 20000000.0) throw std::runtime_error("IMU gap exceeds 20 ms");
            const auto end = std::min(imageTime, b.timestampNs);
            const double dt = static_cast<double>(end - t) * 1e-9;
            const double fraction = (static_cast<double>(t - a.timestampNs) +
                                     0.5 * static_cast<double>(end - t)) / span;
            const cv::Vec3d omega = a.angularVelocity * (1-fraction) +
                                   b.angularVelocity * fraction - initial.gyroBias;
            const cv::Vec3d force = a.specificForce * (1-fraction) +
                                   b.specificForce * fraction - initial.accelBias;
            const cv::Vec3d acceleration = R * rotation(omega * (0.5*dt)) * force + initial.gravityWorld;
            position += velocity * dt + acceleration * (0.5*dt*dt);
            velocity += acceleration * dt;
            R = R * rotation(omega * dt);
            if (!finite(position) || !finite(velocity)) throw std::runtime_error("IMU integration diverged");
            t = end;
        }
        cameraPositions_.push_back(position + R * initial.cameraPositionInImu);
    }
}

double ImuScaleEstimator::stepScale(std::size_t previousFrame, std::size_t currentFrame) const {
    if (previousFrame >= currentFrame || currentFrame >= cameraPositions_.size())
        throw std::out_of_range("Invalid frame pair for IMU scale");
    return cv::norm(cameraPositions_[currentFrame] - cameraPositions_[previousFrame]);
}
