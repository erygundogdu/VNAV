#include "imu.h"
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>

void near(double actual, double expected, double tolerance = 1e-9) {
    if (std::abs(actual - expected) > tolerance)
        throw std::runtime_error("Expected " + std::to_string(expected) + ", got " + std::to_string(actual));
}

template<class F> void rejects(F f) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Expected rejection");
}

int main() try {
    constexpr std::int64_t epoch = 1403715273262142976;
    std::vector<ImuSample> samples;
    for (int i = 0; i <= 200; ++i)
        samples.push_back({epoch + i*5000000LL, {0,0,0}, {0,0,9.81}});
    const std::vector<std::int64_t> times{epoch, epoch+123456789, epoch+500000000, epoch+1000000000};
    ImuInitialState state;
    state.timestampNs = epoch;
    ImuScaleEstimator rest(samples, times, state);
    near(rest.stepScale(0,3), 0);
    state.velocityWorld = {2,0,0};
    ImuScaleEstimator moving(samples, times, state);
    near(moving.stepScale(0,1), 2*0.123456789);
    near(moving.stepScale(0,3), 2); // previous visual frame retained across failures
    near(moving.stepScale(2,3), 1);
    state.velocityWorld = {0,0,0};
    for (auto& s : samples) s.specificForce = {2,0,9.81};
    ImuScaleEstimator accelerating(samples, times, state);
    near(accelerating.stepScale(0,3), 1); // 0.5*a*t^2
    near(accelerating.stepScale(2,3), 0.75); // velocity must persist

    cv::Rodrigues(cv::Vec3d(0, 0.7, 0), state.worldFromImu);
    state.gyroBias = {0.01,0.02,0.03};
    state.accelBias = {0.1,0.2,0.3};
    for (auto& s : samples) {
        s.angularVelocity = state.gyroBias;
        s.specificForce = state.worldFromImu.t()*cv::Vec3d(0,0,9.81) + state.accelBias;
    }
    near(ImuScaleEstimator(samples,times,state).stepScale(0,3), 0);

    state = ImuInitialState{};
    state.timestampNs = epoch;
    state.cameraPositionInImu = {1,0,0};
    for (auto& s : samples) { s.angularVelocity = {0,0,1}; s.specificForce = {0,0,9.81}; }
    near(ImuScaleEstimator(samples,times,state).stepScale(0,3), 2*std::sin(0.5));
    rejects([&]{ moving.stepScale(2,1); });
    rejects([&]{ ImuScaleEstimator(samples,{epoch-1,epoch},state); });
    rejects([&]{ ImuScaleEstimator(samples,{epoch,epoch+1000000001},state); });
    rejects([&]{ ImuScaleEstimator(samples,{epoch,epoch},state); });
    samples.erase(samples.begin()+1, samples.begin()+10);
    rejects([&]{ ImuScaleEstimator(samples,times,state); });
    near(static_cast<double>(imageTimestampNs("data/1403715273262142976.png") - epoch), 0);
    rejects([]{ imageTimestampNs("data/frame.png"); });
    std::cout << "IMU integration checks passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
