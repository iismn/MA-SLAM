#include "ma_slam/imuPreintegration.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {

void require(bool value, const char* message)
{
    // -------------------------------------------------------------
    // - Keep invariant checks active in Release builds.
    // -------------------------------------------------------------
    if (!value) throw std::runtime_error(message);
}

}  // namespace

int main()
{
    // -------------------------------------------------------------
    // - Exercise original 2/3-LiDAR MA-LIO in a static three-plane scene.
    // - Rotate after startup with asynchronous 20/100ms sensor scan durations.
    // - Check physical sensor permutations, covariance, deskew and intensity.
    // - Write no map files.
    // -------------------------------------------------------------
    using namespace ma_slam;
    try {
        const int count = ImuPreintegration::lidarCount();
        std::vector<Matrix4> transforms(count, Matrix4::Identity());
        const double radians = std::acos(-1.0) / 180.0;
        transforms[1].topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(55 * radians, Eigen::Vector3d::UnitY()).toRotationMatrix();
        transforms[1].topRightCorner<3, 1>() = Eigen::Vector3d(.3, 0, -.2);
        if (count == 3) {
            transforms[2].topLeftCorner<3, 3>() =
                (Eigen::AngleAxisd(55 * radians, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(91 * radians, Eigen::Vector3d::UnitZ())).toRotationMatrix();
            transforms[2].topRightCorner<3, 1>() = Eigen::Vector3d(-.27, 0, -.22);
        }
        ImuPreintegration frontend(FrontendConfig{}, transforms);
        frontend.testCovariancePermutation();

        std::mt19937 random(8);
        std::uniform_real_distribution<double> uniform(-10, 10);
        std::vector<Eigen::Vector3d> scene;
        scene.reserve(6000);
        for (int plane = 0; plane < 3; ++plane) {
            for (int point = 0; point < 2000; ++point) {
                const double a = uniform(random), b = uniform(random);
                if (plane == 0) scene.emplace_back(a, b, -2);
                if (plane == 1) scene.emplace_back(8, a, b);
                if (plane == 2) scene.emplace_back(a, 8, b);
            }
        }
        double previous = 0;
        FrontendResult result;
        for (int batch = 0; batch < 10; ++batch) {
            std::vector<TimedScan> scans;
            double batch_end = 0;
            for (int sensor = 0; sensor < count; ++sensor) {
                const double duration = sensor == 2 ? .019 : .099;
                const double end = batch * .1 + .099 + (sensor == batch % count ? .003 : 0);
                TimedScan scan;
                scan.sensor_id = sensor;
                scan.begin_seconds = end - duration;
                scan.points.reserve(scene.size());
                for (std::size_t index = 0; index < scene.size(); ++index) {
                    const double offset = duration * index / (scene.size() - 1);
                    const double yaw = .3 * std::max(scan.begin_seconds + offset - .5, 0.0);
                    const auto rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                    const Eigen::Vector3d body = rotation.transpose() * scene[index];
                    const Eigen::Vector3d point = transforms[sensor].topLeftCorner<3, 3>().transpose() *
                        (body - transforms[sensor].topRightCorner<3, 1>());
                    scan.points.push_back({float(point.x()), float(point.y()), float(point.z()),
                                           float(index % 255), offset});
                }
                batch_end = std::max(batch_end, end);
                scans.push_back(std::move(scan));
            }
            std::vector<ImuSample> imu;
            for (double time = previous + 1e-6; time < batch_end + .05; time += .0025) {
                imu.push_back({time, Eigen::Vector3d(0, 0, 9.81),
                               Eigen::Vector3d(0, 0, time >= .5 ? .3 : 0)});
            }
            result = frontend.process(scans, imu);
            previous = result.end_seconds;
            require(result.center_pose.allFinite(), "Non-finite pose");
            require(result.center_pose.topRightCorner<3, 1>().norm() < .08, "Static scene translation drift");
            require(result.position_covariance.allFinite(), "Non-finite covariance");
            require(result.position_covariance.isApprox(result.position_covariance.transpose(), 1e-10),
                    "Asymmetric covariance");
            require(Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(result.position_covariance)
                        .eigenvalues().minCoeff() >= -1e-9, "Indefinite covariance");
            for (const auto& sensor : result.clouds) {
                std::vector<double> errors;
                for (const auto& point : sensor.points) {
                    require(std::isfinite(point.intensity) && point.intensity >= 0 &&
                                point.intensity < 255 && point.intensity == std::floor(point.intensity),
                            "Measured intensity was not preserved");
                    const Eigen::Vector3d world = result.center_pose.topLeftCorner<3, 3>() *
                        Eigen::Vector3d(point.x, point.y, point.z) + result.center_pose.topRightCorner<3, 1>();
                    require(world.allFinite(), "Non-finite deskewed point");
                    errors.push_back(std::min({std::abs(world.z() + 2), std::abs(world.x() - 8),
                                               std::abs(world.y() - 8)}));
                }
                if (batch >= 2) {
                    require(sensor.points.size() == scene.size(), "Supported points disappeared");
                    std::sort(errors.begin(), errors.end());
                    require(errors[std::size_t(.95 * (errors.size() - 1))] < .1,
                            "Rotating 20/100ms scan deskew failed");
                }
            }
        }
        require(result.effective_points > 100, "Planar scan matching did not run");
        std::cout << "PASS: " << count << "-LiDAR original C++ MA-LIO; covariance permutations, "
                     "20/100ms moving deskew, planar IESKF and intensity.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
