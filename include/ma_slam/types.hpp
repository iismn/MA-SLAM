#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>
#include <vector>

namespace ma_slam {

using Cloud = pcl::PointCloud<pcl::PointXYZI>;
using Matrix4 = Eigen::Matrix4d;

struct TimedPoint {
    float x = 0, y = 0, z = 0, intensity = 0;
    double offset_seconds = 0;
};

// Timestamps passed to MA-LIO are relative to one fixed run epoch.
struct TimedScan {
    int sensor_id = 0;
    double begin_seconds = 0;
    std::vector<TimedPoint> points;
};

struct ImuSample {
    double time_seconds = 0;
    Eigen::Vector3d acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
};

struct SensorCloud {
    int sensor_id = 0;
    Cloud points;
};

struct FrontendResult {
    Matrix4 imu_pose = Matrix4::Identity();
    Matrix4 center_pose = Matrix4::Identity();
    Matrix4 predicted_imu_pose = Matrix4::Identity();
    Matrix4 center_to_imu = Matrix4::Identity();
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Zero();
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    std::vector<SensorCloud> clouds;
    double end_seconds = 0;
    std::size_t map_points = 0, effective_points = 0, accepted_batches = 0, unsupported_points = 0;
};

}  // namespace ma_slam
