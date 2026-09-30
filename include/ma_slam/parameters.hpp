#pragma once

#include "ma_slam/pipeline.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>
#include <vector>
#include <string>

namespace ma_slam {

struct Parameters {
    PipelineConfig pipeline;
    std::string imu_topic = "/imu/data";
    double imu_time_offset_seconds = 0;
    double pair_tolerance_seconds = .08;
    std::size_t scan_queue_limit = 100;
    std::size_t imu_queue_limit = 20000;
    std::string frame_map = "map";
    std::string frame_odom = "odom";
    std::string frame_lidar = "lidar_center";
    std::size_t map_publish_every_submaps = 5;
};

using ParameterValue = std::variant<bool, std::int64_t, double, std::string,
                                    std::vector<double>, std::vector<std::int64_t>>;
using ParameterGetter = std::function<ParameterValue(const std::string&, const ParameterValue&)>;

// ROS2 declarations and notebook YAML overrides use the same defaults/checks.
Parameters readParameters(const ParameterGetter& get);

}  // namespace ma_slam
