#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace ma_slam {

struct GraphConfig {
    double submap_voxel = 0.5;
    double submap_distance = 10.0;
    double min_range = 2.0;
    double max_range = 100.0;
    double lidar_position_sigma = 0.1;
    int graph_max_iterations = 30;
    double loop_search_radius = 15.0;
    double loop_max_correction = 3.0;
    double closure_overlap_threshold = 0.4;
    double closure_max_rmse = 0.25;
    double closure_fine_distance = 0.35;
    double closure_min_time_separation = 30.0;
    int closure_max_candidates = 3;
    int no_of_sub_maps_to_skip = 3;
    bool use_gt = false;
    bool gt_gate_covariance = true;
    bool gt_gate_graph_covariance = true;
    double gt_covariance_threshold = 0.04;
    double gt_covariance_ratio = 1.0;
    double gt_min_factor_distance = 5.0;
    // A GT sample closer than gt_min_factor_distance is still accepted once this many seconds
    // passed since the last GT factor (stops, degenerate stretches). 0 keeps the distance-only gate.
    double gt_min_factor_interval = 0.0;
};

struct GraphStats {
    std::size_t scans = 0;
    std::size_t submaps = 0;
    std::size_t closures = 0;
    std::size_t gt_edges = 0;
    std::size_t gt_candidates = 0;
    std::size_t gt_skipped_covariance = 0;
    std::size_t gt_skipped_distance = 0;
    std::size_t loop_candidates = 0;
    std::size_t loop_rejected = 0;
    double loop_translation_rmse_m = 0.0;
    double loop_rotation_rmse_deg = 0.0;
    double loop_min_robust_weight = 1.0;
};

struct GraphResult {
    std::vector<std::int64_t> times_ns;
    std::vector<Eigen::Matrix4d> poses;
    std::vector<std::size_t> submap_start_indices;
    std::vector<std::pair<int, int>> loop_pairs;
    std::vector<std::int64_t> gt_factor_times_ns;
    GraphStats stats;
};

// Poses map the selected reference LiDAR into the same world frame as GT.
// Clouds must already be deskewed into that reference LiDAR at time_ns.
// This class is serialized by the pipeline worker; it does not own ROS or threads.
class MapOptimization {
public:
    explicit MapOptimization(const GraphConfig& config = {});
    ~MapOptimization();
    MapOptimization(const MapOptimization&) = delete;
    MapOptimization& operator=(const MapOptimization&) = delete;
    MapOptimization(MapOptimization&&) noexcept;
    MapOptimization& operator=(MapOptimization&&) noexcept;

    void setGroundTruth(const std::vector<std::int64_t>& times_ns,
                        const std::vector<Eigen::Vector3d>& positions,
                        const Eigen::Matrix3d& covariance, double huber_delta);
    bool addScan(std::int64_t time_ns, const pcl::PointCloud<pcl::PointXYZI>& cloud,
                 const Eigen::Matrix4d& pose, const Eigen::Matrix3d& position_covariance);
    void finalize();
    GraphResult trajectory() const;
    GraphResult snapshot() const;
    Eigen::Matrix4d latestPose() const;
    GraphStats stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ma_slam
