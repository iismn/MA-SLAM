#pragma once

#include "ma_slam/groundTruth.hpp"
#include "ma_slam/imageProjection.hpp"
#include "ma_slam/imuPreintegration.hpp"
#include "ma_slam/mapOptimization.hpp"
#include "ma_slam/mapStorage.hpp"
#include <memory>

namespace ma_slam {

struct PipelineConfig {
    FrontendConfig frontend;
    GraphConfig graph;
    std::vector<SensorConfig> sensors;
    std::vector<Matrix4> lidar_to_imu;
    OutputConfig output;
    bool slam_enabled = true;
    bool gt_align_initial = true;
    bool gt_evaluate = true;
    bool interpolate_pose_corrections = true;
    std::string gt_file;
    Eigen::Matrix3d gt_covariance = Eigen::Vector3d(.01, .01, .04).asDiagonal();
    double gt_huber_delta = 3;
    Eigen::Vector3d gt_lidar_lever_arm = Eigen::Vector3d(0, 0, .42);
    double preview_voxel = .5;
    std::size_t preview_max_points = 200000;
};

struct PipelineUpdate {
    std::int64_t stamp_ns = 0;
    Matrix4 odom_pose = Matrix4::Identity(), map_pose = Matrix4::Identity();
    Eigen::Matrix3d position_covariance = Eigen::Matrix3d::Zero();
    Cloud cloud;
    GraphStats graph;
    std::size_t effective_points = 0, unsupported_points = 0;
    bool submap_changed = false;
};

// All estimator/graph/storage calls run on the same worker; ROS is only transport.
class Pipeline {
public:
    explicit Pipeline(PipelineConfig config);
    std::optional<PipelineUpdate> process(const SynchronizedBatch& batch);
    GraphResult trajectory() const;
    Cloud preview() const;
    std::string finalize();
    SavedMap save();
    bool finalized() const { return finalized_; }
    std::size_t frames() const { return times_.size(); }

private:
    struct PreviewFrame { std::size_t index; Cloud cloud; };
    PipelineConfig config_;
    ImuPreintegration frontend_;
    std::unique_ptr<MapOptimization> graph_;
    std::optional<GroundTruth> gt_;
    MapStorage storage_;
    std::optional<std::int64_t> epoch_ns_;
    Matrix4 alignment_ = Matrix4::Identity();
    std::vector<std::int64_t> times_;
    std::vector<Matrix4> raw_poses_;
    std::vector<PreviewFrame> preview_frames_;
    std::size_t last_submaps_ = 0;
    bool finalized_ = false;
};

}  // namespace ma_slam
