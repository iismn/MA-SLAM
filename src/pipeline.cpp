#include "ma_slam/pipeline.hpp"
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace ma_slam {
namespace {
Cloud downsample(const Cloud& cloud, double voxel) {
    Cloud out;
    pcl::VoxelGrid<pcl::PointXYZI> filter;
    filter.setInputCloud(cloud.makeShared());
    filter.setLeafSize(voxel, voxel, voxel);
    filter.setDownsampleAllData(true);
    filter.filter(out);
    return out;
}
}

Pipeline::Pipeline(PipelineConfig config)
    : config_(std::move(config)), frontend_(config_.frontend, config_.lidar_to_imu) {
    if (config_.sensors.size() != config_.lidar_to_imu.size())
        throw std::invalid_argument("Sensor topics and extrinsics must have the same count");
    if (!std::isfinite(config_.preview_voxel) || config_.preview_voxel <= 0 ||
        config_.preview_max_points == 0)
        throw std::invalid_argument("Preview voxel and maximum point count must be positive");
    if (config_.gt_align_initial || config_.gt_evaluate || config_.graph.use_gt) {
        if (config_.gt_file.empty()) throw std::invalid_argument("GT enabled but gt.file is empty");
        gt_ = GroundTruth::load(config_.gt_file, config_.gt_lidar_lever_arm);
    }
    if (config_.graph.use_gt && !config_.gt_align_initial)
        throw std::invalid_argument("GT position factors require gt.align_initial to enter the ENU frame");
    if (config_.slam_enabled) {
        graph_ = std::make_unique<MapOptimization>(config_.graph);
        if (config_.graph.use_gt)
            graph_->setGroundTruth(gt_->timesNs(), gt_->positions(),
                                   config_.gt_covariance, config_.gt_huber_delta);
    } else if (config_.graph.use_gt) {
        throw std::invalid_argument("GT graph factors require slam.enabled");
    }
}

std::optional<PipelineUpdate> Pipeline::process(const SynchronizedBatch& batch) {
    if (finalized_) throw std::logic_error("Pipeline is finalized; restart the node to process more data");
    if (batch.scans.size() != config_.sensors.size() || batch.imu.empty())
        throw std::invalid_argument("Incomplete synchronized batch");
    if (!epoch_ns_) epoch_ns_ = std::min(batch.start_ns, batch.imu.front().stamp_ns);
    std::vector<TimedScan> scans;
    std::vector<ImuSample> imu;
    for (const auto& item : batch.scans) {
        TimedScan scan = item.scan;
        scan.begin_seconds = (item.start_ns - *epoch_ns_) * 1e-9;
        scans.push_back(std::move(scan));
    }
    for (const auto& item : batch.imu) {
        ImuSample sample = item.sample;
        sample.time_seconds = (item.stamp_ns - *epoch_ns_) * 1e-9;
        imu.push_back(sample);
    }
    FrontendResult result = frontend_.process(scans, imu);
    Cloud combined;
    for (const auto& sensor : result.clouds) combined += sensor.points;
    if (combined.empty()) return std::nullopt;  // IMU initialization or unsupported startup spline.
    const std::int64_t stamp = *epoch_ns_ + std::llround(result.end_seconds * 1e9);
    if (!times_.empty() && stamp <= times_.back())
        throw std::runtime_error("Frontend returned a non-increasing reference timestamp");
    if (times_.empty() && config_.gt_align_initial)
        alignment_ = gt_->alignment(stamp, result.center_pose);
    const Matrix4 aligned = alignment_ * result.center_pose;
    const Eigen::Matrix3d rotation = alignment_.block<3, 3>(0, 0);
    const Eigen::Matrix3d covariance = rotation * result.position_covariance * rotation.transpose();
    if (graph_) graph_->addScan(stamp, combined, aligned, covariance);
    storage_.append(combined);
    times_.push_back(stamp);
    raw_poses_.push_back(aligned);

    PipelineUpdate update;
    update.stamp_ns = stamp;
    update.odom_pose = result.center_pose;
    update.map_pose = graph_ ? graph_->latestPose() : aligned;
    update.position_covariance = result.position_covariance;
    update.graph = graph_ ? graph_->stats() : GraphStats{};
    update.effective_points = result.effective_points;
    update.unsupported_points = result.unsupported_points;
    update.submap_changed = update.graph.submaps != last_submaps_;
    // Preview is deliberately small; every dense scan remains in the disk spool.
    if (times_.size() == 1 || update.submap_changed || (!graph_ && times_.size() % 10 == 0))
        preview_frames_.push_back({times_.size() - 1, downsample(combined, config_.preview_voxel)});
    last_submaps_ = update.graph.submaps;
    update.cloud = std::move(combined);
    return update;
}

GraphResult Pipeline::trajectory() const {
    GraphResult result;
    if (graph_) {
        result = graph_->snapshot();
        if (config_.interpolate_pose_corrections && !result.poses.empty())
            result.poses = interpolateCorrections(raw_poses_, result.poses, result.submap_start_indices);
    } else {
        result.times_ns = times_;
        result.poses = raw_poses_;
        result.stats.scans = times_.size();
    }
    return result;
}

Cloud Pipeline::preview() const {
    const auto solution = trajectory();
    Cloud map;
    for (const auto& frame : preview_frames_) {
        if (frame.index >= solution.poses.size()) continue;
        Cloud transformed;
        pcl::transformPointCloud(frame.cloud, transformed, solution.poses[frame.index]);
        map += transformed;
    }
    if (map.empty()) return map;
    map = downsample(map, config_.preview_voxel);
    if (map.size() > config_.preview_max_points) {
        Cloud bounded;
        bounded.reserve(config_.preview_max_points);
        for (std::size_t i = 0; i < config_.preview_max_points; ++i)
            bounded.push_back(map[i * map.size() / config_.preview_max_points]);
        return bounded;
    }
    return map;
}

std::string Pipeline::finalize() {
    if (!finalized_) {
        if (graph_ && !times_.empty()) graph_->finalize();
        finalized_ = true;
    }
    const auto solution = trajectory();
    std::ostringstream report;
    report << "MA-SLAM: " << times_.size() << " scans, " << solution.stats.submaps
           << " submaps, " << solution.stats.closures << " loops, " << solution.stats.gt_edges
           << " GT factors\nLoop residual: " << solution.stats.loop_translation_rmse_m
           << " m / " << solution.stats.loop_rotation_rmse_deg << " deg; minimum robust weight: "
           << solution.stats.loop_min_robust_weight;
    if (gt_ && config_.gt_evaluate && !times_.empty()) {
        if (config_.gt_align_initial)
            report << '\n' << GroundTruth::report(gt_->evaluate(solution.times_ns, solution.poses),
                                                  config_.graph.use_gt);
        else report << "\nATE skipped: trajectory is not in GT ENU (gt.align_initial=false).";
    }
    return report.str();
}

SavedMap Pipeline::save() {
    if (!finalized_) finalize();
    if (times_.empty()) throw std::runtime_error("No registered scans; nothing to save");
    const auto solution = trajectory();
    return storage_.save(config_.output, solution.times_ns, solution.poses);
}

}  // namespace ma_slam
