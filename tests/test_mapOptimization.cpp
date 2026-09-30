#include "ma_slam/mapOptimization.hpp"
#include <Eigen/Geometry>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Eigen::Matrix4d pose(double x, double y = 0.0, double yaw = 0.0) {
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    result.topLeftCorner<3, 3>() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    result(0, 3) = x;
    result(1, 3) = y;
    return result;
}

pcl::PointCloud<pcl::PointXYZI> room(bool planar) {
    pcl::PointCloud<pcl::PointXYZI> points;
    auto append = [&](float x, float y, float z) {
        pcl::PointXYZI point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.intensity = 17.0f;
        points.push_back(point);
    };
    for (int i = 0; i <= 40; ++i) {
        for (int j = 0; j <= 30; ++j) {
            const float x = -2.0f + 0.2f * i;
            const float y = -2.0f + 0.2f * j;
            append(x, y, 0.0f);
            if (!planar) {
                append(-2.0f, x, 0.1f * j);
                append(x, 6.0f, 0.1f * j);
            }
        }
    }
    return points;
}

pcl::PointCloud<pcl::PointXYZI> observe(const pcl::PointCloud<pcl::PointXYZI>& world,
                                      const Eigen::Matrix4d& world_from_lidar) {
    pcl::PointCloud<pcl::PointXYZI> local;
    const Eigen::Matrix4d lidar_from_world = world_from_lidar.inverse();
    for (const auto& source : world) {
        const Eigen::Vector4d p = lidar_from_world * Eigen::Vector4d(source.x, source.y, source.z, 1.0);
        pcl::PointXYZI point = source;
        point.x = static_cast<float>(p.x());
        point.y = static_cast<float>(p.y());
        point.z = static_cast<float>(p.z());
        local.push_back(point);
    }
    return local;
}

ma_slam::GraphResult closeLoop(double sigma, bool planar = false) {
    ma_slam::GraphConfig config;
    config.submap_voxel = 0.15;
    config.submap_distance = 1.0;
    config.min_range = 0.0;
    config.loop_search_radius = 1.5;
    config.closure_min_time_separation = 0.0;
    config.closure_fine_distance = 0.5;
    config.lidar_position_sigma = sigma;
    ma_slam::MapOptimization graph(config);
    const auto world = room(planar);
    const std::vector<Eigen::Vector2d> path = {{0, 0}, {2, 0}, {4, 0}, {4, 2}, {4, 4},
                                               {2, 4}, {0, 4}, {0, 2}, {0, 0}};
    require(graph.latestPose().isApprox(Eigen::Matrix4d::Identity()), "Empty graph pose is not identity.");
    for (std::size_t i = 0; i < path.size(); ++i) {
        const double alpha = double(i) / (path.size() - 1);
        const auto truth = pose(path[i].x(), path[i].y());
        const auto odom = pose(path[i].x() + .6 * alpha, path[i].y() - .3 * alpha, .024 * alpha);
        graph.addScan(static_cast<std::int64_t>(i + 1) * 1000000000LL,
                      observe(world, truth), odom, Eigen::Matrix3d::Identity() * 0.01);
        const auto partial = graph.snapshot();
        require(partial.poses.size() == i + 1, "Snapshot discarded processed scans.");
        require(partial.poses.back().isApprox(graph.latestPose()), "Live pose and snapshot disagree.");
    }
    graph.finalize();
    graph.finalize();  // Finalization is idempotent.
    return graph.trajectory();
}

void checkBoundaryGT() {
    ma_slam::GraphConfig config;
    config.use_gt = true;
    config.gt_gate_covariance = false;
    config.gt_gate_graph_covariance = false;
    config.submap_distance = 1.0;
    config.loop_search_radius = 0.0;
    config.min_range = 0.0;
    ma_slam::MapOptimization graph(config);
    // 1.5 s is between the old submap's final scan (1 s) and the new anchor (2 s).
    const std::vector<std::int64_t> times = {0, 1500000000LL, 2000000000LL, 2500000000LL, 3000000000LL};
    const std::vector<Eigen::Vector3d> positions = {{0, 0, 0}, {.95, 0, 0}, {1.5, 0, 0},
                                                  {1.7, 0, 0}, {1.9, 0, 0}};
    graph.setGroundTruth(times, positions, Eigen::Matrix3d::Identity() * .01, 3.0);
    const double xs[] = {0.0, 0.4, 1.5, 1.9};
    const auto cloud = room(false);
    for (std::size_t i = 0; i < 4; ++i)
        graph.addScan(i * 1000000000LL, cloud, pose(xs[i]), Eigen::Matrix3d::Identity());
    graph.finalize();
    const auto result = graph.trajectory();
    require(result.gt_factor_times_ns == times, "GT boundary samples were lost, duplicated or assigned late.");
    require(result.stats.gt_edges == times.size(), "GT edge count differs from accepted timestamps.");
    require(result.submap_start_indices == std::vector<std::size_t>({0, 2}), "Unexpected submap starts.");
    require(result.poses.back().topRightCorner<3, 1>().isApprox(Eigen::Vector3d(1.9, 0, 0), 1e-5),
            "Boundary GT interpolation introduced position error.");
}

void checkCovarianceGate() {
    const std::vector<Eigen::Vector3d> route = {{0, 0, 0}, {2.5, 0, 0}, {5, 0, 0}, {5, 2.5, 0},
        {5, 5, 0}, {2.5, 5, 0}, {0, 5, 0}, {0, 2.5, 0}, {0, 0, 0}, {2.5, 0, 0}};
    std::vector<std::int64_t> times;
    for (std::size_t i = 0; i < route.size(); ++i) times.push_back((i * 10 + 1) * 1000000000LL);
    const auto cloud = room(false);
    for (bool graph_covariance : {false, true}) {
        for (bool activate : {false, true}) {
            ma_slam::GraphConfig config;
            config.use_gt = true;
            config.submap_distance = 2.0;
            config.loop_search_radius = 0.0;
            config.min_range = 0.0;
            config.gt_gate_graph_covariance = graph_covariance;
            config.gt_covariance_threshold = activate ? .04 : 1e6;
            ma_slam::MapOptimization graph(config);
            graph.setGroundTruth(times, route, Eigen::Matrix3d::Identity() * .01, 3.0);
            double before_squared = 0.0;
            for (std::size_t i = 0; i < route.size(); ++i) {
                const auto odom = pose(route[i].x() + .05 * i, route[i].y());
                const double variance = !graph_covariance && i >= 4 ? .25 : .001;
                graph.addScan(times[i], cloud, odom, Eigen::Matrix3d::Identity() * variance);
                before_squared += .05 * i * .05 * i;
            }
            graph.finalize();
            const auto result = graph.trajectory();
            require((result.stats.gt_edges > 0) == activate, "GT covariance gate selected the wrong samples.");
            require(result.stats.gt_skipped_covariance > 0, "Low uncertainty unexpectedly added GT factors.");
            if (activate) {
                double after_squared = 0.0;
                for (std::size_t i = 0; i < route.size(); ++i)
                    after_squared += (result.poses[i].topRightCorner<3, 1>() - route[i]).squaredNorm();
                require(after_squared < before_squared * .49, "Conditional GT factors did not reduce drift.");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        checkBoundaryGT();
        checkCovarianceGate();
        const auto normal = closeLoop(0.1);
        const auto rescaled = closeLoop(1.0);
        require(normal.stats.closures > 0, "Known non-degenerate revisit was not closed.");
        require(normal.stats.closures == rescaled.stats.closures, "Information rescaling changed loop selection.");
        require(normal.poses.back().topRightCorner<3, 1>().norm() < .2,
                "Loop optimization did not correct the known 0.67 m drift.");
        for (std::size_t i = 0; i < normal.poses.size(); ++i)
            require(normal.poses[i].isApprox(rescaled.poses[i], 2e-4),
                    "Cauchy robust cutoff was not scaled with edge information.");
        require(normal.stats.loop_rotation_rmse_deg < .5, "Loop rotation residual has the wrong angular scale.");
        const auto planar = closeLoop(.1, true);
        require(planar.stats.loop_candidates > 0 && planar.stats.closures == 0,
                "A plane-only, rank-deficient loop was accepted.");
        std::cout << "PASS: GT boundary samples, graph/LIO covariance gates, known loop correction, robust scaling, angular residual, "
                     "plane rejection and live snapshots.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
