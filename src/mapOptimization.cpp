#include "ma_slam/mapOptimization.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <g2o/core/block_solver.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/factory.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/edge_se3_prior.h>
#include <g2o/types/slam3d/parameter_se3_offset.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/centroid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

G2O_USE_TYPE_GROUP(slam3d);

namespace ma_slam {

using Cloud = pcl::PointCloud<pcl::PointXYZ>;
using SurfaceCloud = pcl::PointCloud<pcl::PointNormal>;
using Pose = Eigen::Isometry3d;

class PositionEdge final : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3> {
public:
    explicit PositionEdge(Eigen::Vector3d local) : local_(std::move(local)) {}
    void computeError() override { _error = vertexXn<0>()->estimate() * local_ - _measurement; }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
private:
    Eigen::Vector3d local_;
};

struct Submap {
    std::size_t start;
    Pose odom;
    std::vector<std::int64_t> times;
    std::vector<Pose> local_poses;
    Cloud::Ptr cloud{new Cloud};
    SurfaceCloud::Ptr surfaces{new SurfaceCloud};
};

class MapOptimization::Impl {
public:
    explicit Impl(const GraphConfig& c)
        : voxel_(c.submap_voxel), split_(c.submap_distance),
          max_range_(c.max_range), min_range_(c.min_range),
          loop_radius_(c.loop_search_radius), max_correction_(c.loop_max_correction),
          overlap_threshold_(c.closure_overlap_threshold), max_rmse_(c.closure_max_rmse),
          fine_distance_(c.closure_fine_distance), loop_time_(c.closure_min_time_separation),
          max_candidates_(c.closure_max_candidates), skip_submaps_(c.no_of_sub_maps_to_skip),
          iterations_(c.graph_max_iterations), gt_enabled_(c.use_gt),
          gate_covariance_(c.gt_gate_covariance), graph_covariance_(c.gt_gate_graph_covariance),
          covariance_threshold_(c.gt_covariance_threshold), covariance_ratio_(c.gt_covariance_ratio),
          min_gt_distance_(c.gt_min_factor_distance),
          min_gt_interval_ns_(static_cast<std::int64_t>(std::llround(c.gt_min_factor_interval * 1e9))),
          closure_kernel_delta_(1. / c.lidar_position_sigma),
          odom_information_(1. / (c.lidar_position_sigma * c.lidar_position_sigma)) {
        const double values[] = {c.submap_voxel, c.submap_distance, c.min_range, c.max_range,
            c.lidar_position_sigma, c.loop_search_radius, c.loop_max_correction,
            c.closure_overlap_threshold, c.closure_max_rmse, c.closure_fine_distance,
            c.closure_min_time_separation, c.gt_covariance_threshold, c.gt_covariance_ratio,
            c.gt_min_factor_distance, c.gt_min_factor_interval};
        if (!std::all_of(std::begin(values), std::end(values), [](double v) { return std::isfinite(v); }) ||
            !(voxel_ > 0 && split_ > 0 && max_range_ > min_range_ && min_range_ >= 0 &&
              c.lidar_position_sigma > 0 && fine_distance_ > 0 && iterations_ > 0 &&
              max_candidates_ > 0 && skip_submaps_ > 0 && loop_radius_ >= 0 &&
              max_correction_ > 0 && overlap_threshold_ > 0 && overlap_threshold_ <= 1 &&
              max_rmse_ > 0 && loop_time_ >= 0 && covariance_threshold_ >= 0 &&
              covariance_ratio_ >= 0 && min_gt_distance_ >= 0 && c.gt_min_factor_interval >= 0 &&
              c.gt_min_factor_interval < 1e6))
            throw std::invalid_argument("Invalid MA-SLAM graph settings.");
        using Solver = g2o::BlockSolverX;
        auto linear = std::make_unique<g2o::LinearSolverEigen<Solver::PoseMatrixType>>();
        graph_.setAlgorithm(new g2o::OptimizationAlgorithmDogleg(
            std::make_unique<Solver>(std::move(linear))));
    }

    void set_gt(const std::vector<std::int64_t>& times, const Eigen::MatrixXd& positions,
                const Eigen::Matrix3d& covariance, double huber) {
        if (!gt_enabled_ || !maps_.empty() || gt_ready_ || times.size() < 2 ||
            positions.rows() != static_cast<Eigen::Index>(times.size()) || positions.cols() != 3 ||
            !positions.allFinite() || !covariance.allFinite() ||
            !covariance.isApprox(covariance.transpose(), 1e-10) ||
            Eigen::LLT<Eigen::Matrix3d>(covariance).info() != Eigen::Success ||
            !std::isfinite(huber) || huber <= 0)
            throw std::invalid_argument("Invalid GT samples or covariance.");
        for (std::size_t i = 1; i < times.size(); ++i)
            if (times[i] <= times[i - 1]) throw std::invalid_argument("GT times must increase.");
        gt_times_ = times;
        gt_positions_ = positions;
        gt_covariance_ = covariance;
        huber_ = huber;
        gt_ready_ = true;
    }

    bool add_scan(std::int64_t time, const pcl::PointCloud<pcl::PointXYZI>& points,
                  const Eigen::Matrix4d& matrix, const Eigen::Matrix3d& covariance) {
        if (finalized_) throw std::runtime_error("Graph is finalized.");
        if (gt_enabled_ && !gt_ready_) throw std::runtime_error("Call setGroundTruth first.");
        if (!matrix.allFinite() ||
            !matrix.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8) ||
            !(matrix.topLeftCorner<3, 3>().transpose() * matrix.topLeftCorner<3, 3>())
                .isApprox(Eigen::Matrix3d::Identity(), 1e-5) ||
            matrix.topLeftCorner<3, 3>().determinant() < 0.0)
            throw std::invalid_argument("Scan pose is not a finite SE(3) transform.");
        if (!maps_.empty() && time <= maps_.back().times.back())
            throw std::invalid_argument("Scan times must increase.");
        if (!covariance.allFinite() || !covariance.isApprox(covariance.transpose(), 1e-9) ||
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(covariance).eigenvalues().minCoeff() < -1e-9)
            throw std::invalid_argument("Invalid LiDAR position covariance.");
        covariances_.emplace_back(time, covariance);
        Pose pose = Pose::Identity();
        pose.matrix() = matrix;
        bool split = false;
        if (maps_.empty()) {
            add_vertex(pose);
            maps_.push_back(Submap{0, pose});
        } else if ((pose.translation() - maps_.back().odom.translation()).norm() >= split_) {
            close_submap(maps_.size() - 1, &pose, time);
            const auto previous = keypose(maps_.size() - 1);
            const auto relative = maps_.back().odom.inverse() * pose;
            add_vertex(previous * relative);
            add_relative_edge(maps_.size() - 1, maps_.size(), relative, false);
            maps_.push_back(Submap{total_scans_, pose});
            split = true;
        }
        auto& map = maps_.back();
        map.times.push_back(time);
        map.local_poses.push_back(map.odom.inverse() * pose);
        for (const auto& p : points) {
            const Eigen::Vector3d point(p.x, p.y, p.z);
            if (!point.allFinite() || point.norm() < min_range_ || point.norm() > max_range_) continue;
            const auto local = map.local_poses.back() * point;
            map.cloud->push_back(pcl::PointXYZ(static_cast<float>(local.x()),
                                               static_cast<float>(local.y()),
                                               static_cast<float>(local.z())));
        }
        ++total_scans_;
        return split;
    }

    void finalize() {
        if (finalized_) return;
        if (!maps_.empty()) close_submap(maps_.size() - 1);
        finalized_ = true;
    }

    GraphResult snapshot() const {
        GraphResult result;
        result.times_ns.reserve(total_scans_);
        result.poses.reserve(total_scans_);
        for (std::size_t id = 0; id < maps_.size(); ++id) {
            result.submap_start_indices.push_back(maps_[id].start);
            for (std::size_t j = 0; j < maps_[id].times.size(); ++j) {
                result.times_ns.push_back(maps_[id].times[j]);
                result.poses.push_back((keypose(id) * maps_[id].local_poses[j]).matrix());
            }
        }
        result.loop_pairs = loop_pairs_;
        result.gt_factor_times_ns = accepted_gt_times_;
        result.stats = stats();
        return result;
    }

    GraphResult trajectory() const {
        if (!finalized_) throw std::runtime_error("Call finalize before requesting the final trajectory.");
        return snapshot();
    }

    Eigen::Matrix4d latestPose() const {
        if (maps_.empty()) return Eigen::Matrix4d::Identity();
        return (keypose(maps_.size() - 1) * maps_.back().local_poses.back()).matrix();
    }

    GraphStats stats() const {
        GraphStats s;
        s.scans = total_scans_;
        s.submaps = maps_.size();
        s.closures = loop_pairs_.size();
        s.gt_edges = gt_edges_;
        s.gt_candidates = gt_candidates_;
        s.gt_skipped_covariance = skipped_covariance_;
        s.gt_skipped_distance = skipped_distance_;
        s.loop_candidates = loop_candidates_;
        s.loop_rejected = loop_rejected_;
        double translation_squared = 0., rotation_squared = 0., minimum_weight = 1.;
        for (auto* edge : loop_edges_) {
            edge->computeError();
            translation_squared += edge->error().head<3>().squaredNorm();
            const double angle = 2. * std::asin(std::min(1., edge->error().tail<3>().norm()));
            rotation_squared += angle * angle;
            minimum_weight = std::min(minimum_weight,
                1. / (1. + edge->chi2() / (closure_kernel_delta_ * closure_kernel_delta_)));
        }
        const double count = std::max(std::size_t{1}, loop_edges_.size());
        s.loop_translation_rmse_m = std::sqrt(translation_squared / count);
        s.loop_rotation_rmse_deg = std::sqrt(rotation_squared / count) * 180. / std::acos(-1.0);
        s.loop_min_robust_weight = minimum_weight;
        return s;
    }

private:
    g2o::VertexSE3* vertex(std::size_t id) {
        return static_cast<g2o::VertexSE3*>(graph_.vertex(static_cast<int>(id)));
    }
    const g2o::VertexSE3* vertex(std::size_t id) const {
        return static_cast<const g2o::VertexSE3*>(graph_.vertex(static_cast<int>(id)));
    }
    Pose keypose(std::size_t id) const { return vertex(id)->estimate(); }
    void add_vertex(const Pose& pose) {
        auto* v = new g2o::VertexSE3;
        v->setId(static_cast<int>(maps_.size()));
        v->setEstimate(pose);
        if (maps_.empty() && !gt_enabled_) v->setFixed(true);
        graph_.addVertex(v);
        if (maps_.empty() && gt_enabled_) {
            auto* offset = new g2o::ParameterSE3Offset;
            offset->setId(0);
            if (!graph_.addParameter(offset)) throw std::runtime_error("g2o gauge parameter failed.");
            auto* prior = new g2o::EdgeSE3Prior;
            prior->setVertex(0, v);
            prior->setMeasurement(pose);
            g2o::EdgeSE3Prior::InformationType gauge =
                g2o::EdgeSE3Prior::InformationType::Identity();
            gauge.topLeftCorner<3, 3>() *= 1e4;  // Initial ENU position is already GT aligned.
            gauge.bottomRightCorner<3, 3>() *= 1e4;  // Initial IMU tilt/GT heading are observed.
            prior->setInformation(gauge);
            prior->setParameterId(0, 0);
            if (!graph_.addEdge(prior)) throw std::runtime_error("g2o gauge edge failed.");
        }
    }
    g2o::EdgeSE3* add_relative_edge(std::size_t from, std::size_t to,
                                    const Pose& relative, bool closure) {
        auto* edge = new g2o::EdgeSE3;
        edge->setVertex(0, vertex(from));
        edge->setVertex(1, vertex(to));
        edge->setMeasurement(relative);
        g2o::EdgeSE3::InformationType information =
            g2o::EdgeSE3::InformationType::Identity() * odom_information_;
        // EdgeSE3 uses quaternion xyz (~angle/2), not the SE(3) rotation vector.
        // Preserve the former 100:1 angular/translation information in radians.
        information.bottomRightCorner<3, 3>() *= 4. * 100.;
        edge->setInformation(information);
        if (closure) {
            auto* kernel = new g2o::RobustKernelCauchy;
            // Scaling information by 1/sigma^2 also scales the robust cutoff by 1/sigma.
            // Otherwise valid 0.5 m closures have only ~4% of their intended weight.
            kernel->setDelta(closure_kernel_delta_);
            edge->setRobustKernel(kernel);
        }
        if (!graph_.addEdge(edge)) throw std::runtime_error("g2o relative edge failed.");
        return edge;
    }
    bool optimize() {
        if (graph_.edges().empty()) return true;
        graph_.initializeOptimization();
        return graph_.optimize(iterations_) > 0;
    }
    Eigen::Matrix3d marginal(std::size_t id, const Eigen::Vector3d& local) {
        auto* v = vertex(id);
        if (v->fixed()) return Eigen::Matrix3d::Zero();
        g2o::SparseBlockMatrix<g2o::MatrixX> inverse;
        if (!graph_.computeMarginals(inverse, v)) throw std::runtime_error("g2o marginal failed.");
        const auto* block = inverse.block(v->hessianIndex(), v->hessianIndex());
        if (!block || !block->allFinite()) throw std::runtime_error("Invalid g2o covariance.");
        Eigen::Matrix<double, 3, 6> jacobian;
        for (int k = 0; k < 6; ++k) {
            Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Zero();
            delta[k] = 1e-6;
            v->push(); v->oplus(delta.data());
            auto plus = (v->estimate() * local).eval();
            v->pop(); delta[k] = -1e-6;
            v->push(); v->oplus(delta.data());
            auto minus = (v->estimate() * local).eval();
            v->pop();
            jacobian.col(k) = (plus - minus) / 2e-6;
        }
        return jacobian * *block * jacobian.transpose();
    }
    void add_gt(std::size_t id, const Pose* next_pose, std::int64_t next_time) {
        if (!gt_enabled_) return;
        const auto& map = maps_[id];
        while (gt_cursor_ < gt_times_.size() && gt_times_[gt_cursor_] < map.times.front()) ++gt_cursor_;
        if (graph_covariance_ && !optimize()) throw std::runtime_error("g2o odometry solve failed.");
        while (gt_cursor_ < gt_times_.size() &&
               (next_pose ? gt_times_[gt_cursor_] < next_time : gt_times_[gt_cursor_] <= map.times.back())) {
            const auto time = gt_times_[gt_cursor_];
            const Eigen::Vector3d measurement = gt_positions_.row(gt_cursor_).transpose();
            ++gt_candidates_;
            const auto after = std::lower_bound(map.times.begin(), map.times.end(), time);
            std::size_t k = static_cast<std::size_t>(after - map.times.begin());
            if (k == map.times.size()) k = map.times.size() - 1;
            Eigen::Vector3d local = map.local_poses[k].translation();
            if (next_pose && time > map.times.back()) {
                // Retain GT samples between the last scan here and the next submap anchor.
                const double alpha = double(time - map.times.back()) / double(next_time - map.times.back());
                const Eigen::Vector3d next_local = (map.odom.inverse() * *next_pose).translation();
                local = (1 - alpha) * local + alpha * next_local;
            } else if (k > 0 && time < map.times[k]) {
                const double alpha = double(time - map.times[k - 1]) /
                                     double(map.times[k] - map.times[k - 1]);
                local = (1 - alpha) * map.local_poses[k - 1].translation() +
                        alpha * map.local_poses[k].translation();
            }
            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            if (gate_covariance_) {
                if (graph_covariance_) covariance = marginal(id, local);
                else {
                    const auto found = std::lower_bound(covariances_.begin(), covariances_.end(), time,
                        [](const auto& sample, std::int64_t stamp) { return sample.first < stamp; });
                    if (found == covariances_.end() || (found == covariances_.begin() && found->first != time)) {
                        ++skipped_covariance_; ++gt_cursor_; continue;
                    }
                    covariance = found->second;
                    if (found->first != time) {
                        const auto before = std::prev(found);
                        const double alpha = double(time - before->first) / double(found->first - before->first);
                        covariance = (1 - alpha) * before->second + alpha * found->second;
                    }
                }
                const double variance = std::max(covariance(0, 0), covariance(1, 1));
                const double gt_variance = std::max(gt_covariance_(0, 0), gt_covariance_(1, 1));
                if (variance <= covariance_threshold_ || variance <= covariance_ratio_ * gt_variance) {
                    ++skipped_covariance_; ++gt_cursor_; continue;
                }
                if (!accepted_gt_times_.empty() &&
                    (measurement - last_gt_position_).norm() < min_gt_distance_ &&
                    (min_gt_interval_ns_ == 0 || time - accepted_gt_times_.back() < min_gt_interval_ns_)) {
                    ++skipped_distance_; ++gt_cursor_; continue;
                }
            }
            auto* edge = new PositionEdge(local);
            edge->setVertex(0, vertex(id));
            edge->setMeasurement(measurement);
            edge->setInformation(gt_covariance_.inverse());
            auto* kernel = new g2o::RobustKernelHuber;
            kernel->setDelta(huber_);
            edge->setRobustKernel(kernel);
            if (!graph_.addEdge(edge)) throw std::runtime_error("g2o GT edge failed.");
            accepted_gt_times_.push_back(time);
            last_gt_position_ = measurement;
            ++gt_edges_; ++gt_cursor_;
            if (graph_covariance_ && !optimize()) throw std::runtime_error("g2o GT solve failed.");
        }
        if (gt_edges_ && !optimize()) throw std::runtime_error("g2o GT solve failed.");
    }
    void close_submap(std::size_t id, const Pose* next_pose = nullptr, std::int64_t next_time = 0) {
        auto& map = maps_[id];
        pcl::VoxelGrid<pcl::PointXYZ> filter;
        filter.setInputCloud(map.cloud);
        filter.setLeafSize(voxel_, voxel_, voxel_);
        auto reduced = Cloud::Ptr(new Cloud);
        filter.filter(*reduced);
        map.cloud = reduced;
        // Surface normals are computed once per closed submap and reused for revisits.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normals;
        normals.setInputCloud(map.cloud);
        normals.setKSearch(12);
        pcl::PointCloud<pcl::Normal> estimated;
        normals.compute(estimated);
        for (std::size_t k = 0; k < estimated.size(); ++k) {
            const auto& normal = estimated[k];
            if (!std::isfinite(normal.normal_x) || !std::isfinite(normal.curvature) || normal.curvature > .15f) continue;
            pcl::PointNormal point;
            point.getVector3fMap() = (*map.cloud)[k].getVector3fMap();
            point.getNormalVector3fMap() = normal.getNormalVector3fMap();
            point.curvature = normal.curvature;
            map.surfaces->push_back(point);
        }
        add_gt(id, next_pose, next_time);
        if (loop_radius_ <= 0 || id <= static_cast<std::size_t>(skip_submaps_)) return;
        std::vector<std::pair<double, std::size_t>> candidates;
        for (std::size_t target = 0; target + skip_submaps_ < id; ++target) {
            if (double(map.times.front() - maps_[target].times.front()) * 1e-9 < loop_time_) continue;
            const double distance = (keypose(id).translation() - keypose(target).translation()).norm();
            if (distance <= loop_radius_) candidates.emplace_back(distance, target);
        }
        std::sort(candidates.begin(), candidates.end());
        int tested = 0;
        for (const auto& [distance, target] : candidates) {
            if (tested++ >= max_candidates_) break;
            ++loop_candidates_;
            const auto& fixed = maps_[target];
            if (map.cloud->size() < 30 || fixed.cloud->size() < 30) { ++loop_rejected_; continue; }
            Eigen::Vector4f mean;
            Eigen::Matrix3f covariance;
            pcl::computeMeanAndCovarianceMatrix(*fixed.cloud, covariance, mean);
            if (Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f>(covariance).eigenvalues().minCoeff() < .01f) {
                ++loop_rejected_; continue;
            }
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(map.cloud);
            icp.setInputTarget(fixed.cloud);
            icp.setMaxCorrespondenceDistance(std::max(1.f, fine_distance_ * 4));
            icp.setMaximumIterations(50);
            const Pose initial = keypose(target).inverse() * keypose(id);
            Cloud aligned;
            icp.align(aligned, initial.matrix().cast<float>());
            if (!icp.hasConverged()) { ++loop_rejected_; continue; }
            if (map.surfaces->size() < 30 || fixed.surfaces->size() < 30) { ++loop_rejected_; continue; }
            pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal> fine;
            fine.setInputSource(map.surfaces);
            fine.setInputTarget(fixed.surfaces);
            fine.setMaxCorrespondenceDistance(fine_distance_);
            fine.setMaximumIterations(40);
            fine.setTransformationEpsilon(1e-8);
            SurfaceCloud refined_surfaces;
            fine.align(refined_surfaces, icp.getFinalTransformation());
            if (!fine.hasConverged() || !fine.getFinalTransformation().allFinite()) { ++loop_rejected_; continue; }
            pcl::transformPointCloud(*map.cloud, aligned, fine.getFinalTransformation());
            pcl::KdTreeFLANN<pcl::PointXYZ> tree;
            tree.setInputCloud(fixed.cloud);
            std::vector<int> neighbor(1);
            std::vector<float> squared(1);
            std::size_t inliers = 0;
            double sum = 0;
            for (const auto& point : aligned) {
                if (tree.nearestKSearch(point, 1, neighbor, squared) && squared[0] <= fine_distance_ * fine_distance_) {
                    ++inliers; sum += squared[0];
                }
            }
            // A directed inlier fraction must use the source count, never the smaller cloud.
            const double overlap = double(inliers) / map.cloud->size();
            const double rmse = inliers ? std::sqrt(sum / inliers) : INFINITY;
            Pose refined = Pose::Identity();
            refined.matrix() = fine.getFinalTransformation().cast<double>();
            const Pose correction = initial.inverse() * refined;
            if (overlap < overlap_threshold_ || rmse > max_rmse_ ||
                correction.translation().norm() > max_correction_ ||
                Eigen::AngleAxisd(correction.linear()).angle() > .175) {
                ++loop_rejected_; continue;
            }
            auto* edge = add_relative_edge(target, id, refined, true);
            if (!optimize()) {
                graph_.removeEdge(edge); delete edge;
                ++loop_rejected_;
                if (!optimize()) throw std::runtime_error("g2o failed after loop rejection.");
                continue;
            }
            loop_pairs_.emplace_back(static_cast<int>(id), static_cast<int>(target));
            loop_edges_.push_back(edge);
            break;
        }
    }

    g2o::SparseOptimizer graph_;
    std::vector<Submap> maps_;
    std::size_t total_scans_ = 0;
    float voxel_, split_, max_range_, min_range_, loop_radius_, max_correction_;
    float overlap_threshold_, max_rmse_, fine_distance_, loop_time_;
    int max_candidates_, skip_submaps_, iterations_;
    bool gt_enabled_, gate_covariance_, graph_covariance_, gt_ready_ = false, finalized_ = false;
    double covariance_threshold_, covariance_ratio_, min_gt_distance_;
    std::int64_t min_gt_interval_ns_;
    double closure_kernel_delta_, odom_information_, huber_ = 3.;
    std::vector<std::int64_t> gt_times_, accepted_gt_times_;
    Eigen::MatrixXd gt_positions_;
    Eigen::Matrix3d gt_covariance_;
    std::vector<std::pair<std::int64_t, Eigen::Matrix3d>> covariances_;
    std::size_t gt_cursor_ = 0, gt_edges_ = 0, gt_candidates_ = 0;
    std::size_t skipped_covariance_ = 0, skipped_distance_ = 0, loop_candidates_ = 0, loop_rejected_ = 0;
    Eigen::Vector3d last_gt_position_ = Eigen::Vector3d::Zero();
    std::vector<std::pair<int, int>> loop_pairs_;
    std::vector<g2o::EdgeSE3*> loop_edges_;
};

MapOptimization::MapOptimization(const GraphConfig& config) : impl_(std::make_unique<Impl>(config)) {}
MapOptimization::~MapOptimization() = default;
MapOptimization::MapOptimization(MapOptimization&&) noexcept = default;
MapOptimization& MapOptimization::operator=(MapOptimization&&) noexcept = default;

void MapOptimization::setGroundTruth(const std::vector<std::int64_t>& times_ns,
                                     const std::vector<Eigen::Vector3d>& positions,
                                     const Eigen::Matrix3d& covariance, double huber_delta) {
    Eigen::MatrixXd values(positions.size(), 3);
    for (std::size_t i = 0; i < positions.size(); ++i) values.row(i) = positions[i].transpose();
    impl_->set_gt(times_ns, values, covariance, huber_delta);
}

bool MapOptimization::addScan(std::int64_t time_ns, const pcl::PointCloud<pcl::PointXYZI>& cloud,
                              const Eigen::Matrix4d& pose, const Eigen::Matrix3d& position_covariance) {
    return impl_->add_scan(time_ns, cloud, pose, position_covariance);
}

void MapOptimization::finalize() { impl_->finalize(); }
GraphResult MapOptimization::trajectory() const { return impl_->trajectory(); }
GraphResult MapOptimization::snapshot() const { return impl_->snapshot(); }
Eigen::Matrix4d MapOptimization::latestPose() const { return impl_->latestPose(); }
GraphStats MapOptimization::stats() const { return impl_->stats(); }

}  // namespace ma_slam
