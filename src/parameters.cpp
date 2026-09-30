#include "ma_slam/parameters.hpp"

#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace ma_slam {
namespace {

void require(bool condition, const std::string& message)
{
    // -------------------------------------------------------------
    // - Reject invalid user parameters before estimator state is created.
    // -------------------------------------------------------------
    if (!condition) throw std::invalid_argument("MA-SLAM parameter: " + message);
}

std::vector<double> defaultExtrinsic(int sensor)
{
    // -------------------------------------------------------------
    // - UrbanNav TST-20210517 calibrated transforms, expressed as LiDAR -> IMU.
    // - Compose CENTER_LiDAR_T_IMU * {RIGHT,LEFT}_LiDAR_T_CENTER_LiDAR.
    // - Values are row-major; only the published rounding error is normalized.
    // -------------------------------------------------------------
    if (sensor == 0) return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, .28, 0, 0, 0, 1};
    if (sensor == 1) return {
        .566085, .0347042, .823616, .323744,
        -.0296934, .999323, -.0216991, -.00124153,
        -.823812, -.0121725, .566732, .079124,
        0, 0, 0, 1};
    return {
        -.00848239, -.561875, -.827179, -.267094,
        .999415, -.0321631, .0115987, -.000706537,
        -.0331216, -.826597, .561819, .055962,
        0, 0, 0, 1};
}

Matrix4 rigidTransform(const std::vector<double>& values, const std::string& name)
{
    // -------------------------------------------------------------
    // - Validate one finite rigid 4x4 matrix before rounding onto SO(3).
    // - Reject reflections, scale/shear and matrix-order mistakes.
    // -------------------------------------------------------------
    require(values.size() == 16, name + " must contain 16 row-major numbers");
    Matrix4 transform = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(values.data());
    require(transform.allFinite(), name + " contains non-finite values");
    require(transform.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-10),
            name + " last row must be [0, 0, 0, 1]");
    const Eigen::Matrix3d rotation = transform.topLeftCorner<3, 3>();
    require((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() <= 1e-4 &&
                rotation.determinant() > 0,
            name + " rotation must be near SO(3)");
    const Eigen::JacobiSVD<Eigen::Matrix3d> decomposition(rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);
    transform.topLeftCorner<3, 3>() = decomposition.matrixU() * decomposition.matrixV().transpose();
    return transform;
}

}  // namespace

Parameters readParameters(const ParameterGetter& get)
{
    // -------------------------------------------------------------
    // - Read typed startup parameters from ROS2 or flattened notebook YAML.
    // - Keep sensor timestamps/extrinsics explicit and independent per LiDAR.
    // - Validate all settings before starting subscriptions and worker threads.
    // -------------------------------------------------------------
    Parameters parameters;
    auto& config = parameters.pipeline;
    const auto real = [&](const std::string& name, double value) {
        const double result = std::get<double>(get(name, value));
        require(std::isfinite(result), name + " must be finite");
        return result;
    };
    const auto integer = [&](const std::string& name, int value) {
        const auto result = std::get<std::int64_t>(get(name, static_cast<std::int64_t>(value)));
        require(result >= std::numeric_limits<int>::min() && result <= std::numeric_limits<int>::max(),
                name + " is outside the supported integer range");
        return static_cast<int>(result);
    };
    const auto positiveCount = [&](const std::string& name, int value) {
        const int result = integer(name, value);
        require(result > 0, name + " must be positive");
        return static_cast<std::size_t>(result);
    };
    const auto flag = [&](const std::string& name, bool value) {
        return std::get<bool>(get(name, value));
    };
    const auto text = [&](const std::string& name, const std::string& value) {
        return std::get<std::string>(get(name, value));
    };

    parameters.imu_topic = text("imu.topic", "/imu/data");
    parameters.imu_time_offset_seconds = real("imu.time_offset_seconds", 0);
    parameters.pair_tolerance_seconds = real("synchronization.pair_tolerance_seconds", .08);
    parameters.scan_queue_limit = positiveCount("synchronization.scan_queue_limit", 100);
    parameters.imu_queue_limit = positiveCount("synchronization.imu_queue_limit", 20000);
    require(!parameters.imu_topic.empty(), "imu.topic must not be empty");
    require(parameters.pair_tolerance_seconds > 0, "synchronization.pair_tolerance_seconds must be positive");
    require(parameters.imu_queue_limit >= 17, "synchronization.imu_queue_limit must permit 15 future IMU samples");

    const int count = integer("sensors.count", ImuPreintegration::lidarCount());
    require(count == ImuPreintegration::lidarCount(), "sensors.count must match MA_SLAM_LIDAR_COUNT at build time");
    const std::vector<std::string> default_topics = {
        "/velodyne_points", "/right/velodyne_points", "/left/lslidar_point_cloud"};
    const std::vector<std::string> default_frames = {"velodyne", "velodyne_right", "laser_link"};
    std::set<std::string> seen_topics;
    for (int index = 0; index < count; ++index) {
        const std::string prefix = "sensors.lidar" + std::to_string(index) + ".";
        SensorConfig sensor;
        sensor.topic = text(prefix + "topic", default_topics[index]);
        sensor.frame_id = text(prefix + "frame_id", default_frames[index]);
        sensor.timing_mode = text(prefix + "timing_mode", index == 2 ? "azimuth_rings" : "measured");
        sensor.time_field = text(prefix + "time_field", "time");
        sensor.time_unit_seconds = real(prefix + "time_unit_seconds", 1);
        sensor.time_offset_seconds = real(prefix + "time_offset_seconds", 0);
        sensor.scan_duration_seconds = real(prefix + "scan_duration_seconds", .1);
        sensor.header_is_end = flag(prefix + "header_is_end", index == 2);
        require(!sensor.topic.empty() && seen_topics.insert(sensor.topic).second,
                prefix + "topic must be nonempty and unique");
        require(!sensor.frame_id.empty(), prefix + "frame_id must not be empty");
        require(sensor.timing_mode == "measured" || sensor.timing_mode == "azimuth_rings",
                prefix + "timing_mode must be measured or azimuth_rings");
        require(sensor.timing_mode != "measured" || !sensor.time_field.empty(),
                prefix + "time_field must be set for measured timing");
        require(sensor.time_unit_seconds > 0, prefix + "time_unit_seconds must be positive");
        require(sensor.scan_duration_seconds > 0 && sensor.scan_duration_seconds <= .5,
                prefix + "scan_duration_seconds must be in (0, 0.5]");
        config.sensors.push_back(sensor);
        config.lidar_to_imu.push_back(rigidTransform(
            std::get<std::vector<double>>(get(prefix + "lidar_to_imu", defaultExtrinsic(index))),
            prefix + "lidar_to_imu"));
    }

    auto& frontend = config.frontend;
    frontend.max_iteration = integer("estimator.max_iteration", frontend.max_iteration);
    frontend.filter_size_surf = real("estimator.filter_size_surf", frontend.filter_size_surf);
    frontend.filter_size_map = real("estimator.filter_size_map", frontend.filter_size_map);
    frontend.cube_side_length = real("estimator.cube_side_length", frontend.cube_side_length);
    frontend.det_range = real("estimator.det_range", frontend.det_range);
    frontend.plane_th = real("estimator.plane_th", frontend.plane_th);
    frontend.max_neighbor_distance = real("estimator.max_neighbor_distance", frontend.max_neighbor_distance);
    frontend.max_plane_residual = real("estimator.max_plane_residual", frontend.max_plane_residual);
    frontend.blind = real("estimator.blind", frontend.blind);
    frontend.extrinsic_est_en = flag("estimator.extrinsic_est_en", frontend.extrinsic_est_en);
    frontend.acc_cov = real("estimator.acc_cov", frontend.acc_cov);
    frontend.gyr_cov = real("estimator.gyr_cov", frontend.gyr_cov);
    frontend.b_acc_cov = real("estimator.b_acc_cov", frontend.b_acc_cov);
    frontend.b_gyr_cov = real("estimator.b_gyr_cov", frontend.b_gyr_cov);
    frontend.cov_threshold = real("estimator.cov_threshold", frontend.cov_threshold);
    frontend.point_cov_max = real("estimator.point_cov_max", frontend.point_cov_max);
    frontend.point_cov_min = real("estimator.point_cov_min", frontend.point_cov_min);
    frontend.plane_cov_max = real("estimator.plane_cov_max", frontend.plane_cov_max);
    frontend.plane_cov_min = real("estimator.plane_cov_min", frontend.plane_cov_min);
    frontend.localize_cov_max = real("estimator.localize_cov_max", frontend.localize_cov_max);
    frontend.localize_cov_min = real("estimator.localize_cov_min", frontend.localize_cov_min);
    frontend.localize_thresh_max = real("estimator.localize_thresh_max", frontend.localize_thresh_max);
    frontend.localize_thresh_min = real("estimator.localize_thresh_min", frontend.localize_thresh_min);
    const auto strides = std::get<std::vector<std::int64_t>>(get(
        "estimator.point_filter_num", std::vector<std::int64_t>(count, 1)));
    require(strides.size() == static_cast<std::size_t>(count), "estimator.point_filter_num must have one value per sensor");
    frontend.point_filter_num.clear();
    for (const auto stride : strides) {
        require(stride >= 1 && stride <= std::numeric_limits<int>::max(), "estimator.point_filter_num values must be positive integers");
        frontend.point_filter_num.push_back(static_cast<int>(stride));
    }
    require(frontend.max_iteration > 0 && frontend.filter_size_surf > 0 && frontend.filter_size_map > 0 &&
                frontend.cube_side_length > 0 && frontend.det_range > 0 && frontend.plane_th > 0 &&
                frontend.max_neighbor_distance > 0 && frontend.max_plane_residual > 0 && frontend.blind >= 0,
            "estimator iteration, distance and voxel settings are invalid");
    require(frontend.acc_cov > 0 && frontend.gyr_cov > 0 && frontend.b_acc_cov > 0 && frontend.b_gyr_cov > 0 &&
                frontend.cov_threshold >= 0 && frontend.point_cov_min > 0 && frontend.point_cov_max >= frontend.point_cov_min &&
                frontend.plane_cov_min >= 0 && frontend.plane_cov_max >= frontend.plane_cov_min &&
                frontend.localize_cov_min >= 0 && frontend.localize_cov_max >= frontend.localize_cov_min &&
                frontend.localize_thresh_max > frontend.localize_thresh_min,
            "estimator covariance bounds are invalid");

    config.slam_enabled = flag("slam.enabled", true);
    auto& graph = config.graph;
    graph.submap_voxel = real("slam.submap_voxel", graph.submap_voxel);
    graph.submap_distance = real("slam.submap_distance", graph.submap_distance);
    graph.min_range = real("slam.min_range", graph.min_range);
    graph.max_range = real("slam.max_range", graph.max_range);
    graph.graph_max_iterations = integer("slam.graph_max_iterations", graph.graph_max_iterations);
    graph.closure_overlap_threshold = real("slam.closure_overlap_threshold", graph.closure_overlap_threshold);
    graph.closure_max_rmse = real("slam.closure_max_rmse", graph.closure_max_rmse);
    graph.closure_fine_distance = real("slam.closure_fine_distance", graph.closure_fine_distance);
    graph.closure_min_time_separation = real("slam.closure_min_time_separation", graph.closure_min_time_separation);
    graph.closure_max_candidates = integer("slam.closure_max_candidates", graph.closure_max_candidates);
    graph.loop_search_radius = real("slam.loop_search_radius", graph.loop_search_radius);
    graph.loop_max_correction = real("slam.loop_max_correction", graph.loop_max_correction);
    graph.no_of_sub_maps_to_skip = integer("slam.no_of_sub_maps_to_skip", graph.no_of_sub_maps_to_skip);
    graph.lidar_position_sigma = real("slam.lidar_position_sigma", graph.lidar_position_sigma);
    require(graph.submap_voxel > 0 && graph.submap_distance > 0 && graph.min_range >= 0 &&
                graph.max_range > graph.min_range && graph.graph_max_iterations > 0 && graph.lidar_position_sigma > 0 &&
                graph.closure_overlap_threshold > 0 && graph.closure_overlap_threshold <= 1 &&
                graph.closure_max_rmse > 0 && graph.closure_fine_distance > 0 && graph.closure_min_time_separation >= 0 &&
                graph.closure_max_candidates > 0 && graph.loop_search_radius >= 0 && graph.loop_max_correction > 0 &&
                graph.no_of_sub_maps_to_skip > 0,
            "slam geometry, solver or closure parameters are invalid");

    graph.use_gt = flag("gt.enabled", true);
    config.gt_align_initial = flag("gt.align_initial", true);
    config.gt_evaluate = flag("gt.evaluate", true);
    config.gt_file = text("gt.file", "/Users/grey.haus/Documents/InhouseBuild/CrossModal_Localization/Dataset/UrbanNavDataset/UrbanNav-HK-Medium-Urban-1/UrbanNav_TST_GT_raw.txt");
    graph.gt_gate_covariance = flag("gt.gate_covariance", true);
    const std::string covariance_source = text("gt.covariance_source", "graph");
    require(covariance_source == "graph" || covariance_source == "lio", "gt.covariance_source must be graph or lio");
    graph.gt_gate_graph_covariance = covariance_source == "graph";
    graph.gt_covariance_threshold = real("gt.pose_covariance_threshold", graph.gt_covariance_threshold);
    graph.gt_covariance_ratio = real("gt.covariance_ratio", graph.gt_covariance_ratio);
    graph.gt_min_factor_distance = real("gt.min_factor_distance", graph.gt_min_factor_distance);
    config.gt_huber_delta = real("gt.huber_delta", 3);
    const auto covariance = std::get<std::vector<double>>(get("gt.position_covariance", std::vector<double>{.01, 0, 0, 0, .01, 0, 0, 0, .04}));
    require(covariance.size() == 9, "gt.position_covariance requires 9 row-major values in m^2");
    config.gt_covariance = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(covariance.data());
    require(config.gt_covariance.allFinite() && config.gt_covariance.isApprox(config.gt_covariance.transpose(), 1e-10) &&
                Eigen::LLT<Eigen::Matrix3d>(config.gt_covariance).info() == Eigen::Success,
            "gt.position_covariance must be finite, symmetric and positive definite");
    const auto lever_arm = std::get<std::vector<double>>(get("gt.lidar_lever_arm", std::vector<double>{0., 0., .42}));
    require(lever_arm.size() == 3, "gt.lidar_lever_arm requires 3 values in metres");
    config.gt_lidar_lever_arm = Eigen::Map<const Eigen::Vector3d>(lever_arm.data());
    require(config.gt_lidar_lever_arm.allFinite(), "gt.lidar_lever_arm must be finite");
    require(graph.gt_covariance_threshold >= 0 && graph.gt_covariance_ratio > 0 &&
                graph.gt_min_factor_distance >= 0 && config.gt_huber_delta > 0,
            "gt covariance gate, factor distance or Huber delta is invalid");
    require(!graph.use_gt || config.slam_enabled, "gt.enabled requires slam.enabled");
    require(!(graph.use_gt || config.gt_align_initial || config.gt_evaluate) || !config.gt_file.empty(),
            "gt.file is required for factors, alignment or evaluation");

    auto& output = config.output;
    config.interpolate_pose_corrections = flag("output.interpolate_pose_corrections", true);
    output.directory = text("output.directory", "/Users/grey.haus/Documents/InhouseBuild/CrossModal_Localization/Test/SkyNet/Dataset/UrbanNav/Medium/Map/MA_SLAM");
    output.prefix = text("output.prefix", "MA_SLAM_map");
    output.save_full = flag("output.save_full", true);
    output.save_voxel = flag("output.save_voxel", true);
    output.save_trajectory = flag("output.save_trajectory", true);
    output.overwrite = flag("output.overwrite", false);
    output.save_on_shutdown = flag("output.save_on_shutdown", false);
    output.voxel_size = real("output.voxel_size", .05);
    output.tile_size = real("output.tile_size", 20);
    require(!output.directory.empty(), "output.directory must not be empty");
    require(!output.prefix.empty() && output.prefix.find_first_of("/\\") == std::string::npos &&
                output.prefix != "." && output.prefix != "..", "output.prefix must be a plain filename prefix");
    require(output.voxel_size > 0 && output.tile_size > 0, "output voxel_size and tile_size must be positive");

    config.preview_voxel = real("visualization.preview_voxel", .5);
    config.preview_max_points = positiveCount("visualization.preview_max_points", 200000);
    parameters.map_publish_every_submaps = positiveCount("visualization.map_publish_every_submaps", 5);
    require(config.preview_voxel > 0, "visualization.preview_voxel must be positive");
    parameters.frame_map = text("frames.map", "map");
    parameters.frame_odom = text("frames.odom", "odom");
    parameters.frame_lidar = text("frames.lidar", "lidar_center");
    require(!parameters.frame_map.empty() && !parameters.frame_odom.empty() && !parameters.frame_lidar.empty() &&
                parameters.frame_map != parameters.frame_odom && parameters.frame_map != parameters.frame_lidar &&
                parameters.frame_odom != parameters.frame_lidar, "frames must have three distinct nonempty names");
    return parameters;
}

}  // namespace ma_slam
