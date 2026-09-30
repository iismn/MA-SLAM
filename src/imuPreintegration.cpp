// MA-SLAM C++ frontend adapter; numerical estimator is the original MA-LIO core.
#include "ma_slam/imuPreintegration.hpp"

#include <omp.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <pcl/filters/voxel_grid.h>
#include <ikd-Tree/ikd_Tree.h>
#include "IMU_Processing.hpp"

namespace {
// State retained from the original estimator; one instance per process.
bool extrinsic_est_en = true;
int NUM_MAX_ITERATIONS = 4, lid_num = MA_SLAM_LIDAR_COUNT, add_point_size = 0;
int kdtree_delete_counter = 0, feats_down_size = 0;
double filter_size_surf_min = .5, filter_size_map_min = .5, cube_len = 200;
double range_min = 0, range_max = 1, cov_threshold = .5;
float plane_th = .1, DET_RANGE = 100;
double point_cov_max = .00125, point_cov_min = .00075;
double plane_cov_max = 1, plane_cov_min = .8;
double localize_cov_max = 2, localize_cov_min = .3;
double localize_thresh_max = .7, localize_thresh_min = .2;
double max_neighbor_distance = 1.0, max_plane_residual = .2;

#include "lidarOdometry.inc"

static std::shared_ptr<ImuProcess> p_imu;
static bool configured = false;
static double blind = 0;
static double previous_end = -1;
static std::vector<int> input_stride(MA_SLAM_LIDAR_COUNT, 1);
static int accepted_batches = 0;

}  // namespace

namespace ma_slam {

ImuPreintegration::ImuPreintegration(const FrontendConfig& settings,
                                     const std::vector<Matrix4>& transforms)
{
    // -------------------------------------------------------------
    // - Configure the original MA-LIO IESKF and physical sensor extrinsics.
    // - Preserve one estimator instance per process, as in the original node.
    // -------------------------------------------------------------
    if (configured) throw std::runtime_error("Only one MA-LIO estimator is supported per process.");
    if (transforms.size() != static_cast<std::size_t>(lid_num))
        throw std::invalid_argument("Transforms must match the compiled LiDAR manifold.");
    for (const double value : {settings.filter_size_surf, settings.filter_size_map,
            settings.cube_side_length, settings.det_range, settings.plane_th,
            settings.cov_threshold, settings.point_cov_max, settings.point_cov_min,
            settings.plane_cov_max, settings.plane_cov_min, settings.localize_cov_max,
            settings.localize_cov_min, settings.localize_thresh_max, settings.localize_thresh_min,
            settings.blind, settings.max_neighbor_distance, settings.max_plane_residual}) {
        if (!std::isfinite(value)) throw std::invalid_argument("MA-LIO parameters must be finite.");
    }
    for (const auto& transform : transforms) {
        if (!transform.allFinite() || !transform.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1)) ||
            std::abs(transform.topLeftCorner<3, 3>().determinant() - 1.0) > 1e-5 ||
            !(transform.topLeftCorner<3, 3>().transpose() * transform.topLeftCorner<3, 3>())
                .isApprox(M3D::Identity(), 1e-5))
            throw std::invalid_argument("Invalid LiDAR-to-IMU transform.");
    }
    NUM_MAX_ITERATIONS = settings.max_iteration;
    filter_size_surf_min = settings.filter_size_surf;
    filter_size_map_min = settings.filter_size_map;
    cube_len = settings.cube_side_length;
    DET_RANGE = settings.det_range;
    plane_th = settings.plane_th;
    extrinsic_est_en = settings.extrinsic_est_en;
    cov_threshold = settings.cov_threshold;
    point_cov_max = settings.point_cov_max; point_cov_min = settings.point_cov_min;
    plane_cov_max = settings.plane_cov_max; plane_cov_min = settings.plane_cov_min;
    localize_cov_max = settings.localize_cov_max; localize_cov_min = settings.localize_cov_min;
    localize_thresh_max = settings.localize_thresh_max; localize_thresh_min = settings.localize_thresh_min;
    blind = settings.blind; input_stride = settings.point_filter_num;
    max_neighbor_distance = settings.max_neighbor_distance;
    max_plane_residual = settings.max_plane_residual;
    if (input_stride.size() != lid_num || std::any_of(input_stride.begin(), input_stride.end(), [](int s){ return s < 1; }) ||
        filter_size_surf_min <= 0 || filter_size_map_min <= 0 || NUM_MAX_ITERATIONS < 1 ||
        cube_len <= 0 || DET_RANGE <= 0 || blind < 0 || point_cov_min <= 0 ||
        point_cov_max < point_cov_min || localize_thresh_max <= localize_thresh_min ||
        max_neighbor_distance <= 0 || max_plane_residual <= 0)
        throw std::invalid_argument("Invalid MA-LIO parameters.");
    p_imu = std::make_shared<ImuProcess>();
    const auto noise = [](double value) {
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("IMU noise variances must be finite and positive.");
        return V3D::Constant(value).eval();
    };
    p_imu->set_acc_cov(noise(settings.acc_cov));
    p_imu->set_gyr_cov(noise(settings.gyr_cov));
    p_imu->set_acc_bias_cov(noise(settings.b_acc_cov));
    p_imu->set_gyr_bias_cov(noise(settings.b_gyr_cov));
    for (int i = 0; i < lid_num; ++i) {
        const auto& t = transforms[i];
        ext_t.push_back(t.topRightCorner<3, 1>());
        ext_q.emplace_back(t.topLeftCorner<3, 3>());
    }
    for (int i = lid_num - 1; i >= 0; --i) last_indices.push_back(i);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS);
    kf.extrinsicInit(ext_t, ext_q);
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    configured = true;
}

void ImuPreintegration::testCovariancePermutation() const
{
    if (!configured || previous_end >= 0) throw std::runtime_error("Run this check just after configure.");
    auto saved = kf.get_P();
    Eigen::MatrixXd a(state_ikfom::DOF, state_ikfom::DOF);
    for (int row = 0; row < a.rows(); ++row)
        for (int col = 0; col < a.cols(); ++col) a(row, col) = std::sin((row + 1) * (col + 3) * .13);
    auto synthetic = saved;
    synthetic = .01 * a * a.transpose();
    synthetic.diagonal().array() += .1;
    kf.change_P(synthetic);
    auto current = last_indices;
    std::vector<int> order;
    for (int id = 0; id < lid_num; ++id) order.push_back(id);
    do {
        kf.change_ext(order, current);
        current = order;
        std::vector<int> mapping(state_ikfom::DOF);
        for (int i = 0; i < state_ikfom::DOF; ++i) mapping[i] = i;
        for (int slot = 0; slot < lid_num; ++slot) {
            const int id = order[lid_num - 1 - slot];
            const auto& state = kf.get_x();
            const auto& q = *static_cast<MTK::SO3<double>*>(state.SO3_state_ptr[1 + slot]);
            const auto& t = *static_cast<MTK::vect<3, double>*>(state.vect_state_ptr[1 + slot]);
            if (!q.toRotationMatrix().isApprox(ext_q[id].toRotationMatrix(), 1e-12) || !t.isApprox(ext_t[id], 1e-12))
                throw std::runtime_error("Physical-sensor extrinsic permutation failed.");
            for (int k = 0; k < 3; ++k) {
                mapping[6 + 3 * slot + k] = 6 + 3 * id + k;
                mapping[6 + 3 * (lid_num + slot) + k] = 6 + 3 * (lid_num + id) + k;
            }
        }
        for (int row = 0; row < state_ikfom::DOF; ++row)
            for (int col = 0; col < state_ikfom::DOF; ++col)
                if (kf.get_P()(row, col) != synthetic(mapping[row], mapping[col]))
                    throw std::runtime_error("Full covariance/cross-term permutation failed.");
    } while (std::next_permutation(order.begin(), order.end()));
    kf.change_ext(last_indices, current);
    kf.change_P(saved);
}

FrontendResult ImuPreintegration::process(const std::vector<TimedScan>& scans,
                                         const std::vector<ImuSample>& samples)
{
    // -------------------------------------------------------------
    // - Preserve independent LiDAR scan epochs and per-point timestamps.
    // - Propagate IMU, B-spline deskew, planar IESKF, then update the local map.
    // - Export center-LiDAR points with measured intensity and covariance.
    // -------------------------------------------------------------
    if (!configured || scans.size() != static_cast<std::size_t>(lid_num))
        throw std::invalid_argument("Pass one distinct scan per configured LiDAR.");
    if (samples.size() < 17)
        throw std::invalid_argument("IMU samples must include 15 future samples.");
    struct Scan { int id; double begin, end; PointCloudXYZI::Ptr cloud; };
    std::vector<Scan> input;
    for (const auto& scan : scans) {
        const int id = scan.sensor_id;
        const double begin = scan.begin_seconds;
        if (id < 0 || id >= lid_num || !std::isfinite(begin) || scan.points.size() < 2)
            throw std::invalid_argument("Invalid scan sensor, timestamp or point count.");
        PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
        cloud->reserve(scan.points.size());
        double max_offset = 0;
        for (std::size_t j = 0; j < scan.points.size(); ++j) {
            const auto& p = scan.points[j];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(p.intensity) || !std::isfinite(p.offset_seconds))
                throw std::invalid_argument("Non-finite scan input.");
            if (p.offset_seconds < 0 || p.offset_seconds > .5)
                throw std::invalid_argument("Point timing is not a relative scan time.");
            max_offset = std::max(max_offset,
                static_cast<double>(static_cast<float>(p.offset_seconds * 1000)) / 1000);
            if (j % input_stride[id] || p.x*p.x + p.y*p.y + p.z*p.z <= blind*blind)
                continue;
            PointType q;
            q.x = p.x; q.y = p.y; q.z = p.z;
            q.intensity = p.intensity; q.curvature = p.offset_seconds * 1000;
            q.normal_x = 0; q.normal_y = id;
            q.normal_z = p.intensity; // Internal intensity becomes an uncertainty index.
            cloud->push_back(q);
        }
        if (cloud->size() < 6) throw std::invalid_argument("Too few valid LiDAR points.");
        input.push_back({id, begin, begin + max_offset, cloud});
    }
    std::vector<bool> seen(lid_num, false);
    for (const auto& scan : input) {
        if (seen[scan.id]) throw std::invalid_argument("Duplicated sensor ID.");
        seen[scan.id] = true;
    }
    std::sort(input.begin(), input.end(), [](const Scan& a, const Scan& b) { return a.end < b.end; });
    if (input.back().end <= previous_end) throw std::invalid_argument("Non-monotonic batch end time.");
    Measures = MeasureGroup();
    std::vector<int> indices;
    for (const auto& scan : input) {
        indices.push_back(scan.id);
        Measures.lidar_multi.push_back(scan.cloud);
        Measures.lidar_beg_time.push_back(scan.begin);
        Measures.lidar_end_time.push_back(scan.end);
    }
    double last = -std::numeric_limits<double>::infinity();
    for (const auto& sample : samples) {
        if (!std::isfinite(sample.time_seconds) || !sample.acceleration.allFinite() ||
            !sample.angular_velocity.allFinite())
            throw std::invalid_argument("Non-finite IMU input.");
        if (sample.time_seconds <= last)
            throw std::invalid_argument("IMU times must strictly increase.");
        last = sample.time_seconds;
        auto msg = std::make_shared<ma_lio_core::Imu>();
        msg->time_seconds = last;
        msg->linear_acceleration = {sample.acceleration.x(), sample.acceleration.y(), sample.acceleration.z()};
        msg->angular_velocity = {sample.angular_velocity.x(), sample.angular_velocity.y(), sample.angular_velocity.z()};
        if (last <= input.back().end) Measures.imu.push_back(msg);
        else if (Measures.imu_cont.size() < 16) Measures.imu_cont.push_back(msg);
    }
    if (Measures.imu.empty() || Measures.imu_cont.size() < 15)
        throw std::invalid_argument("IMU must cover the scan and 15 samples beyond its end.");
    Measures.imu_cont.push_front(Measures.imu.back());
    if (Measures.imu_cont.size() > 16) Measures.imu_cont.pop_back();
    Eigen::Matrix4d prediction = Eigen::Matrix4d::Identity();
    size_t unsupported = 0;
    {
        if (indices != last_indices) kf.change_ext(indices, last_indices);
        last_indices = indices;
        state_point = kf.get_x();
        extrinsic_update();
        if (flg_first_scan) {
            first_lidar_time = input.front().begin;
            p_imu->first_lidar_time = first_lidar_time;
            flg_first_scan = false;
        }
        feats_undistort->clear(); feats_down_body->clear();
        feats_undistort_vec.clear(); feats_down_vec.clear();
        for (int i = 0; i < lid_num; ++i) {
            feats_undistort_vec.emplace_back(new PointCloudXYZI());
            feats_down_vec.emplace_back(new PointCloudXYZI());
        }
        p_imu->Process(Measures, kf, feats_undistort_vec);
        state_point = kf.get_x(); extrinsic_update();
        prediction.topLeftCorner<3, 3>() = kf.get_x().rot.toRotationMatrix();
        prediction.topRightCorner<3, 1>() = kf.get_x().pos;
        for (int num = 0; num < lid_num; ++num) {
            auto& cloud = *feats_undistort_vec[num];
            const auto original_count = cloud.size();
            cloud.points.erase(std::remove_if(cloud.points.begin(), cloud.points.end(),
                [](const PointType& p) { return p.normal_x < 0; }), cloud.points.end());
            unsupported += original_count - cloud.size();
            cloud.width = cloud.size(); cloud.height = 1;
            downSizeFilterSurf.setInputCloud(feats_undistort_vec[num]);
            downSizeFilterSurf.filter(*feats_down_vec[num]);
            for (auto& p : feats_down_vec[num]->points) { p.normal_x = p.intensity; p.intensity = num; }
            for (auto& p : cloud.points) p.intensity = num;
            *feats_undistort += cloud;
            *feats_down_body += *feats_down_vec[num];
        }
        feats_down_size = feats_down_body->size();
        if (feats_down_size > 100000) throw std::runtime_error("Upstream registration arrays support 100000 points; increase filter_size_surf.");
        // Do not seed the map with a partial startup spline trajectory.
        if (feats_down_size > 5 && (ikdtree.Root_Node != nullptr || unsupported == 0)) {
            state_point = kf.get_x(); extrinsic_update();
            pos_lid = state_point.pos + state_point.rot * extrinsic_trans[0];
            flg_EKF_inited = input.front().begin - first_lidar_time >= INIT_TIME;
            lasermap_fov_segment();
            feats_down_world->resize(feats_down_size);
            if (ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(filter_size_map_min);
                for (int i = 0; i < feats_down_size; ++i) {
                    pointBodyToWorld(&feats_down_body->points[i], &feats_down_world->points[i]);
                    feats_down_world->points[i].normal_y = .001;
                }
                ikdtree.Build(feats_down_world->points);
            } else {
                normvec->resize(feats_down_size);
                pointSearchInd_surf.resize(feats_down_size); Nearest_Points.resize(feats_down_size);
                pose_unc.assign(lid_num, {});
                Pose point;
                for (int num = 0; num < lid_num; ++num) {
                    for (int i = 0; i < static_cast<int>(kf.lidar_uncertainty[num].size()) - 1; ++i) {
                        if (num == 0) pose_unc[num].push_back(kf.lidar_uncertainty[num][i]);
                        else {
                            compoundPoseWithCov(extrinsic[num], extrinsic[num].cov_, kf.lidar_uncertainty[num][i], kf.lidar_uncertainty[num][i].cov_, point, point.cov_, 2);
                            compoundPoseWithCov(kf.temporal_comp[num-1], kf.temporal_comp[num-1].cov_, point, point.cov_, point, point.cov_, 2);
                            compoundInvPoseWithCov(extrinsic[0], extrinsic[0].cov_, point, point.cov_, point, point.cov_, 2);
                            pose_unc[num].push_back(point);
                        }
                    }
                    if (pose_unc[num].size() < 2) throw std::runtime_error("Insufficient per-point uncertainty history.");
                }
                for (auto& p : feats_down_body->points)
                    p.normal_x = std::clamp(p.normal_x, 0.f, static_cast<float>(pose_unc[static_cast<int>(p.intensity)].size()-2));
                double solve_time = 0;
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_time);
                state_point = kf.get_x(); extrinsic_update(); // Read the updated state before its extrinsics.
                map_incremental();
            }
            if (!state_point.pos.allFinite() || !state_point.rot.toRotationMatrix().allFinite())
                throw std::runtime_error("MA-LIO produced a non-finite state.");
            ++accepted_batches;
        }
        previous_end = input.back().end;
        if (ikdtree.Root_Node == nullptr)
            for (auto& cloud : feats_undistort_vec) cloud->clear();
    }
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    pose.topLeftCorner<3, 3>() = state_point.rot.toRotationMatrix();
    pose.topRightCorner<3, 1>() = state_point.pos;
    Eigen::Matrix4d center_to_imu = Eigen::Matrix4d::Identity();
    int center_slot = 0;
    while (input[lid_num - 1 - center_slot].id != 0) ++center_slot;
    center_to_imu.topLeftCorner<3, 3>() = extrinsic_quat[center_slot].toRotationMatrix();
    center_to_imu.topRightCorner<3, 1>() = extrinsic_trans[center_slot];
    FrontendResult result;
    // All sensor clouds refer to the center HDL32 frame at this batch end.
    for (int num = 0; num < lid_num; ++num) {
        const auto& cloud = feats_undistort_vec[num];
        SensorCloud output;
        output.sensor_id = input[lid_num-1-num].id;
        output.points.reserve(cloud->size());
        for (const auto& p : cloud->points) {
            V3D xyz = extrinsic_quat[num] * V3D(p.x, p.y, p.z) + extrinsic_trans[num];
            if (num) xyz = kf.temporal_comp[num-1].q_ * xyz + kf.temporal_comp[num-1].t_;
            xyz = center_to_imu.topLeftCorner<3, 3>().transpose() * (xyz - center_to_imu.topRightCorner<3, 1>());
            pcl::PointXYZI output_point;
            output_point.x = xyz.x(); output_point.y = xyz.y(); output_point.z = xyz.z();
            output_point.intensity = p.normal_z;
            output.points.push_back(output_point);
        }
        result.clouds.push_back(std::move(output));
    }
    result.imu_pose = pose;
    result.center_pose = pose * center_to_imu;
    result.predicted_imu_pose = prediction;
    result.center_to_imu = center_to_imu;
    // LiDAR-origin covariance, including attitude, lever-arm and cross terms.
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, state_ikfom::DOF);
    const M3D rotation = pose.topLeftCorner<3, 3>();
    const V3D lever_arm = center_to_imu.topRightCorner<3, 1>();
    jacobian.block<3, 3>(0, 0).setIdentity();
    jacobian.block<3, 3>(0, 3) = -rotation * skewSymmetric(lever_arm);
    jacobian.block<3, 3>(0, 6 + 3 * (lid_num + center_slot)) = rotation;
    Eigen::Matrix3d position_covariance = jacobian * kf.get_P() * jacobian.transpose();
    position_covariance = (0.5 * (position_covariance + position_covariance.transpose())).eval();
    result.position_covariance = position_covariance;
    result.gyro_bias = V3D(kf.get_x().bg);
    result.velocity = V3D(kf.get_x().vel);
    result.end_seconds = input.back().end; result.map_points = ikdtree.size();
    result.effective_points = effct_feat_num; result.accepted_batches = accepted_batches;
    result.unsupported_points = unsupported;
    return result;
}

int ImuPreintegration::lidarCount() { return MA_SLAM_LIDAR_COUNT; }

}  // namespace ma_slam
