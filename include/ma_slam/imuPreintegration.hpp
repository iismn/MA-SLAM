#pragma once

#include "ma_slam/types.hpp"

#ifndef MA_SLAM_LIDAR_COUNT
#define MA_SLAM_LIDAR_COUNT 3
#endif

namespace ma_slam {

struct FrontendConfig {
    int max_iteration = 4;
    double filter_size_surf = .5;
    double filter_size_map = .5;
    double cube_side_length = 200;
    double det_range = 100;
    double plane_th = .1;
    double max_neighbor_distance = 1;
    double max_plane_residual = .2;
    std::vector<int> point_filter_num = std::vector<int>(MA_SLAM_LIDAR_COUNT, 1);
    double blind = 0;
    bool extrinsic_est_en = true;
    double acc_cov = .011197412605492375;
    double gyr_cov = .010270904839480961;
    double b_acc_cov = .00011751767903346351;
    double b_gyr_cov = .000091355383994881894;
    double cov_threshold = .5;
    double point_cov_max = .00125;
    double point_cov_min = .00075;
    double plane_cov_max = 1;
    double plane_cov_min = .8;
    double localize_cov_max = 2;
    double localize_cov_min = .3;
    double localize_thresh_max = .7;
    double localize_thresh_min = .2;
};

class ImuPreintegration {
public:
    // Original MA-LIO state is process-global: create one frontend per process.
    ImuPreintegration(const FrontendConfig& config,
                      const std::vector<Matrix4>& lidar_to_imu);
    ImuPreintegration(const ImuPreintegration&) = delete;
    ImuPreintegration& operator=(const ImuPreintegration&) = delete;

    // Supply one scan per physical sensor, ordered IMU, and 15 future IMU samples.
    FrontendResult process(const std::vector<TimedScan>& scans,
                           const std::vector<ImuSample>& imu);
    static int lidarCount();
    void testCovariancePermutation() const;
};

}  // namespace ma_slam
