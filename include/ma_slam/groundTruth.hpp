#pragma once

#include "ma_slam/types.hpp"
#include <optional>
#include <string>

namespace ma_slam {

struct AteStats {
    std::size_t matched = 0, total = 0;
    double rmse = 0, mean = 0, median = 0, p95 = 0, maximum = 0, last = 0;
    double horizontal_rmse = 0, vertical_rmse = 0;
};

class GroundTruth {
public:
    static GroundTruth load(const std::string& path,
                            const Eigen::Vector3d& lidar_lever_arm = Eigen::Vector3d(0, 0, 0.42));
    std::optional<Matrix4> poseAt(std::int64_t stamp_ns) const;
    Matrix4 alignment(std::int64_t stamp_ns, const Matrix4& local_center_pose) const;
    AteStats evaluate(const std::vector<std::int64_t>& stamps_ns,
                      const std::vector<Matrix4>& poses_enu) const;
    const std::vector<std::int64_t>& timesNs() const { return times_; }
    const std::vector<Eigen::Vector3d>& positions() const { return positions_; }
    // SPAN solution quality Q (last GT column): 1 best .. 6 worst.
    const std::vector<int>& qualities() const { return qualities_; }
    static std::string report(const AteStats& stats, bool gt_used_in_optimization);

private:
    std::vector<std::int64_t> times_;
    std::vector<Eigen::Vector3d> positions_;
    std::vector<int> qualities_;
    std::vector<Eigen::Quaterniond> orientations_;
};

}  // namespace ma_slam
