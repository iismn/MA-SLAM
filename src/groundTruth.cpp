#include "ma_slam/groundTruth.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace ma_slam {
namespace {
constexpr double radians = 3.14159265358979323846 / 180.;

Eigen::Vector3d ecef(double latitude, double longitude, double altitude) {
    // -----------------------------------------------------------------
    // - WGS84 ellipsoidal altitude, matching the official UrbanNav GT file.
    // -----------------------------------------------------------------
    constexpr double e2 = 6.6943799901413165e-3;
    const double radius = 6378137. / std::sqrt(1 - e2 * std::pow(std::sin(latitude), 2));
    return {(radius + altitude) * std::cos(latitude) * std::cos(longitude),
            (radius + altitude) * std::cos(latitude) * std::sin(longitude),
            (radius * (1 - e2) + altitude) * std::sin(latitude)};
}

Eigen::Matrix3d orientation(double roll, double pitch, double heading) {
    // -----------------------------------------------------------------
    // - Preserve the established SPAN body-to-ENU convention exactly.
    // - This is not a generic ROS roll/pitch/yaw Euler-angle conversion.
    // -----------------------------------------------------------------
    const double cr = std::cos(roll * radians), sr = std::sin(roll * radians);
    const double cp = std::cos(pitch * radians), sp = std::sin(pitch * radians);
    const double cy = std::cos(-heading * radians), sy = std::sin(-heading * radians);
    Eigen::Matrix3d rotation;
    rotation << cy * cr - sy * sp * sr, -sy * cp, cy * sr + sy * sp * cr,
                sy * cr + cy * sp * sr, cy * cp, sy * sr - cy * sp * cr,
                -cp * sr, sp, cp * cr;
    return rotation;
}

double percentile(const std::vector<double>& sorted, double q) {
    const double index = (sorted.size() - 1) * q;
    const auto low = static_cast<std::size_t>(std::floor(index));
    const auto high = static_cast<std::size_t>(std::ceil(index));
    return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
}
}  // namespace

GroundTruth GroundTruth::load(const std::string& path, const Eigen::Vector3d& lidar_lever_arm) {
    // -----------------------------------------------------------------
    // - Parse the official two-header, 20-column UTC SPAN GT text file.
    // - Convert ECEF to the first GT's ENU frame, then apply the LiDAR lever arm.
    // - Keep all poses strictly time ordered; reject partial/malformed records.
    // -----------------------------------------------------------------
    if (!lidar_lever_arm.allFinite()) throw std::invalid_argument("GT LiDAR lever arm must be finite.");
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open GT trajectory: " + path);
    std::string line;
    if (!std::getline(stream, line) || !std::getline(stream, line)) throw std::runtime_error("GT file is missing two header rows.");
    GroundTruth result;
    Eigen::Matrix3d basis;
    Eigen::Vector3d origin;
    std::size_t row = 2;
    while (std::getline(stream, line)) {
        ++row;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        std::istringstream parser(line);
        std::array<double, 20> values{};
        for (auto& value : values) {
            if (!(parser >> value) || !std::isfinite(value))
                throw std::runtime_error("GT row " + std::to_string(row) + " must contain 20 finite numbers.");
        }
        std::string extra;
        if (parser >> extra) throw std::runtime_error("Extra columns at GT row " + std::to_string(row));
        if (values[0] < 0 || values[0] >= 9.22e9) throw std::runtime_error("GT UTC seconds out of range.");
        const auto stamp = static_cast<std::int64_t>(std::llround(static_cast<long double>(values[0]) * 1e9L));
        if (!result.times_.empty() && stamp <= result.times_.back())
            throw std::runtime_error("GT UTC timestamps must strictly increase.");
        const double latitude = (values[3] + values[4] / 60 + values[5] / 3600) * radians;
        const double longitude = (values[6] + values[7] / 60 + values[8] / 3600) * radians;
        if (std::abs(latitude) > 90 * radians || std::abs(longitude) > 180 * radians)
            throw std::runtime_error("GT latitude/longitude out of range.");
        const auto point = ecef(latitude, longitude, values[9]);
        if (result.times_.empty()) {
            origin = point;
            basis << -std::sin(longitude), std::cos(longitude), 0,
                     -std::sin(latitude) * std::cos(longitude), -std::sin(latitude) * std::sin(longitude), std::cos(latitude),
                     std::cos(latitude) * std::cos(longitude), std::cos(latitude) * std::sin(longitude), std::sin(latitude);
        }
        const auto rotation = orientation(values[16], values[17], values[18]);
        result.times_.push_back(stamp);
        result.positions_.push_back(basis * (point - origin) + rotation * lidar_lever_arm);
        result.orientations_.emplace_back(rotation);
        result.orientations_.back().normalize();
    }
    if (result.times_.size() < 2) throw std::runtime_error("GT trajectory needs at least two records.");
    return result;
}

std::optional<Matrix4> GroundTruth::poseAt(std::int64_t stamp_ns) const {
    // -----------------------------------------------------------------
    // - Interpolate translation and shortest-path quaternion orientation.
    // - Never extrapolate beyond measured GT coverage.
    // -----------------------------------------------------------------
    if (times_.empty() || stamp_ns < times_.front() || stamp_ns > times_.back()) return std::nullopt;
    const auto found = std::lower_bound(times_.begin(), times_.end(), stamp_ns);
    const auto high = static_cast<std::size_t>(std::distance(times_.begin(), found));
    Matrix4 pose = Matrix4::Identity();
    if (*found == stamp_ns) {
        pose.topLeftCorner<3, 3>() = orientations_[high].toRotationMatrix();
        pose.topRightCorner<3, 1>() = positions_[high];
    } else {
        const auto low = high - 1;
        const double alpha = static_cast<double>(stamp_ns - times_[low]) / static_cast<double>(times_[high] - times_[low]);
        pose.topLeftCorner<3, 3>() = orientations_[low].slerp(alpha, orientations_[high]).toRotationMatrix();
        pose.topRightCorner<3, 1>() = (1 - alpha) * positions_[low] + alpha * positions_[high];
    }
    return pose;
}

Matrix4 GroundTruth::alignment(std::int64_t stamp_ns, const Matrix4& local_center_pose) const {
    // -----------------------------------------------------------------
    // - Align all three orientation axes and position at the center LiDAR.
    // -----------------------------------------------------------------
    const auto reference = poseAt(stamp_ns);
    if (!reference) throw std::out_of_range("Initial center LiDAR pose is outside GT coverage.");
    if (!local_center_pose.allFinite()) throw std::invalid_argument("Non-finite initial pose.");
    Matrix4 result = Matrix4::Identity();
    result.topLeftCorner<3, 3>() = reference->topLeftCorner<3, 3>() * local_center_pose.topLeftCorner<3, 3>().transpose();
    result.topRightCorner<3, 1>() = reference->topRightCorner<3, 1>() - result.topLeftCorner<3, 3>() * local_center_pose.topRightCorner<3, 1>();
    return result;
}

AteStats GroundTruth::evaluate(const std::vector<std::int64_t>& stamps_ns, const std::vector<Matrix4>& poses_enu) const {
    // -----------------------------------------------------------------
    // - Compare ENU center-LiDAR origins without a new best-fit alignment.
    // - Report GT-constrained residuals separately from independent accuracy.
    // -----------------------------------------------------------------
    if (stamps_ns.size() != poses_enu.size()) throw std::invalid_argument("ATE pose/time lengths differ.");
    AteStats stats;
    stats.total = poses_enu.size();
    std::vector<double> errors;
    double sum_squared = 0, horizontal_squared = 0, vertical_squared = 0;
    for (std::size_t i = 0; i < stamps_ns.size(); ++i) {
        if (!poses_enu[i].allFinite()) throw std::invalid_argument("ATE received non-finite pose.");
        const auto reference = poseAt(stamps_ns[i]);
        if (!reference) continue;
        const Eigen::Vector3d delta = poses_enu[i].topRightCorner<3, 1>() - reference->topRightCorner<3, 1>();
        errors.push_back(delta.norm());
        sum_squared += delta.squaredNorm();
        horizontal_squared += delta.head<2>().squaredNorm();
        vertical_squared += delta.z() * delta.z();
    }
    stats.matched = errors.size();
    if (errors.empty()) return stats;
    stats.last = errors.back();
    stats.mean = std::accumulate(errors.begin(), errors.end(), 0.) / errors.size();
    stats.rmse = std::sqrt(sum_squared / errors.size());
    stats.horizontal_rmse = std::sqrt(horizontal_squared / errors.size());
    stats.vertical_rmse = std::sqrt(vertical_squared / errors.size());
    std::sort(errors.begin(), errors.end());
    stats.median = percentile(errors, 0.5);
    stats.p95 = percentile(errors, 0.95);
    stats.maximum = errors.back();
    return stats;
}

std::string GroundTruth::report(const AteStats& stats, bool gt_used_in_optimization) {
    // -----------------------------------------------------------------
    // - Format the same ATE quantities shown by the prior MA-SLAM runner.
    // -----------------------------------------------------------------
    std::ostringstream text;
    text << "ATE translation | ENU, center LiDAR origin | no additional alignment\n";
    if (gt_used_in_optimization) text << "GT used in optimization: constraint residual, not independent accuracy.\n";
    text << "Matched: " << stats.matched << '/' << stats.total << " (GT interpolated; no extrapolation)\n";
    if (!stats.matched) return text.str();
    text << std::fixed << std::setprecision(4)
         << "3D RMSE: " << stats.rmse << " m | Mean: " << stats.mean << " m | Median: " << stats.median << " m\n"
         << "P95: " << stats.p95 << " m | Max: " << stats.maximum << " m | Last matched: " << stats.last << " m\n"
         << "Horizontal RMSE: " << stats.horizontal_rmse << " m | Vertical RMSE: " << stats.vertical_rmse << " m";
    return text.str();
}
}  // namespace ma_slam
