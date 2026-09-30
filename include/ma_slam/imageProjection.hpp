#pragma once

#include "ma_slam/types.hpp"
#include <deque>
#include <limits>
#include <optional>
#include <string>

namespace ma_slam {

struct PointCloudField {
    std::string name;
    std::uint32_t offset = 0;
    std::uint8_t datatype = 0;
    std::uint32_t count = 1;
};

// A borrowed sensor_msgs/PointCloud2 payload; decode() never retains this view.
struct PointCloudView {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::uint32_t width = 0, height = 0, point_step = 0, row_step = 0;
    bool bigendian = false;
    std::vector<PointCloudField> fields;
    std::int64_t header_ns = 0;
};

struct SensorConfig {
    std::string topic, frame_id;
    std::string timing_mode = "measured";
    std::string time_field = "time";
    double time_unit_seconds = 1.0;
    double time_offset_seconds = 0.0;
    double scan_duration_seconds = 0.1;
    // Measured: true anchors the measured span at the header; false preserves
    // raw relative offsets to the header. Azimuth: header is scan end/start.
    bool header_is_end = false;
};

struct StampedScan {
    std::int64_t start_ns = 0, end_ns = 0;
    TimedScan scan;
    std::string timing_source;
};

StampedScan decode(const PointCloudView& view, int sensor_id, const SensorConfig& config);

struct StampedImu {
    std::int64_t stamp_ns = 0;
    ImuSample sample;
};

struct SynchronizedBatch {
    std::vector<StampedScan> scans;
    std::vector<StampedImu> imu;
    std::int64_t start_ns = 0, end_ns = 0;
};

struct SynchronizerStats {
    std::vector<std::size_t> dropped_scans;
    std::size_t dropped_imu = 0, batches = 0;
};

// Called by a serialized ROS callback/worker; does not own a thread or mutex.
class Synchronizer {
public:
    explicit Synchronizer(std::size_t sensors, std::int64_t tolerance_ns = 80000000,
                          std::size_t scan_queue_limit = 100, std::size_t imu_queue_limit = 20000,
                          std::size_t future_samples = 15);
    void pushScan(StampedScan scan);
    void pushImu(StampedImu sample);
    std::optional<SynchronizedBatch> pop();
    const SynchronizerStats& stats() const { return stats_; }
    std::size_t queuedScans() const;

private:
    std::vector<std::deque<StampedScan>> scans_;
    std::deque<StampedImu> imu_;
    std::vector<std::int64_t> last_scan_;
    std::int64_t last_imu_ = std::numeric_limits<std::int64_t>::min();
    std::optional<std::int64_t> previous_end_;
    std::int64_t tolerance_ns_;
    std::size_t scan_limit_, imu_limit_, future_samples_;
    SynchronizerStats stats_;
};

}  // namespace ma_slam
