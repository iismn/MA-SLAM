#include "ma_slam/imageProjection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ma_slam {
namespace {
constexpr double pi = 3.14159265358979323846;

std::size_t fieldSize(std::uint8_t type) {
    // -----------------------------------------------------------------
    // - Use sensor_msgs/PointField numeric datatype identifiers.
    // -----------------------------------------------------------------
    constexpr std::array<std::size_t, 9> sizes{0, 1, 1, 2, 2, 4, 4, 4, 8};
    if (type == 0 || type >= sizes.size()) throw std::invalid_argument("Unsupported PointField datatype.");
    return sizes[type];
}

const PointCloudField& field(const PointCloudView& view, const std::string& name) {
    // -----------------------------------------------------------------
    // - Validate the selected scalar field against the actual point stride.
    // -----------------------------------------------------------------
    auto found = std::find_if(view.fields.begin(), view.fields.end(), [&](const auto& f) { return f.name == name; });
    if (found == view.fields.end()) throw std::invalid_argument("PointCloud2 missing field: " + name);
    if (found->count != 1 || found->offset > view.point_step ||
        fieldSize(found->datatype) > view.point_step - found->offset)
        throw std::invalid_argument("PointCloud2 invalid scalar field: " + name);
    return *found;
}

template <class T> double scalar(const std::uint8_t* bytes, bool bigendian) {
    // -----------------------------------------------------------------
    // - Avoid alignment/aliasing assumptions and decode both wire endians.
    // -----------------------------------------------------------------
    std::array<std::uint8_t, sizeof(T)> data{};
    std::memcpy(data.data(), bytes, sizeof(T));
    const std::uint16_t one = 1;
    const bool host_bigendian = *reinterpret_cast<const std::uint8_t*>(&one) == 0;
    if (bigendian != host_bigendian) std::reverse(data.begin(), data.end());
    T value;
    std::memcpy(&value, data.data(), sizeof(T));
    return static_cast<double>(value);
}

double read(const std::uint8_t* point, const PointCloudField& f, bool bigendian) {
    // -----------------------------------------------------------------
    // - Read the wire offset, not the order of entries in fields[].
    // -----------------------------------------------------------------
    const auto* p = point + f.offset;
    switch (f.datatype) {
        case 1: return scalar<std::int8_t>(p, bigendian);
        case 2: return scalar<std::uint8_t>(p, bigendian);
        case 3: return scalar<std::int16_t>(p, bigendian);
        case 4: return scalar<std::uint16_t>(p, bigendian);
        case 5: return scalar<std::int32_t>(p, bigendian);
        case 6: return scalar<std::uint32_t>(p, bigendian);
        case 7: return scalar<float>(p, bigendian);
        case 8: return scalar<double>(p, bigendian);
        default: throw std::invalid_argument("Unsupported PointField datatype.");
    }
}

std::int64_t nanoseconds(double seconds) {
    // -----------------------------------------------------------------
    // - Check range before floating-point-to-integer conversion.
    // -----------------------------------------------------------------
    const long double ns = static_cast<long double>(seconds) * 1000000000.L;
    if (!std::isfinite(seconds) || ns <= static_cast<long double>(std::numeric_limits<std::int64_t>::min()) ||
        ns >= static_cast<long double>(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Point timestamp or time offset is out of range.");
    return std::llround(ns);
}

std::int64_t addTime(std::int64_t a, std::int64_t b) {
    if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||
        (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))
        throw std::invalid_argument("Timestamp overflow.");
    return a + b;
}
}  // namespace

StampedScan decode(const PointCloudView& view, int sensor_id, const SensorConfig& config) {
    // -----------------------------------------------------------------
    // - Preserve per-point time/intensity and each sensor's independent scan.
    // - Honor organized row padding; UrbanNav single-row row_step=0 is valid.
    // - Azimuth timing is an explicit LS-C16 approximation, never a fallback.
    // -----------------------------------------------------------------
    if (sensor_id < 0 || !view.data || view.width == 0 || view.height == 0 || view.point_step == 0)
        throw std::invalid_argument("Empty/invalid PointCloud2 input.");
    const std::size_t row_bytes = static_cast<std::size_t>(view.width) * view.point_step;
    const std::size_t stride = view.height == 1 ? row_bytes : view.row_step;
    if (stride < row_bytes || row_bytes > view.size ||
        (view.height > 1 && static_cast<std::size_t>(view.height - 1) > (view.size - row_bytes) / stride))
        throw std::invalid_argument("PointCloud2 row stride or payload is invalid.");
    if (config.timing_mode != "measured" && config.timing_mode != "azimuth_rings")
        throw std::invalid_argument("Unknown sensor timing_mode: " + config.timing_mode);
    const auto& fx = field(view, "x");
    const auto& fy = field(view, "y");
    const auto& fz = field(view, "z");
    const auto& fi = field(view, "intensity");
    const auto* ft = config.timing_mode == "measured" ? &field(view, config.time_field) : nullptr;
    if (ft && (!std::isfinite(config.time_unit_seconds) || config.time_unit_seconds <= 0))
        throw std::invalid_argument("time_unit_seconds must be finite and positive.");
    StampedScan result;
    result.scan.sensor_id = sensor_id;
    result.scan.points.reserve(static_cast<std::size_t>(view.width) * view.height);
    std::vector<double> times;
    times.reserve(result.scan.points.capacity());
    for (std::size_t row = 0; row < view.height; ++row) {
        for (std::size_t col = 0; col < view.width; ++col) {
            const auto* p = view.data + row * stride + col * view.point_step;
            TimedPoint point;
            point.x = static_cast<float>(read(p, fx, view.bigendian));
            point.y = static_cast<float>(read(p, fy, view.bigendian));
            point.z = static_cast<float>(read(p, fz, view.bigendian));
            point.intensity = static_cast<float>(read(p, fi, view.bigendian));
            const double t = ft ? read(p, *ft, view.bigendian) * config.time_unit_seconds : 0;
            if (!std::isfinite(t)) throw std::invalid_argument("Non-finite point timestamp.");
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) || !std::isfinite(point.intensity))
                continue;
            result.scan.points.push_back(point);
            times.push_back(t);
        }
    }
    if (times.size() < 2) throw std::invalid_argument("LiDAR scan has fewer than two finite points.");
    const auto header = addTime(view.header_ns, nanoseconds(config.time_offset_seconds));
    if (ft) {
        const auto extrema = std::minmax_element(times.begin(), times.end());
        const double minimum = *extrema.first, maximum = *extrema.second;
        if (!(maximum > minimum && maximum - minimum <= 0.5))
            throw std::invalid_argument("Measured point-time span must be >0 and <=500ms; check timestamp units.");
        const bool absolute = minimum >= 100000000.;
        const auto min_ns = nanoseconds(minimum), max_ns = nanoseconds(maximum);
        if (absolute) {
            result.start_ns = addTime(min_ns, nanoseconds(config.time_offset_seconds));
            result.end_ns = addTime(max_ns, nanoseconds(config.time_offset_seconds));
        } else if (config.header_is_end) {
            result.end_ns = header;
            result.start_ns = addTime(header, -(max_ns - min_ns));
        } else {
            result.start_ns = addTime(header, min_ns);
            result.end_ns = addTime(header, max_ns);
        }
        for (std::size_t i = 0; i < times.size(); ++i)
            result.scan.points[i].offset_seconds = (nanoseconds(times[i]) - min_ns) * 1e-9;
        result.timing_source = "measured_point_time";
    } else {
        if (!std::isfinite(config.scan_duration_seconds) || config.scan_duration_seconds <= 0 || config.scan_duration_seconds > 0.5)
            throw std::invalid_argument("LS-C16 scan duration must be >0 and <=500ms.");
        const auto duration_ns = nanoseconds(config.scan_duration_seconds);
        result.start_ns = config.header_is_end ? addTime(header, -duration_ns) : header;
        result.end_ns = addTime(result.start_ns, duration_ns);
        double previous = std::atan2(static_cast<double>(result.scan.points[0].y), static_cast<double>(result.scan.points[0].x));
        double unwrapped = previous;
        const double first = previous;
        double turns = 0;
        for (auto& point : result.scan.points) {
            const double angle = std::atan2(static_cast<double>(point.y), static_cast<double>(point.x));
            double step = angle - previous;
            if (step > pi) step -= 2 * pi;
            if (step < -pi) step += 2 * pi;
            unwrapped += step;
            previous = angle;
            turns = -(unwrapped - first) / (2 * pi);
            double phase = turns - std::floor(turns);
            if (phase > 1 - 1e-9) phase = 0;
            point.offset_seconds = std::llround(phase * duration_ns) * 1e-9;
        }
        if (!(turns > 12.5 && turns < 16.3))
            throw std::invalid_argument("LS-C16 ordered scan is not 13-16 ring sweeps; azimuth timing is unsafe.");
        result.timing_source = "estimated_azimuth_rings";
    }
    return result;
}

Synchronizer::Synchronizer(std::size_t sensors, std::int64_t tolerance_ns,
                           std::size_t scan_queue_limit, std::size_t imu_queue_limit,
                           std::size_t future_samples)
    : scans_(sensors), last_scan_(sensors, std::numeric_limits<std::int64_t>::min()),
      tolerance_ns_(tolerance_ns), scan_limit_(scan_queue_limit), imu_limit_(imu_queue_limit),
      future_samples_(future_samples) {
    if (sensors == 0 || tolerance_ns < 0 || scan_queue_limit == 0 ||
        future_samples < 15 || imu_queue_limit <= future_samples + 1)
        throw std::invalid_argument("Invalid synchronization queue/tolerance/lookahead settings.");
    stats_.dropped_scans.resize(sensors);
}

void Synchronizer::pushScan(StampedScan scan) {
    // -----------------------------------------------------------------
    // - Bound memory and reject reversed clocks instead of reordering scans.
    // -----------------------------------------------------------------
    const int id = scan.scan.sensor_id;
    if (id < 0 || static_cast<std::size_t>(id) >= scans_.size() || scan.start_ns >= scan.end_ns)
        throw std::invalid_argument("Invalid sensor ID or LiDAR scan interval.");
    if (scan.end_ns <= last_scan_[id]) throw std::invalid_argument("LiDAR scan end timestamps must strictly increase.");
    last_scan_[id] = scan.end_ns;
    auto& queue = scans_[id];
    queue.push_back(std::move(scan));
    if (queue.size() > scan_limit_) {
        queue.pop_front();
        ++stats_.dropped_scans[id];
    }
}

void Synchronizer::pushImu(StampedImu sample) {
    // -----------------------------------------------------------------
    // - Keep physical IMU axes; the frontend applies calibrated extrinsics.
    // -----------------------------------------------------------------
    if (!sample.sample.acceleration.allFinite() || !sample.sample.angular_velocity.allFinite())
        throw std::invalid_argument("Non-finite IMU measurement.");
    if (sample.stamp_ns <= last_imu_) throw std::invalid_argument("IMU timestamps must strictly increase.");
    last_imu_ = sample.stamp_ns;
    imu_.push_back(std::move(sample));
    if (imu_.size() > imu_limit_) {
        imu_.pop_front();
        ++stats_.dropped_imu;
    }
}

std::optional<SynchronizedBatch> Synchronizer::pop() {
    // -----------------------------------------------------------------
    // - Pair independent sensor scans by end time without merging clouds.
    // - Wait for scan-end coverage and 15 future B-spline IMU samples.
    // - Preserve partial first scans for original IMU initialization only.
    // - Future samples remain queued for their later integration interval.
    // -----------------------------------------------------------------
    while (std::all_of(scans_.begin(), scans_.end(), [](const auto& q) { return !q.empty(); })) {
        auto oldest = std::min_element(scans_.begin(), scans_.end(), [](const auto& a, const auto& b) {
            return a.front().end_ns < b.front().end_ns;
        });
        const auto newest = std::max_element(scans_.begin(), scans_.end(), [](const auto& a, const auto& b) {
            return a.front().end_ns < b.front().end_ns;
        });
        if (newest->front().end_ns - oldest->front().end_ns > tolerance_ns_) {
            ++stats_.dropped_scans[oldest->front().scan.sensor_id];
            oldest->pop_front();
            continue;
        }
        if (imu_.empty()) return std::nullopt;
        std::int64_t begin = scans_[0].front().start_ns;
        for (const auto& q : scans_) begin = std::min(begin, q.front().start_ns);
        const auto end = newest->front().end_ns;
        if (previous_end_ && end <= *previous_end_) throw std::runtime_error("Synchronized batch clock regressed.");
        if (previous_end_ && imu_.front().stamp_ns > *previous_end_)
            throw std::runtime_error("IMU queue overrun lost integration history; increase imu_queue_limit or reduce playback rate.");
        if (!previous_end_ && imu_.front().stamp_ns > end) {
            // A bag can start mid-scan. Preserve partial startup scans for the
            // original IMU initialization; native spline support omits points
            // it cannot deskew. Drop only batches ending before the first IMU.
            for (auto& q : scans_) {
                if (q.front().end_ns < imu_.front().stamp_ns) {
                    ++stats_.dropped_scans[q.front().scan.sensor_id];
                    q.pop_front();
                }
            }
            continue;
        }
        auto future = std::upper_bound(imu_.begin(), imu_.end(), end,
                                      [](auto t, const auto& s) { return t < s.stamp_ns; });
        auto first = previous_end_ ? std::upper_bound(imu_.begin(), imu_.end(), *previous_end_,
                                       [](auto t, const auto& s) { return t < s.stamp_ns; }) : imu_.begin();
        if (first == future) throw std::runtime_error("No new IMU samples cover synchronized scan interval.");
        const auto history = static_cast<std::size_t>(std::distance(first, future));
        // The adapter requires 17 rows in total, as well as 15 future samples.
        const auto lookahead = std::max(future_samples_, history < 17 ? 17 - history : 0);
        if (static_cast<std::size_t>(std::distance(future, imu_.end())) < lookahead) return std::nullopt;
        SynchronizedBatch batch;
        batch.start_ns = begin;
        batch.end_ns = end;
        batch.imu.assign(first, future + static_cast<std::ptrdiff_t>(lookahead));
        for (auto& queue : scans_) {
            batch.scans.push_back(std::move(queue.front()));
            queue.pop_front();
        }
        // Preserve one consumed predecessor for the next coverage check.
        const auto consumed = static_cast<std::size_t>(std::distance(imu_.begin(), future));
        for (std::size_t i = 1; i < consumed; ++i) imu_.pop_front();
        previous_end_ = end;
        ++stats_.batches;
        return batch;
    }
    return std::nullopt;
}

std::size_t Synchronizer::queuedScans() const {
    std::size_t total = 0;
    for (const auto& queue : scans_) total += queue.size();
    return total;
}
}  // namespace ma_slam
