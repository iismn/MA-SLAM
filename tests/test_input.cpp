#include "ma_slam/imageProjection.hpp"
#include "ma_slam/groundTruth.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace ma_slam;

void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

template <class Callable> void rejects(Callable operation) {
    bool caught = false;
    try { operation(); } catch (const std::exception&) { caught = true; }
    require(caught, "Invalid input was accepted.");
}

template <class T> void put(std::vector<std::uint8_t>& data, std::size_t offset, T value, bool bigendian = false) {
    std::uint8_t bytes[sizeof(T)];
    std::memcpy(bytes, &value, sizeof(T));
    const std::uint16_t one = 1;
    const bool host_bigendian = *reinterpret_cast<const std::uint8_t*>(&one) == 0;
    if (bigendian != host_bigendian) std::reverse(bytes, bytes + sizeof(T));
    std::copy(bytes, bytes + sizeof(T), data.begin() + offset);
}

void testDecoder() {
    // Field order deliberately differs from wire offsets; rows have padding.
    std::vector<std::uint8_t> bytes(88, 0);
    PointCloudView view;
    view.data = bytes.data(); view.size = bytes.size();
    view.width = 2; view.height = 2; view.point_step = 20; view.row_step = 48;
    view.header_ns = 1000000000;
    view.fields = {{"time", 12, 7, 1}, {"y", 16, 7, 1}, {"intensity", 4, 7, 1},
                   {"z", 0, 7, 1}, {"x", 8, 7, 1}};
    for (bool bigendian : {false, true}) {
        view.bigendian = bigendian;
        for (std::size_t i = 0; i < 4; ++i) {
            const auto offset = (i / 2) * 48 + (i % 2) * 20;
            put(bytes, offset + 0, static_cast<float>(30 + i), bigendian);
            put(bytes, offset + 4, static_cast<float>(100 + i), bigendian);
            put(bytes, offset + 8, static_cast<float>(10 + i), bigendian);
            put(bytes, offset + 12, static_cast<float>(i * .02), bigendian);
            put(bytes, offset + 16, static_cast<float>(20 + i), bigendian);
        }
        auto decoded = decode(view, 1, {});
        require(decoded.scan.points.size() == 4, "Padded organized cloud lost points.");
        require(decoded.scan.points[2].x == 12 && decoded.scan.points[2].y == 22 &&
                decoded.scan.points[2].z == 32 && decoded.scan.points[2].intensity == 102,
                "Endian/field-offset decoding differs.");
        require(decoded.start_ns == view.header_ns && std::abs(decoded.end_ns - 1060000000) < 5,
                "Measured timestamps were changed.");
        require(std::abs(decoded.scan.points[1].offset_seconds - .02) < 1e-8, "20ms point offset was stretched.");
        view.height = 1; view.row_step = 0;
        auto single = decode(view, 0, {});
        require(single.scan.points.size() == 2, "UrbanNav single-row zero row_step rejected.");
        SensorConfig end_config;
        end_config.header_is_end = true; end_config.time_offset_seconds = .005;
        auto end_scan = decode(view, 0, end_config);
        require(end_scan.end_ns == 1005000000 && std::abs(end_scan.start_ns - 985000000) < 5,
                "Measured end-header 20ms scan timing incorrect.");
        view.size = 39;
        rejects([&] { decode(view, 0, {}); });
        view.size = bytes.size(); view.height = 2; view.row_step = 48;
    }
    SensorConfig bad;
    bad.time_field = "absent";
    rejects([&] { decode(view, 0, bad); });
    view.row_step = 39;
    rejects([&] { decode(view, 0, {}); });

    // Explicit ring-major LS-C16 timing; a single unordered sweep must fail.
    constexpr std::size_t count = 16 * 50;
    std::vector<std::uint8_t> ring_bytes(count * 16);
    PointCloudView rings;
    rings.data = ring_bytes.data(); rings.size = ring_bytes.size();
    rings.width = count; rings.height = 1; rings.point_step = 16; rings.header_ns = 2000000000;
    rings.fields = {{"x", 0, 7, 1}, {"y", 4, 7, 1}, {"z", 8, 7, 1}, {"intensity", 12, 7, 1}};
    for (std::size_t i = 0; i < count; ++i) {
        const double angle = -static_cast<double>(i) * (2 * 3.14159265358979323846 / 50);
        put(ring_bytes, i * 16, static_cast<float>(std::cos(angle)));
        put(ring_bytes, i * 16 + 4, static_cast<float>(std::sin(angle)));
        put(ring_bytes, i * 16 + 8, static_cast<float>(i / 50));
        put(ring_bytes, i * 16 + 12, 12.f);
    }
    SensorConfig inferred;
    inferred.timing_mode = "azimuth_rings"; inferred.header_is_end = true;
    auto ring_scan = decode(rings, 2, inferred);
    require(ring_scan.start_ns == 1900000000 && ring_scan.end_ns == 2000000000,
            "LS-C16 scan duration/header convention differs.");
    require(ring_scan.timing_source == "estimated_azimuth_rings", "Estimated timing must be labelled.");
    rings.width = 50;
    rejects([&] { decode(rings, 2, inferred); });

    // Partial rings (returns only over 170 deg, then a 190 deg clockwise gap to the next ring):
    // the shortest-angle unwrap loses a turn per gap, but the scan is valid ring-major data and
    // every point keeps its azimuth phase (offset = clockwise angle / 360 deg * duration).
    constexpr std::size_t per_ring = 35;
    constexpr double sector = 170. * 3.14159265358979323846 / 180.;
    std::vector<std::uint8_t> partial_bytes(16 * per_ring * 16);
    PointCloudView partial = rings;
    partial.data = partial_bytes.data(); partial.size = partial_bytes.size(); partial.width = 16 * per_ring;
    for (std::size_t i = 0; i < 16 * per_ring; ++i) {
        const double angle = -static_cast<double>(i % per_ring) * sector / (per_ring - 1);
        put(partial_bytes, i * 16, static_cast<float>(std::cos(angle)));
        put(partial_bytes, i * 16 + 4, static_cast<float>(std::sin(angle)));
        put(partial_bytes, i * 16 + 8, static_cast<float>(i / per_ring));
        put(partial_bytes, i * 16 + 12, 12.f);
    }
    auto partial_scan = decode(partial, 2, inferred);
    for (std::size_t i = 0; i < partial_scan.scan.points.size(); ++i) {
        const double expected = .1 * static_cast<double>(i % per_ring) * (170. / 360.) / (per_ring - 1);
        require(std::abs(partial_scan.scan.points[i].offset_seconds - expected) < 2e-6,
                "Partial-ring LS-C16 azimuth phase changed.");
    }
}

StampedScan scan(int sensor, std::int64_t begin, std::int64_t end) {
    StampedScan result;
    result.scan.sensor_id = sensor; result.start_ns = begin; result.end_ns = end;
    return result;
}

void testSynchronization() {
    Synchronizer sync(2);
    sync.pushScan(scan(0, 0, 100000000));
    sync.pushScan(scan(1, 20000000, 120000000));
    for (int i = 0; i <= 26; ++i) sync.pushImu({i * 10000000LL, {}});
    require(!sync.pop(), "Batch released before 15 future IMU samples.");
    sync.pushImu({270000000, {}});
    auto first = sync.pop();
    require(first && first->imu.front().stamp_ns == 0 && first->imu.back().stamp_ns == 270000000,
            "Initial IMU history/lookahead lost.");
    require(first->scans.size() == 2, "Independent sensors were not retained.");
    sync.pushScan(scan(0, 100000000, 200000000));
    sync.pushScan(scan(1, 120000000, 220000000));
    for (int i = 28; i <= 70; ++i) sync.pushImu({i * 10000000LL, {}});
    auto second = sync.pop();
    require(second && second->imu.front().stamp_ns == 130000000 && second->imu.back().stamp_ns == 370000000,
            "Future IMUs were consumed early, duplicated, or discarded.");
    sync.pushScan(scan(0, 200000000, 300000000));
    sync.pushScan(scan(1, 400000000, 500000000));
    require(!sync.pop() && sync.stats().dropped_scans[0] == 1, "Unmatched early scan was not counted.");
    sync.pushScan(scan(0, 400000000, 500000000));
    auto third = sync.pop();
    require(third && third->imu.front().stamp_ns == 230000000 && sync.stats().batches == 3,
            "IMU coverage across unmatched scans lost.");
    rejects([&] { sync.pushImu({700000000, {}}); });
    rejects([&] { sync.pushScan(scan(0, 400000000, 500000000)); });
    Synchronizer bounded(2, 80000000, 1, 18);
    bounded.pushScan(scan(0, 0, 100000000));
    bounded.pushScan(scan(0, 100000000, 200000000));
    require(bounded.stats().dropped_scans[0] == 1, "Scan queue did not stay bounded.");
    for (int i = 0; i < 20; ++i) bounded.pushImu({i * 10000000LL, {}});
    require(bounded.stats().dropped_imu == 2, "IMU queue did not stay bounded.");

    // UrbanNav begins mid-scan. Its first partial scan initializes IMU biases;
    // discarding it changes initialization statistics and the entire trajectory.
    Synchronizer startup(2);
    startup.pushScan(scan(0, 0, 100000000));
    startup.pushScan(scan(1, 20000000, 120000000));
    for (int i = 5; i <= 27; ++i) startup.pushImu({i * 10000000LL, {}});
    const auto warmup = startup.pop();
    require(warmup && warmup->start_ns == 0 && warmup->imu.front().stamp_ns == 50000000 &&
            startup.stats().dropped_scans == std::vector<std::size_t>({0, 0}),
            "Partial initial scan was discarded instead of warming up the IMU estimator.");
    Synchronizer before_imu(2);
    before_imu.pushScan(scan(0, 0, 100000000));
    before_imu.pushScan(scan(1, 20000000, 120000000));
    for (int i = 15; i <= 40; ++i) before_imu.pushImu({i * 10000000LL, {}});
    require(!before_imu.pop() && before_imu.stats().dropped_scans == std::vector<std::size_t>({1, 1}),
            "Scans ending before all IMU coverage were not dropped.");
}

void testGroundTruth() {
    const auto path = std::filesystem::temp_directory_path() /
        ("ma_slam_gt_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    {
        std::ofstream out(path);
        out << "UrbanNav GT\n20 columns\n"
            << "1000 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 0 2\n"
            << "1002 0 0 0 0 0 0 0 0 12 0 0 0 0 0 0 0 0 90 2\n";
    }
    const auto gt = GroundTruth::load(path.string());
    std::filesystem::remove(path);
    const auto midpoint = gt.poseAt(1001000000000LL);
    require(midpoint && std::abs((*midpoint)(2, 3) - 1.42) < 1e-9, "GT ENU altitude/lever arm/interpolation incorrect.");
    require(std::abs((*midpoint)(0, 0) - std::sqrt(.5)) < 1e-10 && (*midpoint)(1, 0) < 0,
            "GT full orientation convention/slerp incorrect.");
    require(!gt.poseAt(999999999999LL) && !gt.poseAt(1002000000001LL), "GT extrapolated past coverage.");
    Matrix4 local = Matrix4::Identity();
    local.topLeftCorner<3, 3>() = Eigen::AngleAxisd(.4, Eigen::Vector3d::UnitX()).toRotationMatrix();
    local.topRightCorner<3, 1>() << 5, 3, 1;
    require((gt.alignment(1001000000000LL, local) * local - *midpoint).norm() < 1e-10, "Initial ENU alignment did not restore full GT pose.");
    Matrix4 offset = *midpoint;
    offset(0, 3) += 1;
    const auto stats = gt.evaluate({1001000000000LL, 1003000000000LL}, {offset, offset});
    require(stats.matched == 1 && stats.total == 2 && std::abs(stats.rmse - 1) < 1e-12 && stats.vertical_rmse == 0,
            "ATE coverage or distance calculation wrong.");
}
}  // namespace

int main() {
    try {
        testDecoder();
        testSynchronization();
        testGroundTruth();
        std::cout << "Input decoding, timing, synchronization, GT interpolation and ATE: PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
