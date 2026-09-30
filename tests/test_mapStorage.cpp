#include "ma_slam/mapStorage.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TestDirectory {
    std::filesystem::path path;
    TestDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "ma_slam-storage-test-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("Cannot create test directory.");
        path = pattern;
    }
    ~TestDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

ma_slam::Cloud cloud(std::initializer_list<std::array<float, 4>> records) {
    ma_slam::Cloud result;
    for (const auto& v : records) {
        pcl::PointXYZI p;
        p.x = v[0]; p.y = v[1]; p.z = v[2]; p.intensity = v[3];
        result.push_back(p);
    }
    return result;
}

std::vector<std::array<float, 4>> readPCD(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    std::string line;
    std::size_t count = 0;
    bool fields = false, binary = false;
    while (std::getline(stream, line)) {
        if (line == "FIELDS x y z intensity") fields = true;
        if (line.rfind("POINTS ", 0) == 0) count = std::stoull(line.substr(7));
        if (line == "DATA binary") { binary = true; break; }
    }
    require(fields && binary, "PCD header does not preserve binary XYZI fields.");
    std::vector<std::array<float, 4>> records(count);
    stream.read(reinterpret_cast<char*>(records.data()), records.size() * sizeof(records[0]));
    require(stream.gcount() == static_cast<std::streamsize>(records.size() * sizeof(records[0])),
            "PCD point count differs from its payload.");
    require(stream.peek() == std::char_traits<char>::eof(), "PCD contains trailing records.");
    return records;
}

std::string readText(const std::filesystem::path& file) {
    std::ifstream stream(file, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void checkPCDAndPublication(const std::filesystem::path& root) {
    ma_slam::MapStorage storage;
    const auto scan = cloud({{.1f, .1f, .1f, 10.f}, {.2f, .2f, .2f, 30.f}, {1.2f, 0.f, 0.f, 80.f}});
    storage.append(scan);
    storage.append(scan);
    ma_slam::OutputConfig config;
    config.directory = (root / "map").string();
    config.voxel_size = 1.;
    config.tile_size = 1.;
    std::vector<ma_slam::Matrix4> poses(2, ma_slam::Matrix4::Identity());
    poses[1](0, 3) = 2.;
    const std::vector<std::int64_t> times = {1621187776000000001LL, 1621187776000000002LL};
    const auto saved = storage.save(config, times, poses);
    require(storage.frames() == 2 && saved.full_points == 6 && saved.voxel_points == 4,
            "Full/voxel counts are incorrect.");
    const auto full = readPCD(root / "map/MA_SLAM_map_Full.pcd");
    require(full.size() == 6, "Full map discarded original points.");
    for (std::size_t frame = 0; frame < 2; ++frame)
        for (std::size_t i = 0; i < scan.size(); ++i) {
            const auto& record = full[frame * scan.size() + i];
            require(std::abs(record[0] - (scan[i].x + float(frame * 2))) < 1e-6,
                    "Full map applied the wrong scan pose.");
            require(record[3] == scan[i].intensity, "Full map changed measured intensity.");
        }
    const auto voxel = readPCD(root / "map/MA_SLAM_map.pcd");
    int means = 0;
    for (const auto& point : voxel)
        if (point[3] == 20.f) {
            ++means;
            require(std::abs(point[1] - .15f) < 1e-6, "Voxel geometry is not a centroid.");
        }
    require(means == 2, "Voxel intensity is not the original-point arithmetic mean.");
    const auto trajectory = readText(root / "map/MA_SLAM_map_trajectory.tum");
    require(trajectory.find("1621187776.000000001") != std::string::npos &&
            trajectory.find("1621187776.000000002") != std::string::npos,
            "Trajectory epoch nanoseconds were rounded.");

    const auto original = readText(root / "map/MA_SLAM_map.pcd");
    bool refused = false;
    try { storage.save(config, times, poses); } catch (const std::runtime_error&) { refused = true; }
    require(refused && readText(root / "map/MA_SLAM_map.pcd") == original,
            "Existing output was replaced without overwrite permission.");
    for (const auto& item : std::filesystem::directory_iterator(root / "map"))
        require(!item.is_directory(), "Export left temporary staging directories behind.");
    config.overwrite = true;
    require(storage.save(config, times, poses).full_points == 6, "Explicit overwrite failed.");

    config.directory = (root / "failed").string();
    auto invalid = poses;
    invalid[1](0, 0) = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { storage.save(config, times, invalid); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !std::filesystem::exists(root / "failed/MA_SLAM_map_Full.pcd"),
            "Invalid late-frame pose published a partial map.");
    invalid = poses;
    invalid[1](0, 3) = 1e39;  // Valid double SE(3), but exceeds the PCD float32 range after staging starts.
    rejected = false;
    try { storage.save(config, times, invalid); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && std::filesystem::is_empty(root / "failed"),
            "A failed staged export left a partial PCD or temporary payload behind.");

    config.directory = (root / "symlink").string();
    config.overwrite = false;
    std::filesystem::create_directories(config.directory);
    const auto dangling = root / "symlink/MA_SLAM_map.pcd";
    std::filesystem::create_symlink(root / "does-not-exist.pcd", dangling);
    refused = false;
    try { storage.save(config, times, poses); } catch (const std::runtime_error&) { refused = true; }
    require(refused && std::filesystem::is_symlink(dangling) &&
            !std::filesystem::exists(root / "symlink/MA_SLAM_map_Full.pcd"),
            "A preexisting symlink was overwritten or a partial result was published.");
}

void checkGlobalVoxels(const std::filesystem::path& root) {
    ma_slam::MapStorage storage;
    // 1/(1/93) rounds just below 93. These two distinct float32 values share
    // global voxel 92 but fall on opposite sides of a separately computed 1 m tile.
    const float below_one = std::nextafter(1.f, 0.f);
    const auto points = cloud({{below_one, .1f, 0.f, 10.f}, {1.f, .1f, 0.f, 30.f},
        {-1.f, -.1f, 0.f, 50.f}, {-.999f, -.1f, 0.f, 70.f},
        {0.f, 0.f, 0.f, 90.f}, {-.001f, -.001f, 0.f, 110.f}});
    storage.append(points);
    ma_slam::OutputConfig config;
    config.directory = (root / "tiles").string();
    config.voxel_size = 1. / 93.;
    config.tile_size = 1.;
    config.save_full = config.save_trajectory = false;
    const auto saved = storage.save(config, {1}, {ma_slam::Matrix4::Identity()});
    using Key = std::tuple<long long, long long, long long>;
    struct Mean { Eigen::Vector4d sum = Eigen::Vector4d::Zero(); int count = 0; };
    std::map<Key, Mean> expected;
    for (const auto& p : points) {
        const Key key{std::floor(p.x / config.voxel_size), std::floor(p.y / config.voxel_size),
                      std::floor(p.z / config.voxel_size)};
        auto& value = expected[key];
        value.sum += Eigen::Vector4d(p.x, p.y, p.z, p.intensity);
        ++value.count;
    }
    const auto exported = readPCD(root / "tiles/MA_SLAM_map.pcd");
    require(saved.voxel_points == expected.size() && exported.size() == expected.size(),
            "One global voxel was split across tile boundaries.");
    for (const auto& [key, value] : expected) {
        const Eigen::Vector4d centroid = value.sum / value.count;
        bool found = false;
        for (const auto& p : exported)
            if ((centroid - Eigen::Vector4d(p[0], p[1], p[2], p[3])).norm() < 1e-6) found = true;
        require(found, "Tiled voxel centroid/intensity differs from global aggregation.");
    }
}

ma_slam::Matrix4 transform(double x, double y, double angle) {
    ma_slam::Matrix4 result = ma_slam::Matrix4::Identity();
    result.topLeftCorner<3, 3>() = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    result(0, 3) = x; result(1, 3) = y;
    return result;
}

void checkInterpolation() {
    const auto frame = transform(100., -50., .4);
    std::vector<ma_slam::Matrix4> raw, graph;
    for (double x : {0., 1., 1., 4., 5.}) raw.push_back(frame * transform(x, 0., 0.));
    const auto correction = transform(2., 1., .6);
    graph = raw;
    graph[3] = frame * correction * transform(4., 0., 0.);
    graph[4] = frame * correction * transform(5., 0., 0.);
    const auto corrected = ma_slam::interpolateCorrections(raw, graph, {0, 3});
    const ma_slam::Matrix4 quarter = frame * transform(.5, .25, .15) * transform(1., 0., 0.);
    require(corrected[0].isApprox(graph[0]) && corrected[3].isApprox(graph[3]) &&
            corrected[4].isApprox(graph[4]), "Interpolation moved an optimized anchor or trailing submap.");
    require(corrected[1].isApprox(quarter, 1e-8) && corrected[2].isApprox(quarter, 1e-8),
            "Correction interpolation used time/index rather than distance, or the wrong frame.");
    const std::vector<ma_slam::Matrix4> stationary(3, frame);
    auto stationary_graph = stationary;
    stationary_graph.back() = frame * correction;
    const auto rotated = ma_slam::interpolateCorrections(stationary, stationary_graph, {0, 2});
    require(rotated[1].isApprox(frame * transform(1., .5, .3), 1e-8),
            "Zero-distance interval failed to interpolate a finite rigid correction.");
    bool rejected = false;
    try { ma_slam::interpolateCorrections(raw, graph, {0, 3, 3}); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Duplicate anchors were accepted.");
}

}  // namespace

int main() {
    try {
        TestDirectory directory;
        checkPCDAndPublication(directory.path);
        checkGlobalVoxels(directory.path);
        checkInterpolation();
        std::cout << "PASS: full XYZI, voxel centroid/intensity, global tile boundaries, exact timestamps, "
                     "no-overwrite/staging and SE(3) interpolation.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
