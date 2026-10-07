#include "ma_slam/pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
using namespace ma_slam;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ma_slam_pipeline_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

std::vector<std::array<float, 4>> readPcd(const std::filesystem::path& path) {
    // -----------------------------------------------------------------
    // - Check the actual binary XYZI file produced by the complete pipeline.
    // -----------------------------------------------------------------
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "Saved PCD is missing.");
    std::string line;
    std::size_t points = 0;
    bool intensity = false, binary = false;
    while (std::getline(file, line)) {
        if (line == "FIELDS x y z intensity") intensity = true;
        if (line.rfind("POINTS ", 0) == 0) points = std::stoull(line.substr(7));
        if (line == "DATA binary") { binary = true; break; }
    }
    require(points > 0 && intensity && binary, "Saved PCD does not declare binary XYZI.");
    std::vector<std::array<float, 4>> result(points);
    const auto bytes = result.size() * sizeof(result.front());
    file.read(reinterpret_cast<char*>(result.data()), bytes);
    require(static_cast<std::size_t>(file.gcount()) == bytes, "Saved PCD payload is truncated.");
    require(file.peek() == std::char_traits<char>::eof(), "Saved PCD point count differs from payload.");
    return result;
}
}  // namespace

int main() {
    // -----------------------------------------------------------------
    // - Run original C++ MA-LIO -> g2o graph -> final dense/voxel XYZI export.
    // - Exercise center extrinsic, independent 20/100ms scans and GT alignment.
    // - Create only private temporary output; never touch dataset result files.
    // -----------------------------------------------------------------
    using namespace ma_slam;
    try {
        TemporaryDirectory temporary;
        const auto gt_path = temporary.path / "gt.txt";
        {
            std::ofstream gt(gt_path);
            gt << "UrbanNav GT\n20 columns\n"
               << "1000.65 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 -30 2\n"
               << "1002 0 0 0 0 0 0 0 0 10 0 0 0 0 0 0 0 0 -30 2\n";
        }
        const int count = ImuPreintegration::lidarCount();
        PipelineConfig config;
        config.sensors.resize(count);
        config.lidar_to_imu.assign(count, Matrix4::Identity());
        config.lidar_to_imu[0].topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(.08, Eigen::Vector3d::UnitX()).toRotationMatrix();
        config.lidar_to_imu[0].topRightCorner<3, 1>() << .3, -.05, .1;
        config.lidar_to_imu[1].topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(.96, Eigen::Vector3d::UnitY()).toRotationMatrix();
        config.lidar_to_imu[1].topRightCorner<3, 1>() << .3, 0, -.2;
        if (count == 3) {
            config.lidar_to_imu[2].topLeftCorner<3, 3>() =
                (Eigen::AngleAxisd(.96, Eigen::Vector3d::UnitY()) *
                 Eigen::AngleAxisd(1.59, Eigen::Vector3d::UnitZ())).toRotationMatrix();
            config.lidar_to_imu[2].topRightCorner<3, 1>() << -.27, 0, -.22;
        }
        config.frontend.extrinsic_est_en = false;
        config.graph.submap_distance = .02;
        config.graph.loop_search_radius = 0;  // Synthetic loop recovery has its own graph test.
        config.graph.min_range = 0;
        config.gt_file = gt_path.string();
        config.output.directory = temporary.path.string();
        config.output.prefix = "pipeline";
        config.output.voxel_size = .15;
        config.output.tile_size = 3;
        config.preview_max_points = 1000;
        Pipeline pipeline(config);
        require(pipeline.trajectory().poses.empty() && pipeline.preview().empty(), "Startup snapshot/preview is not empty.");
        std::mt19937 random(8);
        std::uniform_real_distribution<double> uniform(-10, 10);
        std::vector<Eigen::Vector3d> scene;
        for (int plane = 0; plane < 3; ++plane) {
            for (int point = 0; point < 2000; ++point) {
                const double a = uniform(random), b = uniform(random);
                if (plane == 0) scene.emplace_back(a, b, -2);
                if (plane == 1) scene.emplace_back(8, a, b);
                if (plane == 2) scene.emplace_back(a, 8, b);
            }
        }
        constexpr std::int64_t epoch = 1000000000000LL;
        // Partial GT starts after the bag and after motion begins (yaw from 0.5 s):
        // the frontend must initialize at the static bag start, while poses before GT stay unregistered.
        constexpr std::int64_t gt_start = epoch + 650000000LL;
        std::vector<PipelineUpdate> updates;
        Matrix4 alignment = Matrix4::Identity();
        double previous = 0;
        for (int batch_index = 0; batch_index < 16; ++batch_index) {
            SynchronizedBatch batch;
            batch.start_ns = std::numeric_limits<std::int64_t>::max();
            double end_seconds = 0;
            for (int sensor = 0; sensor < count; ++sensor) {
                const double duration = sensor == 2 ? .019 : .099;
                const double end = batch_index * .1 + .099 + (sensor == batch_index % count ? .003 : 0);
                const double begin = end - duration;
                StampedScan item;
                item.start_ns = epoch + std::llround(begin * 1e9);
                item.end_ns = epoch + std::llround(end * 1e9);
                item.scan.sensor_id = sensor;
                const auto& extrinsic = config.lidar_to_imu[sensor];
                for (std::size_t index = 0; index < scene.size(); ++index) {
                    const double offset = duration * index / (scene.size() - 1);
                    const double yaw = .3 * std::max(begin + offset - .5, 0.0);
                    const auto body_rotation = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
                    const Eigen::Vector3d point = extrinsic.topLeftCorner<3, 3>().transpose() *
                        (body_rotation.transpose() * scene[index] - extrinsic.topRightCorner<3, 1>());
                    item.scan.points.push_back({float(point.x()), float(point.y()), float(point.z()),
                                                float(index % 255), offset});
                }
                batch.start_ns = std::min(batch.start_ns, item.start_ns);
                end_seconds = std::max(end_seconds, end);
                batch.scans.push_back(std::move(item));
            }
            batch.end_ns = epoch + std::llround(end_seconds * 1e9);
            for (double time = previous + 1e-6; time < end_seconds + .05; time += .0025) {
                StampedImu imu;
                imu.stamp_ns = epoch + std::llround(time * 1e9);
                imu.sample.acceleration << 0, 0, 9.81;
                imu.sample.angular_velocity << 0, 0, time >= .5 ? .3 : 0;
                batch.imu.push_back(imu);
            }
            previous = end_seconds;
            auto update = pipeline.process(batch);
            if (!update) {
                require(updates.empty(), "A registered frame was dropped after GT coverage began.");
                continue;
            }
            require(update->map_pose.allFinite() && update->odom_pose.allFinite(), "Pipeline returned non-finite pose.");
            if (updates.empty()) {
                require(update->stamp_ns >= gt_start, "A pre-GT pose was registered.");
                const double yaw = std::atan2(update->odom_pose(1, 0), update->odom_pose(0, 0));
                require(std::abs(yaw) > .03,
                        "Frontend was not initialized before GT coverage (odom frame starts at GT).");
                const auto reference = *GroundTruth::load(gt_path.string()).poseAt(update->stamp_ns);
                require(update->map_pose.isApprox(reference, 1e-9), "GT initial orientation/center-LiDAR alignment changed.");
                alignment = update->map_pose * update->odom_pose.inverse();
            }
            require(update->cloud.size() > 1000, "Dense multi-LiDAR clouds were lost.");
            if (batch_index >= 2) {
                std::vector<double> errors;
                for (const auto& p : update->cloud) {
                    const Eigen::Vector3d world = update->odom_pose.topLeftCorner<3, 3>() * Eigen::Vector3d(p.x, p.y, p.z)
                                                   + update->odom_pose.topRightCorner<3, 1>();
                    errors.push_back(std::min({std::abs(world.z() + 2), std::abs(world.x() - 8), std::abs(world.y() - 8)}));
                }
                std::sort(errors.begin(), errors.end());
                require(errors[static_cast<std::size_t>(.95 * (errors.size() - 1))] < .1,
                        "Center cloud frame/extrinsic or dense deskew changed through pipeline.");
            }
            updates.push_back(std::move(*update));
        }
        require(updates.size() >= 8 && pipeline.frames() == updates.size(), "Pipeline discarded valid registered frames.");
        const auto preview = pipeline.preview();
        require(!preview.empty() && preview.size() <= config.preview_max_points, "Live preview is missing or unbounded.");
        const auto report = pipeline.finalize();
        require(pipeline.finalized() && report == pipeline.finalize(), "Finalization is not idempotent.");
        require(report.find("Matched:") != std::string::npos, "GT ATE report was not produced.");
        const auto solution = pipeline.trajectory();
        require(solution.poses.size() == updates.size() && solution.stats.submaps >= 2,
                "Graph submaps or complete corrected trajectory were lost.");
        std::size_t expected_points = 0;
        for (std::size_t i = 0; i < updates.size(); ++i) {
            require(solution.times_ns[i] == updates[i].stamp_ns, "Graph timestamps differ from actual frontend pose times.");
            require(solution.poses[i].isApprox(alignment * updates[i].odom_pose, 1e-6),
                    "Odometry-only graph or correction interpolation changed frame conventions.");
            expected_points += updates[i].cloud.size();
        }
        const auto saved = pipeline.save();
        require(saved.full_points == expected_points && saved.voxel_points > 0 && saved.voxel_points < saved.full_points,
                "Dense and voxel export counts are inconsistent.");
        require(saved.files.size() == 3, "Full/voxel map and trajectory were not saved together.");
        const auto full_path = temporary.path / "pipeline_Full.pcd";
        const auto full = readPcd(full_path);
        const auto voxel = readPcd(temporary.path / "pipeline.pcd");
        require(full.size() == saved.full_points && voxel.size() == saved.voxel_points, "PCD metadata count differs from saved count.");
        std::size_t index = 0;
        for (std::size_t frame = 0; frame < updates.size(); ++frame) {
            for (const auto& point : updates[frame].cloud) {
                const auto& pose = solution.poses[frame];
                const Eigen::Vector3d expected = pose.topLeftCorner<3, 3>() * Eigen::Vector3d(point.x, point.y, point.z)
                                                   + pose.topRightCorner<3, 1>();
                const auto& stored = full[index++];
                require((Eigen::Vector3d(stored[0], stored[1], stored[2]) - expected).norm() < 2e-6,
                        "Saved dense map applied the wrong optimized pose/cloud frame.");
                require(stored[3] == point.intensity, "Dense PCD intensity changed.");
            }
        }
        const auto bytes_before = std::filesystem::file_size(full_path);
        bool refused = false;
        try { pipeline.save(); } catch (const std::exception&) { refused = true; }
        require(refused && std::filesystem::file_size(full_path) == bytes_before, "Repeated save overwrote an existing map.");
        bool rejected = false;
        try { pipeline.process({}); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Finalized pipeline accepted new input.");
        std::cout << "PASS: " << count << "-LiDAR native pipeline; dense deskew, GT frame alignment, g2o trajectory, "
                     "preview, final XYZI/voxel PCD, timestamps, intensity and no-overwrite safety.\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
