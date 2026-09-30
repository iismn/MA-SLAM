#include "ma_slam/parameters.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <set>
#include <type_traits>

namespace py = pybind11;
namespace ma_slam {
namespace {

Parameters notebookParameters(const py::dict& overrides)
{
    // -------------------------------------------------------------
    // - Apply flattened ROS2 YAML values through the shared typed config reader.
    // - Reject spelling mistakes instead of silently running with another setting.
    // -------------------------------------------------------------
    std::set<std::string> consumed{"use_sim_time"};
    if (overrides.contains("use_sim_time") && !py::isinstance<py::bool_>(overrides["use_sim_time"]))
        throw std::invalid_argument("use_sim_time must be boolean");
    auto parameters = readParameters([&](const std::string& name, const ParameterValue& fallback) {
        consumed.insert(name);
        if (!overrides.contains(py::str(name))) return fallback;
        const py::handle input = overrides[py::str(name)];
        return std::visit([&](const auto& value) -> ParameterValue {
            using T = std::decay_t<decltype(value)>;
            try {
                if constexpr (std::is_same_v<T, bool>) {
                    if (!py::isinstance<py::bool_>(input)) throw py::cast_error();
                } else if constexpr (std::is_same_v<T, std::int64_t>) {
                    if (!py::isinstance<py::int_>(input) || py::isinstance<py::bool_>(input))
                        throw py::cast_error();
                } else if constexpr (std::is_same_v<T, double>) {
                    if ((!py::isinstance<py::float_>(input) && !py::isinstance<py::int_>(input)) ||
                        py::isinstance<py::bool_>(input)) throw py::cast_error();
                } else if constexpr (std::is_same_v<T, std::vector<std::int64_t>>) {
                    for (auto item : py::reinterpret_borrow<py::iterable>(input))
                        if (!py::isinstance<py::int_>(item) || py::isinstance<py::bool_>(item))
                            throw py::cast_error();
                }
                return py::cast<T>(input);
            } catch (const py::cast_error&) {
                throw std::invalid_argument("MA-SLAM parameter has incorrect type: " + name);
            }
        }, fallback);
    });
    for (const auto& item : overrides) {
        const auto name = py::cast<std::string>(item.first);
        if (!consumed.contains(name)) throw std::invalid_argument("Unknown MA-SLAM parameter: " + name);
    }
    return parameters;
}

py::dict graphStats(const GraphStats& value)
{
    // - Return counters only; progress reporting never copies the trajectory.
    py::dict result;
#define COUNTER(name) result[#name] = value.name
    COUNTER(scans); COUNTER(submaps); COUNTER(closures); COUNTER(gt_edges);
    COUNTER(gt_candidates); COUNTER(gt_skipped_covariance); COUNTER(gt_skipped_distance);
    COUNTER(loop_candidates); COUNTER(loop_rejected); COUNTER(loop_translation_rmse_m);
    COUNTER(loop_rotation_rmse_deg); COUNTER(loop_min_robust_weight);
#undef COUNTER
    return result;
}

class NotebookEngine {
public:
    explicit NotebookEngine(const py::dict& settings, std::size_t max_batches = 0)
        : parameters_(notebookParameters(settings)),
          synchronizer_(parameters_.pipeline.sensors.size(),
              std::llround(parameters_.pair_tolerance_seconds * 1e9),
              parameters_.scan_queue_limit, parameters_.imu_queue_limit, 15),
          pipeline_(parameters_.pipeline), max_batches_(max_batches) {}

    bool done() const { return done_.load(); }

    void pushCloud(int sensor_id, std::int64_t header_ns, const py::buffer& buffer,
        std::uint32_t width, std::uint32_t height, std::uint32_t point_step,
        std::uint32_t row_step, bool bigendian,
        const std::vector<std::tuple<std::string, std::uint32_t, std::uint8_t, std::uint32_t>>& fields,
        const std::string& frame_id)
    {
        // -------------------------------------------------------------
        // - Borrow PointCloud2 bytes only during decoding; no Python cloud math.
        // - Decode, synchronize, deskew, estimate and optimize in the same C++ core as ROS2.
        // -------------------------------------------------------------
        if (done()) return;
        const auto data = buffer.request();
        if (data.ndim != 1 || data.itemsize != 1 || data.strides[0] != 1)
            throw std::invalid_argument("PointCloud2 data must be a contiguous one-dimensional byte buffer");
        if (sensor_id < 0 || static_cast<std::size_t>(sensor_id) >= parameters_.pipeline.sensors.size())
            throw std::invalid_argument("sensor_id is outside the configured LiDAR list");
        const auto& sensor = parameters_.pipeline.sensors[sensor_id];
        if (frame_id != sensor.frame_id)
            throw std::invalid_argument("Unexpected frame_id on " + sensor.topic + ": " + frame_id);
        PointCloudView view;
        view.data = static_cast<const std::uint8_t*>(data.ptr);
        view.size = static_cast<std::size_t>(data.size);
        view.width = width; view.height = height;
        view.point_step = point_step; view.row_step = row_step;
        view.bigendian = bigendian; view.header_ns = header_ns;
        for (const auto& [name, offset, datatype, count] : fields)
            view.fields.push_back({name, offset, datatype, count});
        py::gil_scoped_release release;
        std::lock_guard lock(mutex_);
        if (done()) return;
        synchronizer_.pushScan(decode(view, sensor_id, sensor));
        drain();
    }

    void pushImu(std::int64_t header_ns, const std::array<double, 3>& acceleration,
                 const std::array<double, 3>& angular_velocity)
    {
        // - Preserve the ROS2 additive IMU time-offset convention and native IMU lookahead.
        if (done()) return;
        StampedImu sample;
        sample.stamp_ns = header_ns + std::llround(parameters_.imu_time_offset_seconds * 1e9);
        sample.sample.acceleration = Eigen::Map<const Eigen::Vector3d>(acceleration.data());
        sample.sample.angular_velocity = Eigen::Map<const Eigen::Vector3d>(angular_velocity.data());
        if (!sample.sample.acceleration.allFinite() || !sample.sample.angular_velocity.allFinite())
            throw std::invalid_argument("IMU acceleration and angular velocity must be finite");
        py::gil_scoped_release release;
        std::lock_guard lock(mutex_);
        if (done()) return;
        synchronizer_.pushImu(std::move(sample));
        drain();
    }

    py::dict stats() const
    {
        std::lock_guard lock(mutex_);
        return statsUnlocked();
    }

    py::dict finish()
    {
        // - Finalize once without saving; preserve the existing user maps for comparison.
        GraphResult trajectory;
        std::string report;
        {
            py::gil_scoped_release release;
            std::lock_guard lock(mutex_);
            done_ = true;
            if (!pipeline_.finalized()) report_ = pipeline_.finalize();
            report = report_;
            trajectory = pipeline_.trajectory();
            graph_stats_ = trajectory.stats;
        }
        const auto count = static_cast<py::ssize_t>(trajectory.poses.size());
        py::array_t<std::int64_t> times(count);
        py::array_t<double> poses(std::vector<py::ssize_t>{count, 4, 4});
        auto t = times.mutable_unchecked<1>();
        auto p = poses.mutable_unchecked<3>();
        for (py::ssize_t i = 0; i < count; ++i) {
            t(i) = trajectory.times_ns[i];
            for (int row = 0; row < 4; ++row)
                for (int col = 0; col < 4; ++col) p(i, row, col) = trajectory.poses[i](row, col);
        }
        py::dict result;
        result["report"] = report;
        result["times_ns"] = times;
        result["poses"] = poses;
        result["loop_pairs"] = trajectory.loop_pairs;
        result["submap_start_indices"] = trajectory.submap_start_indices;
        result["gt_factor_times_ns"] = trajectory.gt_factor_times_ns;
        result["stats"] = stats();
        return result;
    }

    py::dict save()
    {
        // - Saving is an explicit notebook operation, using the native tiled PCD writer.
        SavedMap saved;
        {
            py::gil_scoped_release release;
            std::lock_guard lock(mutex_);
            if (!pipeline_.finalized()) throw std::logic_error("Call finish() before save()");
            saved = pipeline_.save();
        }
        std::vector<std::string> files;
        for (const auto& file : saved.files) files.push_back(file.string());
        py::dict result;
        result["files"] = files;
        result["full_points"] = saved.full_points;
        result["voxel_points"] = saved.voxel_points;
        return result;
    }

    py::array_t<float> preview() const
    {
        // - Return the bounded native preview, not the full dense map in Python RAM.
        Cloud cloud;
        {
            py::gil_scoped_release release;
            std::lock_guard lock(mutex_);
            cloud = pipeline_.preview();
        }
        py::array_t<float> result(std::vector<py::ssize_t>{static_cast<py::ssize_t>(cloud.size()), 4});
        auto output = result.mutable_unchecked<2>();
        for (std::size_t i = 0; i < cloud.size(); ++i) {
            output(i, 0) = cloud[i].x; output(i, 1) = cloud[i].y;
            output(i, 2) = cloud[i].z; output(i, 3) = cloud[i].intensity;
        }
        return result;
    }

private:
    void drain()
    {
        while (!done()) {
            auto batch = synchronizer_.pop();
            if (!batch) break;
            auto update = pipeline_.process(*batch);
            if (update) {
                graph_stats_ = update->graph;
                effective_points_ = update->effective_points;
                unsupported_points_ += update->unsupported_points;
            }
            if (max_batches_ && synchronizer_.stats().batches >= max_batches_) done_ = true;
        }
    }

    py::dict statsUnlocked() const
    {
        auto result = graphStats(graph_stats_);
        result["batches"] = synchronizer_.stats().batches;
        result["frames"] = pipeline_.frames();
        result["dropped_scans"] = synchronizer_.stats().dropped_scans;
        result["dropped_imu"] = synchronizer_.stats().dropped_imu;
        result["queued_scans"] = synchronizer_.queuedScans();
        result["effective_points"] = effective_points_;
        result["unsupported_points"] = unsupported_points_;
        result["done"] = done();
        result["finalized"] = pipeline_.finalized();
        return result;
    }

    Parameters parameters_;
    Synchronizer synchronizer_;
    Pipeline pipeline_;
    std::size_t max_batches_ = 0, effective_points_ = 0, unsupported_points_ = 0;
    std::atomic<bool> done_{false};
    mutable std::mutex mutex_;
    GraphStats graph_stats_;
    std::string report_;
};

}  // namespace
}  // namespace ma_slam

PYBIND11_MODULE(_ma_slam_native, module)
{
    module.doc() = "Optional notebook transport adapter; MA-SLAM estimation remains in the C++ core.";
    module.attr("lidar_count") = ma_slam::ImuPreintegration::lidarCount();
    py::class_<ma_slam::NotebookEngine>(module, "Engine")
        .def(py::init<const py::dict&, std::size_t>(), py::arg("settings"), py::arg("max_batches") = 0)
        .def_property_readonly("done", &ma_slam::NotebookEngine::done)
        .def("push_cloud", &ma_slam::NotebookEngine::pushCloud,
             py::arg("sensor_id"), py::arg("header_ns"), py::arg("buffer"),
             py::arg("width"), py::arg("height"), py::arg("point_step"), py::arg("row_step"),
             py::arg("bigendian"), py::arg("fields"), py::arg("frame_id"))
        .def("push_imu", &ma_slam::NotebookEngine::pushImu,
             py::arg("header_ns"), py::arg("acceleration"), py::arg("angular_velocity"))
        .def("stats", &ma_slam::NotebookEngine::stats)
        .def("finish", &ma_slam::NotebookEngine::finish)
        .def("save", &ma_slam::NotebookEngine::save)
        .def("preview", &ma_slam::NotebookEngine::preview);
}
