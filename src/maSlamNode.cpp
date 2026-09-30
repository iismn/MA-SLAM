#include "ma_slam/parameters.hpp"
#include "ma_slam/pipeline.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <type_traits>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>

namespace ma_slam {
namespace {
Parameters rosParameters(rclcpp::Node& node) {
    // - Preserve ROS2 typed declarations while sharing config validation with native execution.
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.read_only = true;
    return readParameters([&](const std::string& name, const ParameterValue& fallback) {
        return std::visit([&](const auto& value) -> ParameterValue {
            using T = std::decay_t<decltype(value)>;
            return node.declare_parameter<T>(name, value, descriptor);
        }, fallback);
    });
}

std::int64_t nanoseconds(const builtin_interfaces::msg::Time& stamp) {
    return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

geometry_msgs::msg::Pose poseMessage(const Matrix4& matrix) {
    geometry_msgs::msg::Pose pose;
    Eigen::Quaterniond q(matrix.topLeftCorner<3, 3>());
    q.normalize();
    pose.position.x = matrix(0, 3); pose.position.y = matrix(1, 3); pose.position.z = matrix(2, 3);
    pose.orientation.x = q.x(); pose.orientation.y = q.y(); pose.orientation.z = q.z(); pose.orientation.w = q.w();
    return pose;
}

geometry_msgs::msg::TransformStamped transformMessage(const Matrix4& matrix,
    std::int64_t stamp, const std::string& parent, const std::string& child) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = rclcpp::Time(stamp, RCL_ROS_TIME);
    transform.header.frame_id = parent;
    transform.child_frame_id = child;
    const auto pose = poseMessage(matrix);
    transform.transform.translation.x = pose.position.x;
    transform.transform.translation.y = pose.position.y;
    transform.transform.translation.z = pose.position.z;
    transform.transform.rotation = pose.orientation;
    return transform;
}
}

class MaSlamNode final : public rclcpp::Node {
public:
    MaSlamNode() : Node("ma_slam"), parameters_(rosParameters(*this)),
        synchronizer_(parameters_.pipeline.sensors.size(),
            std::llround(parameters_.pair_tolerance_seconds * 1e9),
            parameters_.scan_queue_limit, parameters_.imu_queue_limit, 15),
        pipeline_(parameters_.pipeline) {
        const auto live_qos = rclcpp::QoS(1).reliable();
        const auto retained_qos = rclcpp::QoS(1).reliable().transient_local();
        odometry_ = create_publisher<nav_msgs::msg::Odometry>("~/odometry", live_qos);
        path_ = create_publisher<nav_msgs::msg::Path>("~/path", retained_qos);
        cloud_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/cloud_registered", live_qos);
        map_ = create_publisher<sensor_msgs::msg::PointCloud2>("~/map", retained_qos);
        loops_ = create_publisher<visualization_msgs::msg::MarkerArray>("~/loop_constraints", retained_qos);
        broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        for (std::size_t id = 0; id < parameters_.pipeline.sensors.size(); ++id) {
            const auto sensor = parameters_.pipeline.sensors[id];
            scans_.push_back(create_subscription<sensor_msgs::msg::PointCloud2>(sensor.topic,
                rclcpp::SensorDataQoS().keep_last(parameters_.scan_queue_limit),
                [this, id, sensor](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
                    if (!accepting()) return;
                    try {
                        if (!sensor.frame_id.empty() && message->header.frame_id != sensor.frame_id)
                            throw std::runtime_error("Unexpected frame_id on " + sensor.topic + ": " + message->header.frame_id);
                        PointCloudView view;
                        view.data = message->data.data(); view.size = message->data.size();
                        view.width = message->width; view.height = message->height;
                        view.point_step = message->point_step; view.row_step = message->row_step;
                        view.bigendian = message->is_bigendian; view.header_ns = nanoseconds(message->header.stamp);
                        for (const auto& field : message->fields)
                            view.fields.push_back({field.name, field.offset, field.datatype, field.count});
                        auto scan = decode(view, static_cast<int>(id), sensor);
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            if (!accepting_) return;
                            synchronizer_.pushScan(std::move(scan));
                            ++generation_;
                        }
                        condition_.notify_one();
                    } catch (const std::exception& error) { fail("LiDAR input: " + std::string(error.what())); }
                }));
        }
        imu_ = create_subscription<sensor_msgs::msg::Imu>(parameters_.imu_topic,
            rclcpp::SensorDataQoS().keep_last(parameters_.imu_queue_limit),
            [this](sensor_msgs::msg::Imu::ConstSharedPtr message) {
                if (!accepting()) return;
                try {
                    StampedImu item;
                    item.stamp_ns = nanoseconds(message->header.stamp) +
                                    std::llround(parameters_.imu_time_offset_seconds * 1e9);
                    item.sample.acceleration = {message->linear_acceleration.x,
                        message->linear_acceleration.y, message->linear_acceleration.z};
                    item.sample.angular_velocity = {message->angular_velocity.x,
                        message->angular_velocity.y, message->angular_velocity.z};
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (!accepting_) return;
                        synchronizer_.pushImu(std::move(item));
                        ++generation_;
                    }
                    condition_.notify_one();
                } catch (const std::exception& error) { fail("IMU input: " + std::string(error.what())); }
            });
        finish_ = create_service<std_srvs::srv::Trigger>("~/finish",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr response) { request(false, *response); });
        save_ = create_service<std_srvs::srv::Trigger>("~/save_map",
            [this](const std_srvs::srv::Trigger::Request::SharedPtr,
                   std_srvs::srv::Trigger::Response::SharedPtr response) { request(true, *response); });
        worker_ = std::thread([this] { work(); });
        RCLCPP_INFO(get_logger(), "MA-SLAM ready: %zu LiDARs; C++ MA-LIO + g2o. Saving is explicit via ~/save_map.",
                    parameters_.pipeline.sensors.size());
    }

    ~MaSlamNode() override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            accepting_ = false;
        }
        condition_.notify_one();
        if (worker_.joinable()) worker_.join();
    }

private:
    struct Command {
        bool save;
        std::promise<std::pair<bool, std::string>> completion;
    };

    bool accepting() {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepting_;
    }

    void fail(const std::string& error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_.empty()) return;
            failure_ = error;
            accepting_ = false;
            if (command_) { command_->completion.set_value({false, error}); command_.reset(); }
        }
        RCLCPP_ERROR(get_logger(), "%s; stopped accepting data. No automatic map export.", error.c_str());
        condition_.notify_one();
    }

    void request(bool save, std_srvs::srv::Trigger::Response& response) {
        auto command = std::make_shared<Command>();
        command->save = save;
        auto future = command->completion.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_.empty() || stopping_ || command_) {
                response.success = false;
                response.message = failure_.empty() ? "Finish/save already in progress or shutting down" : failure_;
                return;
            }
            accepting_ = false;
            command_ = command;
        }
        condition_.notify_one();
        const auto result = future.get();
        response.success = result.first;
        response.message = result.second;
    }

    void work() {
        std::size_t observed = 0;
        for (;;) {
            std::optional<SynchronizedBatch> batch;
            std::shared_ptr<Command> command;
            bool stop = false;
            try {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    condition_.wait(lock, [&] { return stopping_ || command_ || !failure_.empty() || generation_ != observed; });
                    if (!failure_.empty()) return;
                    batch = synchronizer_.pop();
                    if (!batch) {
                        observed = generation_;
                        stop = stopping_;
                        command = std::move(command_);
                        if (command || stop) {
                            const auto& stats = synchronizer_.stats();
                            std::size_t dropped = 0;
                            for (auto count : stats.dropped_scans) dropped += count;
                            RCLCPP_INFO(get_logger(), "Input drained: %zu synchronized batches, %zu unmatched/overflow LiDAR scans, %zu queued tail scans (no complete IMU lookahead).",
                                stats.batches, dropped, synchronizer_.queuedScans());
                        }
                    }
                }
                if (batch) {
                    const auto update = pipeline_.process(*batch);
                    if (update) publish(*update);
                    continue;
                }
                if (command || stop) {
                    std::string report = pipeline_.finalize();
                    publishSolution(true);
                    RCLCPP_INFO(get_logger(), "%s", report.c_str());
                    if ((command && command->save) || (stop && parameters_.pipeline.output.save_on_shutdown)) {
                        RCLCPP_INFO(get_logger(), "Exporting optimized XYZI scans; large full maps can take time.");
                        const auto saved = pipeline_.save();
                        for (const auto& path : saved.files) report += "\nSAVED " + path.string();
                        report += "\nFull points: " + std::to_string(saved.full_points) +
                                  "; voxel points: " + std::to_string(saved.voxel_points);
                        RCLCPP_INFO(get_logger(), "%s", report.c_str());
                    }
                    if (command) command->completion.set_value({true, report});
                    if (stop) return;
                }
            } catch (const std::exception& error) {
                if (command) {
                    // A save-path error can be retried after fixing the filesystem.
                    command->completion.set_value({false, error.what()});
                    RCLCPP_ERROR(get_logger(), "Finish/save failed: %s", error.what());
                    if (stop) return;
                } else { fail(error.what()); return; }
            }
        }
    }

    void publish(const PipelineUpdate& update) {
        if (!rclcpp::ok()) return;
        last_stamp_ = update.stamp_ns;
        last_odom_pose_ = update.odom_pose;
        nav_msgs::msg::Odometry odometry;
        odometry.header.stamp = rclcpp::Time(update.stamp_ns, RCL_ROS_TIME);
        odometry.header.frame_id = parameters_.frame_odom;
        odometry.child_frame_id = parameters_.frame_lidar;
        odometry.pose.pose = poseMessage(update.odom_pose);
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                odometry.pose.covariance[row * 6 + column] = update.position_covariance(row, column);
        for (int i = 3; i < 6; ++i) odometry.pose.covariance[i * 6 + i] = 1e6;
        for (int i = 0; i < 6; ++i) odometry.twist.covariance[i * 6 + i] = 1e6;
        odometry_->publish(odometry);
        broadcaster_->sendTransform(std::vector<geometry_msgs::msg::TransformStamped>{
            transformMessage(update.map_pose * update.odom_pose.inverse(), update.stamp_ns,
                             parameters_.frame_map, parameters_.frame_odom),
            transformMessage(update.odom_pose, update.stamp_ns,
                             parameters_.frame_odom, parameters_.frame_lidar)});
        if (cloud_->get_subscription_count() > 0) {
            Cloud registered;
            pcl::transformPointCloud(update.cloud, registered, update.map_pose);
            publishCloud(registered, cloud_);
        }
        if (update.submap_changed || pipeline_.frames() == 1 || pipeline_.frames() % 10 == 0) {
            const bool map_due = update.submap_changed &&
                update.graph.submaps % parameters_.map_publish_every_submaps == 0;
            publishSolution(map_due || pipeline_.frames() == 1);
        }
        if (pipeline_.frames() % 100 == 0)
            RCLCPP_INFO(get_logger(), "scans=%zu submaps=%zu loops=%zu GT=%zu effective=%zu unsupported=%zu",
                pipeline_.frames(), update.graph.submaps, update.graph.closures, update.graph.gt_edges,
                update.effective_points, update.unsupported_points);
    }

    void publishCloud(const Cloud& cloud, const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher) {
        sensor_msgs::msg::PointCloud2 message;
        pcl::toROSMsg(cloud, message);
        message.header.stamp = rclcpp::Time(last_stamp_, RCL_ROS_TIME);
        message.header.frame_id = parameters_.frame_map;
        publisher->publish(message);
    }

    void publishSolution(bool publish_map) {
        if (!rclcpp::ok()) return;
        const auto solution = pipeline_.trajectory();
        if (pipeline_.finalized() && !solution.poses.empty()) {
            // Final graph optimization can change the correction after the last scan.
            broadcaster_->sendTransform(transformMessage(
                solution.poses.back() * last_odom_pose_.inverse(), last_stamp_,
                parameters_.frame_map, parameters_.frame_odom));
        }
        nav_msgs::msg::Path path;
        path.header.stamp = rclcpp::Time(last_stamp_, RCL_ROS_TIME);
        path.header.frame_id = parameters_.frame_map;
        path.poses.reserve(solution.poses.size());
        for (std::size_t i = 0; i < solution.poses.size(); ++i) {
            geometry_msgs::msg::PoseStamped pose;
            pose.header = path.header;
            pose.header.stamp = rclcpp::Time(solution.times_ns[i], RCL_ROS_TIME);
            pose.pose = poseMessage(solution.poses[i]);
            path.poses.push_back(pose);
        }
        path_->publish(path);
        visualization_msgs::msg::Marker marker;
        marker.header = path.header; marker.ns = "loops"; marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1;
        marker.scale.x = .15; marker.color.r = 1; marker.color.g = .3; marker.color.a = 1;
        for (const auto& [a, b] : solution.loop_pairs) {
            for (int id : {a, b}) {
                if (id < 0 || static_cast<std::size_t>(id) >= solution.submap_start_indices.size()) continue;
                const auto index = solution.submap_start_indices[id];
                if (index < path.poses.size()) marker.points.push_back(path.poses[index].pose.position);
            }
        }
        visualization_msgs::msg::MarkerArray markers;
        markers.markers.push_back(marker);
        loops_->publish(markers);
        if (publish_map) publishCloud(pipeline_.preview(), map_);
    }

    Parameters parameters_;
    Synchronizer synchronizer_;
    Pipeline pipeline_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool accepting_ = true, stopping_ = false;
    std::size_t generation_ = 0;
    std::string failure_;
    std::shared_ptr<Command> command_;
    std::thread worker_;
    std::int64_t last_stamp_ = 0;
    Matrix4 last_odom_pose_ = Matrix4::Identity();
    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> scans_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_, map_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr loops_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr finish_, save_;
};
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    int status = 0;
    try {
        auto node = std::make_shared<ma_slam::MaSlamNode>();
        rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
        executor.add_node(node);
        executor.spin();
        executor.remove_node(node);
    } catch (const std::exception& error) {
        RCLCPP_FATAL(rclcpp::get_logger("ma_slam"), "%s", error.what());
        status = 1;
    }
    rclcpp::shutdown();
    return status;
}
