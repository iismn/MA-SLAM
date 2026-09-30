#pragma once

#include <memory>

namespace ma_lio_core {

// Middleware-neutral records consumed by the retained MA-LIO numerical code.
// Times are seconds from one fixed epoch, never wall-clock or receipt times.
struct Vector3 {
    double x = 0, y = 0, z = 0;
};

struct Imu {
    using ConstPtr = std::shared_ptr<const Imu>;
    double time_seconds = 0;
    Vector3 angular_velocity;
    Vector3 linear_acceleration;
};

struct Pose6D {
    double offset_time = 0;
    double acc[3] = {}, gyr[3] = {}, vel[3] = {}, pos[3] = {}, rot[9] = {};
};

}  // namespace ma_lio_core
