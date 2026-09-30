MA-SLAM — C++ / ROS2 (ament_cmake)
================================

This is the new project; MA_SLAM_TMP is retained unchanged for comparison.
The ROS2 executable has no Python or RKO package dependency. There is no ROS1
shim, file_player, irp_sen_msgs or figures directory in this project.
Standard ROS2 PointCloud2 and Imu messages feed the original MA-LIO C++ core.
An optional notebook transport binding calls the same C++ Pipeline offline;
it does not reimplement the estimator in Python or require ROS2 for bag replay.

Source layout
-------------
src/maSlamNode.cpp       ROS2 subscriptions, worker, publishers, TF and services
src/pipeline.cpp         Complete processing order, frames, graph and export
src/imageProjection.cpp PointCloud2 decoding, point timing and bounded synchronization
src/imuPreintegration.cpp MA-LIO IMU initialization, B-spline deskew and IESKF adapter
src/lidarOdometry.inc    Original MA-LIO planar scan matching / local-map update
src/mapOptimization.cpp g2o pose graph, geometric loop verification and GT factors
src/groundTruth.cpp     UrbanNav raw GT -> ENU, lever arm, interpolation, ATE
src/mapStorage.cpp      Dense XYZI spool, corrected full / global voxel PCD export
src/parameters.cpp      Shared typed settings and validation (ROS2 / notebook)
src/notebookBindings.cpp Optional byte-buffer/config transport for the notebook
include/ma_slam/         Matching C++ interfaces
config/params.yaml      All editable runtime parameters, paths and extrinsics
launch/run.launch.xml   ROS2 launch; no Python launch code
rviz/ma_slam.rviz        Live map, latest cloud, trajectory and loop visualization
third_party/ma_lio/     Necessary upstream numerical sources and original notices
third_party/g2o/        Bundled graph optimizer; not an RKO dependency
tests/                  Native regression tests and ROS2 service smoke test

The file naming follows LIO-SAM's separation of responsibilities. The numerical
frontend remains MA-LIO (IESKF + B-spline deskew); it is not LIO-SAM or an exact
unmodified copy of upstream MA-LIO. Retained bug fixes and upstream revision are
listed in third_party/ma_lio/PROVENANCE.txt. Component license headers are retained.

Notebook on this Mac
--------------------
Open Utils/GenCode/Notebook/runBuildMASLAMMap.ipynb and run cells in order.
All Python conversion/build/progress code is in one helper:
Utils/GenCode/Notebook/Utils/maslam_runner.py. It imports no MA_SLAM_TMP code.
The helper uses InhouseBuild/.venv in a separate process and reads the existing
ROS1 bag directly. No whole-bag or ROS2 format conversion is needed. Point timing,
synchronization, deskew, estimation, graph optimization and export remain C++.

CONFIG selects the same config/params.yaml as ROS2. MAX_BATCHES=0 means all;
SAVE_MAP=True exports Full/voxel XYZI and TUM; DRY_RUN=True only builds/checks input.
For a short comparison set MAX_BATCHES=100 and SAVE_MAP=False. OVERRIDES uses YAML
keys, e.g. output.prefix or gt.position_covariance. The OLD grouped YAML is also
accepted with explicit covariance, calibration and IMU-offset conversion. Its
old output path is retained; choose another output prefix/directory for comparison.
Existing output files are refused before processing when overwrite=false.

MA_SLAM_WITH_PYTHON=ON enables the optional native binding; it remains OFF by
default for ROS2. The helper configures this automatically with ROS2 disabled.
Builds/logs live in MA_SLAM/build, outside the dataset. Python tools use uv:

  source ~/Documents/InhouseBuild/.venv/bin/activate
  uv pip install --python "$VIRTUAL_ENV/bin/python" \
    numpy pyyaml rosbags cmake pybind11 tqdm ipywidgets matplotlib

From the notebook directory:
  python Utils/maslam_runner.py --self-check
  python Utils/maslam_runner.py --max-batches 30 --no-save

The result contains optimized poses, loop indices, GT factor times and a bounded
preview in memory. No persistent JSON/CSV/NPZ result is generated. Temporary IPC
arrays and dense scan storage are cleaned up. A forced kill during native PCD
export can leave an incomplete output/ma_slam-* staging folder; existing published
files remain protected. Saving is controlled by SAVE_MAP in this offline wrapper,
independently of the ROS2 output.save_on_shutdown parameter.

Processing and coordinate conventions
------------------------------------
Each LiDAR keeps its own scan start/end and point times. The synchronizer chooses
one scan per sensor within the configured tolerance; it does not concatenate raw
scans into a fictitious common-time scan. The original MA-LIO estimator compensates
all sensor points to the latest scan end using IMU support, including 15 future
IMU samples. Deskewed clouds then share the center (sensor 0) LiDAR frame.
The initial partial scan is retained for IMU warmup even when the bag starts
mid-scan; points outside valid spline support are still omitted. Dropping that
warmup batch changes the original gravity/bias initialization.

lidar_to_imu is T_IMU_from_LiDAR, row-major. The default matrices are the TST
calibration, including tilted VLP16 and LS-C16. Near-orthogonal rounded matrices
are projected onto SO(3) once. Other sequences require their own calibration.
HDL32/VLP16 use measured relative `time`. This bag's LS-C16 has no point-time or
ring field: azimuth_rings explicitly estimates ring/sweep timing from geometry.
That fallback is not a measured timestamp. Change duration to 0.02 ONLY for an
actual 20ms scan; measured spans are never stretched to 100ms.

map: ENU when GT initial alignment is enabled. odom: local MA-LIO frame.
lidar_center: physical sensor 0 at the current batch reference time.
TF: map -> odom -> lidar_center. The graph never reinterprets raw sensor frames.
GT factors constrain the center-LiDAR position after the body-frame lever arm.
GT covariance in YAML is an explicit assumption in m^2, not a covariance column
read from UrbanNav GT. graph covariance gating and LIO covariance gating are
selectable. GT used by optimization makes the displayed ATE a constraint residual,
not an independent accuracy measurement. Timestamp matching never extrapolates.

Loops use spatial/time candidate gates, coarse point ICP and fine point-to-plane
ICP, overlap/RMSE/degeneracy gates, then robust g2o optimization. No place-recognition
descriptor is added. The pose graph corrects the exported map/path, not the MA-LIO
filter state or its internal local map. Smooth submap corrections are interpolated
onto every scan before final export; original points and intensities are retained.

ROS2 build and run (Linux with ROS2 Humble installed)
---------------------------------------------------
Keep build/install/log outside the dataset. Python developer tools, if needed,
belong in the user's uv environment. C++ libraries and ROS2 itself are system
dependencies; they cannot be installed as uv Python packages.

  source ~/Documents/InhouseBuild/.venv/bin/activate
  source /opt/ros/humble/setup.bash
  uv pip install --python "$VIRTUAL_ENV/bin/python" colcon-common-extensions

Install missing native dependencies with your ROS2/system package manager:
ament_cmake, rclcpp, sensor_msgs, nav_msgs, geometry_msgs, visualization_msgs,
std_srvs, tf2_ros, pcl_conversions, Eigen3, PCL, Boost and OpenMP.
For launch/RViz: launch_ros, launch_xml, rviz2. g2o is bundled.

From a workspace containing this directory as src/ma_slam:

  colcon build --packages-select ma_slam --cmake-args \
    -DCMAKE_BUILD_TYPE=Release -DMA_SLAM_LIDAR_COUNT=3 \
    -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
  source install/setup.bash
  colcon test --packages-select ma_slam --event-handlers console_direct+
  colcon test-result --verbose

The explicit system Python variables are only for ament's installed ROS build
tools. No Python interpreter is embedded in the SLAM executable.
Edit config/params.yaml absolute GT/output paths for the Linux host, then:

  ros2 launch ma_slam run.launch.xml params_file:=/absolute/path/to/params.yaml

Custom parameters are read-only after startup. Edit YAML and restart to change
settings; a ros2 param set call cannot silently change stored values without
reconfiguring the estimator.

Headless:
  ros2 launch ma_slam run.launch.xml params_file:=/absolute/path/to/params.yaml rviz:=false

Two sensors require a separate -DMA_SLAM_LIDAR_COUNT=2 build, sensors.count=2,
and estimator.point_filter_num=[1, 1]. Sensor 0 must remain the reference LiDAR.
The upstream filter manifold is compiled for 2 or 3 sensors, not dynamically
resized by changing YAML alone. Use one MA-LIO instance per process.

ROS1 bag input
-------------
The existing UrbanNav .bag is ROS1. Convert only the required topics once, using
the existing uv environment's rosbags-convert CLI (not part of the SLAM runtime):

  source ~/Documents/InhouseBuild/.venv/bin/activate
  rosbags-convert --src /absolute/path/UrbanNav-HK_TST-20210517_sensors.bag \
    --dst /absolute/path/UrbanNav_TST_ros2 --dst-storage sqlite3 --dst-version 8 \
    --dst-typestore ros2_humble --include-topic /imu/data /velodyne_points \
    /right/velodyne_points /left/lslidar_point_cloud

Start the node first; then in a separate ROS2 terminal:
  ros2 bag play /absolute/path/UrbanNav_TST_ros2 --clock --rate 1.0

Do not loop playback into the same estimator. Reversed sensor clocks stop it.
Queue drops are reported; reduce playback rate if processing cannot keep up.
Final scans without enough future IMU support remain unprocessed and are reported.

Finish, compare and explicitly save
---------------------------------
After bag playback, finish without writing a map:
  ros2 service call /ma_slam/finish std_srvs/srv/Trigger '{}'

This drains complete queued batches, closes the final submap, optimizes, publishes
the final preview/path and prints GT ATE plus loop residual statistics. New sensor
input is then rejected until restart. Review the final RViz result first.

Save after review (also finalizes if /finish was not called):
  ros2 service call /ma_slam/save_map std_srvs/srv/Trigger '{}'

Default output.directory is a new Medium/Map/MA_SLAM subdirectory:
  MA_SLAM_map_Full.pcd       all accepted deskewed XYZI points, no export voxel filter
  MA_SLAM_map.pcd            globally voxelized XYZI; intensity is per-voxel mean
  MA_SLAM_map_trajectory.tum timestamp x y z qx qy qz qw, center LiDAR in map frame

Full means valid/accepted points: invalid, blind/strided and unsupported startup
points are omitted by preprocessing/deskew. No JSON, CSV or NPZ is generated.
Existing output files are refused unless output.overwrite=true is explicitly set.
Ctrl-C does not export by default (output.save_on_shutdown=false).
The dense scan spool is private temporary disk storage and is removed on normal
exit; it is required to reproject all points after final graph optimization.
Budget enough temporary disk for dense scans and export staging. Preview sampling
does not change saved full-map density.

ROS2 topics
-----------
/ma_slam/odometry         nav_msgs/Odometry, local MA-LIO (odom)
/ma_slam/path             nav_msgs/Path, optimized global trajectory (map)
/ma_slam/cloud_registered sensor_msgs/PointCloud2, latest compensated cloud (map)
/ma_slam/map              sensor_msgs/PointCloud2, bounded optimized preview (map)
/ma_slam/loop_constraints visualization_msgs/MarkerArray, accepted graph loops

Native core verification on this Mac (no ROS installation required)
-----------------------------------------------------------------
  source ~/Documents/InhouseBuild/.venv/bin/activate
  cd /Users/grey.haus/Documents/InhouseBuild/CrossModal_Localization/Test/SkyNet/Utils/Pre_Dataset/MA_SLAM
  cmake -S . -B /tmp/ma_slam_native_build -DMA_SLAM_WITH_ROS2=OFF \
    -DMA_SLAM_LIDAR_COUNT=3 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
  cmake --build /tmp/ma_slam_native_build --parallel 6
  ctest --test-dir /tmp/ma_slam_native_build --output-on-failure

Native tests exercise input/timing, GT interpolation, moving multi-LiDAR deskew,
covariance permutation, loop/GT graph behavior, and complete synthetic pipeline
export with original intensity and no-overwrite checks. Their map files are private
temporary fixtures only. ROS2 startup/TF/DDS and full-dataset map quality require
the actual ROS2 environment; native tests do not establish those properties.

Verification performed on 2026-09-30:
- Native CMake Release builds: both 2- and 3-LiDAR configurations, 5/5 CTest passes.
- Actual bag: first 3 scans from each sensor matched prior XYZ/intensity/point times.
- All 787 GT records matched prior conversion within 8.62e-10 m.
- First 30 synchronized actual batches produced 28 supported poses. Given identical
  exact relative timestamps, all 28 frontend poses matched the previous native
  MA-LIO core bit-for-bit. The new int64 epoch subtraction avoids old float64
  absolute-nanosecond quantization; outputs need not match the old Python runner
  bit-for-bit when it supplies differently rounded timestamps.
- The corrected previous backend and new graph accepted the same synthetic loop;
  maximum pose-matrix difference was 1.22e-6 (float vs double boundary inputs).
- No real-dataset PCD or trajectory was exported or overwritten during validation.
- ROS2 node compilation and DDS/launch/service execution remain unverified on this
  Mac because ROS2 is not installed. The actual ROS2 smoke test is included for
  colcon test in the target environment; syntax/header review is not a runtime test.

References
----------
MA-LIO: https://github.com/minwoo0611/MA-LIO
LIO-SAM source organization reference: https://github.com/TixiaoShan/LIO-SAM/tree/ros2
g2o: https://github.com/RainerKuemmerle/g2o
