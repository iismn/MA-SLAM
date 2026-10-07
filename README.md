# MA-SLAM

**Multi-LiDAR–Inertial SLAM with a MA-LIO frontend and a g2o pose-graph backend.**

MA-SLAM integrates the C++ estimator from [MA-LIO](https://github.com/minwoo0611/MA-LIO) with ROS 2, geometric loop closing, optional ground-truth position constraints, and dense map export. The current configuration targets the UrbanNav HK TST sequence with HDL-32E, tilted VLP-16, LS-C16, and SPAN IMU measurements.

The frontend retains MA-LIO's IMU initialization, continuous-time B-spline deskew, iterated error-state Kalman filter, and point-wise uncertainty processing. Integration changes and numerical fixes are recorded in [PROVENANCE.txt](third_party/ma_lio/PROVENANCE.txt). This is a derived integration, not the authors' unchanged MA-LIO release.

[Upstream MA-LIO](https://github.com/minwoo0611/MA-LIO) · [MA-LIO paper](https://arxiv.org/abs/2305.16792) · [Parameters](config/params.yaml) · [Source notices](NOTICE.txt)

## Features

- **Multiple LiDAR inputs:** two- or three-sensor builds; each scan retains its own point times and extrinsics.
- **C++ processing:** ROS 2 `PointCloud2` and `Imu` inputs feed the MA-LIO core directly.
- **Loop closing:** spatial/time candidate selection, coarse ICP, fine point-to-plane ICP, geometric acceptance checks, and robust g2o optimization.
- **GT constraints:** initial ENU alignment and optional position factors gated by graph or frontend covariance.
- **Dense XYZI maps:** export accepted deskewed points with intensity, alongside a globally voxelized map and optimized trajectory.
- **YAML configuration:** sensor calibration, timing, estimator, graph, visualization, and output settings in one file.
- **RViz visualization:** registered clouds, optimized trajectory, bounded map preview, and accepted loop constraints.

The backend adjusts the global trajectory and exported map. It does **not** feed graph corrections back into the frontend filter or its internal local map. Loop candidates currently use spatial proximity, without a place-recognition descriptor.

## 1. Prerequisites

### ROS 2 application

The target environment is **Ubuntu 22.04 / ROS 2 Humble**, using `ament_cmake` and a C++20 compiler.

Native dependencies include Eigen3, PCL 1.12 or newer, Boost, OpenMP, and the ROS packages declared in [package.xml](package.xml). The required g2o sources are included in `third_party/g2o`; no RKO package is required.

**Validation status:** native two- and three-LiDAR builds and regression tests have passed on macOS. The ROS 2 node, DDS transport, launch, and service execution have not yet been verified in the target ROS 2 environment. Short UrbanNav replay checks are not a full-sequence accuracy benchmark.

### Python tooling

Python is optional for the C++ ROS 2 runtime. For notebook or bag-conversion tools, use a `uv` environment. In the existing InhouseBuild workspace:

```bash
source ~/Documents/InhouseBuild/.venv/bin/activate
```

ROS 2 and native C++ libraries are system dependencies; install them through the relevant system package manager, independently of the Python environment.

## 2. Build Package

### 2.1. ROS 2 workspace

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/ma_slam_ws/src
cd ~/ma_slam_ws/src
git clone https://github.com/iismn/MA-SLAM.git ma_slam
cd ~/ma_slam_ws

# Requires rosdep to be installed and initialized.
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y

colcon build --packages-select ma_slam --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DMA_SLAM_LIDAR_COUNT=3 \
  -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

If `colcon` is missing, install its Python tooling through your activated environment:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" colcon-common-extensions
```

The explicit system Python paths in the build command are for ROS's installed ament tooling. The SLAM executable does not embed Python. Keep `build/`, `install/`, and `log/` outside dataset directories.

### 2.2. Native core without ROS 2

With the native dependencies available, build and test the C++ core independently:

```bash
cd /path/to/MA-SLAM
cmake -S . -B build/native \
  -DMA_SLAM_WITH_ROS2=OFF \
  -DMA_SLAM_LIDAR_COUNT=3 \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON
cmake --build build/native --parallel 4
ctest --test-dir build/native --output-on-failure
```

On Apple Silicon, CMake also checks `/opt/homebrew/opt/libomp` for OpenMP. Native tests do not exercise ROS 2 communication or launch behavior.

## 3. Configure MA-SLAM

Edit [config/params.yaml](config/params.yaml) before launching. The bundled file contains **local development paths** for `gt.file` and `output.directory`; replace these with paths on your machine.

### 3.1. Sensor inputs

| Index | Sensor | Default topic | Point timing |
| --- | --- | --- | --- |
| 0 | HDL-32E, reference LiDAR | `/velodyne_points` | Measured `time` field |
| 1 | Tilted VLP-16 | `/right/velodyne_points` | Measured `time` field |
| 2 | LS-C16 | `/left/lslidar_point_cloud` | Explicit `azimuth_rings` estimate |
| — | SPAN IMU | `/imu/data` | Message timestamp with configurable offset |

- `sensors.lidarN.lidar_to_imu` is a row-major **4 × 4 transform from LiDAR to IMU**, `T_IMU_from_LiDAR`.
- Each LiDAR has independent timestamp units, header convention, duration, and clock offset.
- Scan pairing uses `synchronization.pair_tolerance_seconds`; deskew preserves individual acquisition times and compensates points to the latest scan end.
- The original B-spline deskew requires future IMU support. The last scans of a bag can remain unprocessed if that support is unavailable.
- The supplied calibration is for the TST sequence. Other sequences require their own sensor calibration.

**LS-C16 timing:** the inspected TST bag lacks measured LS-C16 point-time and ring fields. Its configured fallback infers ring/sweep timing from geometry; these are not hardware timestamps. For another bag with measured point times, configure `timing_mode: measured`, the field name, and units. If a scan actually spans 20 ms, use `scan_duration_seconds: 0.02`; do not stretch measured 20 ms timing to 100 ms.

### 3.2. Two-LiDAR configuration

Build separately with `-DMA_SLAM_LIDAR_COUNT=2`, then set:

```yaml
sensors:
  count: 2
estimator:
  point_filter_num: [1, 1]
```

Apply these values under `ros__parameters` in the existing YAML. `lidar0` and `lidar1` are used; `lidar2` is ignored. Sensor 0 remains the reference. Changing YAML alone cannot resize the compiled filter manifold. Use one estimator instance per process.

### 3.3. Loop closing and GT factors

| Parameter | Purpose |
| --- | --- |
| `slam.enabled` | Enable the SLAM backend |
| `slam.loop_search_radius` | Spatial revisit search radius; `0` disables loop proposals |
| `slam.closure_min_time_separation` | Minimum time separation for a loop candidate |
| `slam.closure_overlap_threshold` | Minimum registration overlap |
| `slam.closure_max_rmse` | Maximum accepted loop registration RMSE |
| `gt.align_initial` | Align the initial trajectory to the GT ENU frame |
| `gt.enabled` | Add GT position constraints during optimization |
| `gt.covariance_source` | Use `graph` or `lio` covariance for factor gating |
| `gt.position_covariance` | Assumed ENU GT position covariance, in m² |
| `gt.pose_covariance_threshold` | Minimum estimated horizontal variance for a GT factor |
| `gt.covariance_ratio` | Required estimated-to-GT horizontal variance ratio |
| `gt.min_factor_distance` | Minimum spacing between GT factors |

With covariance gating enabled, the estimated horizontal variance must exceed both the absolute threshold and the configured multiple of GT variance. Factor insertion also requires valid GT time coverage and sufficient spacing.

The UrbanNav raw GT file does not supply the covariance used here: the YAML matrix is an explicit assumption. `gt.lidar_lever_arm` converts the GT reference position to the center-LiDAR origin. Review this calibration before comparing trajectories.

**ATE interpretation:** when GT participates in optimization, the reported ATE measures residual agreement with those constraints; it is not an independent accuracy estimate. GT timestamps are interpolated without extrapolation.

## 4. Run on UrbanNav

Dataset information is available from [UrbanNav](https://github.com/IPNL-POLYU/UrbanNavDataset). The bundled preset targets `UrbanNav-HK_TST-20210517_sensors.bag`.

### 4.1. Launch the node

After editing the YAML and sourcing the workspace:

```bash
ros2 launch ma_slam run.launch.xml \
  params_file:=/absolute/path/to/params.yaml
```

Add `rviz:=false` for headless operation. Parameters are read-only after startup; edit YAML and restart to change the estimator configuration.

### 4.2. Replay a ROS 1 bag through ROS 2

For ROS 2 playback, first convert the required topics using the Python environment:

```bash
uv pip install --python "$VIRTUAL_ENV/bin/python" rosbags
rosbags-convert \
  --src /absolute/path/UrbanNav-HK_TST-20210517_sensors.bag \
  --dst /absolute/path/UrbanNav_TST_ros2 \
  --dst-storage sqlite3 --dst-version 8 --dst-typestore ros2_humble \
  --include-topic /imu/data /velodyne_points \
  /right/velodyne_points /left/lslidar_point_cloud
```

With the node running, replay in another ROS 2 terminal:

```bash
ros2 bag play /absolute/path/UrbanNav_TST_ros2 --clock --rate 1.0
```

Reduce playback speed if queues drop messages. Do not loop playback into the same estimator: reversed sensor timestamps stop processing.

### 4.3. Finish and save

After playback, finalize and inspect the map **without saving**:

```bash
ros2 service call /ma_slam/finish std_srvs/srv/Trigger '{}'
```

This drains supported batches, closes the final submap, optimizes the graph, publishes the final preview, and reports trajectory/loop statistics. New sensor input is rejected after finalization.

To export the reviewed result:

```bash
ros2 service call /ma_slam/save_map std_srvs/srv/Trigger '{}'
```

Saving also finalizes the pipeline if needed. With the default prefix, the output is:

| File | Contents |
| --- | --- |
| `MA_SLAM_map_Full.pcd` | Accepted deskewed XYZI points without export voxel downsampling |
| `MA_SLAM_map.pcd` | Globally voxelized XYZI map; intensity is the per-voxel mean |
| `MA_SLAM_map_trajectory.tum` | `timestamp x y z qx qy qz qw`, center LiDAR in the map frame |

`output.voxel_size` controls the downsampled export, independently of estimator voxels and preview sampling. “Full” still excludes points rejected by preprocessing, stride/blind filters, or insufficient spline support.

Existing outputs are protected by `output.overwrite: false`. Ctrl-C does not save by default (`output.save_on_shutdown: false`). Temporary dense scan storage enables reprojection after graph optimization and is removed on normal exit; allow enough disk space for it and export staging.

### 4.4. ROS 2 visualization

| Topic | Message | Contents |
| --- | --- | --- |
| `/ma_slam/odometry` | `nav_msgs/Odometry` | Local MA-LIO estimate |
| `/ma_slam/path` | `nav_msgs/Path` | Optimized global trajectory |
| `/ma_slam/cloud_registered` | `sensor_msgs/PointCloud2` | Latest compensated cloud |
| `/ma_slam/map` | `sensor_msgs/PointCloud2` | Bounded optimized map preview |
| `/ma_slam/loop_constraints` | `visualization_msgs/MarkerArray` | Accepted graph loops |

The TF chain is `map → odom → lidar_center`. `map` is ENU when GT initial alignment is enabled; `odom` is the local frontend frame. Preview sampling does not reduce the saved full map.

## 5. Notebook Adapter (separate directory)

This package is the ROS 2 runtime: launch the node, receive topics, and process them with the C++ core. It contains no Python or pybind build.

Offline notebook runs use the sibling **`../MA_SLAM_ADAPTER`** directory. Its CMake project adds this directory with `MA_SLAM_WITH_ROS2=OFF` and links the same `ma_slam_pipeline` library into a pybind module. Estimator, graph, and export changes therefore apply to both the ROS 2 node and the notebook. See `../MA_SLAM_ADAPTER/README.md`.

## 6. Code Structure

```text
MA-SLAM/
├── config/params.yaml          # Runtime settings and calibration
├── include/ma_slam/            # C++ interfaces
├── launch/run.launch.xml       # ROS 2 launch
├── rviz/ma_slam.rviz           # Visualization configuration
├── src/
│   ├── maSlamNode.cpp          # Subscriptions, worker, publishers, TF, services
│   ├── pipeline.cpp            # End-to-end processing and finalization
│   ├── imageProjection.cpp     # Point decoding, timing, synchronization
│   ├── imuPreintegration.cpp   # MA-LIO initialization and deskew adapter
│   ├── lidarOdometry.inc       # MA-LIO planar registration and local map
│   ├── mapOptimization.cpp     # g2o graph, loop verification, GT factors
│   ├── groundTruth.cpp         # GT conversion, lever arm, interpolation, ATE
│   ├── mapStorage.cpp          # Dense storage and corrected XYZI export
│   └── parameters.cpp          # Shared parameter parsing and validation
├── tests/                     # Native regressions and ROS 2 service smoke test
└── third_party/
    ├── ma_lio/                # Retained upstream numerical components
    └── g2o/                   # Bundled graph optimizer
```

Source organization follows the separation of responsibilities used by [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM/tree/ros2). The estimation frontend remains MA-LIO.

## 7. Tests

For a ROS 2 workspace:

```bash
colcon test --packages-select ma_slam --event-handlers console_direct+
colcon test-result --verbose
```

The native tests cover input timing, GT interpolation, moving multi-LiDAR deskew, covariance ordering, graph constraints, and pipeline export with intensity and overwrite protection. Both sensor-count configurations passed five native tests in the recorded integration checks. Short real-bag comparisons also checked frontend equivalence; full-sequence map quality remains a separate evaluation.

The included ROS 2 service smoke test is intended for the target ROS environment. Passing native tests alone does not validate DDS, TF publication, or ROS launch behavior.

## 8. Acknowledgments and Citation

This project builds on [MA-LIO](https://github.com/minwoo0611/MA-LIO) and [g2o](https://github.com/RainerKuemmerle/g2o). Retained numerical components include IKFoM, ikd-Tree, and OpenVINS B-spline code. Original authorship and component notices are preserved in the source tree.

If your work uses the MA-LIO method, please cite the upstream paper:

```bibtex
@article{jung2023asynchronous,
  title={Asynchronous Multiple LiDAR-Inertial Odometry using Point-wise Inter-LiDAR Uncertainty Propagation},
  author={Jung, Minwoo and Jung, Sangwoo and Kim, Ayoung},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```

## License

See [LICENSE](LICENSE), [NOTICE.txt](NOTICE.txt), and the license headers of individual components. The retained source tree includes GPL-2.0, GPL-3.0-or-later, BSD, and other upstream component notices; the repository-level license text does not replace those notices.
