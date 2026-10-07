cmake_minimum_required(VERSION 3.20)
project(ma_slam VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(MA_SLAM_WITH_ROS2 "Build the ROS 2 ament package" ON)
# The notebook pybind adapter lives in ../MA_SLAM_ADAPTER and links ma_slam_pipeline from here.
set(MA_SLAM_LIDAR_COUNT 3 CACHE STRING "MA-LIO manifold sensor count: 2 or 3")
if(NOT MA_SLAM_LIDAR_COUNT MATCHES "^[23]$")
  message(FATAL_ERROR "MA_SLAM_LIDAR_COUNT must be 2 or 3")
endif()
# OpenMP threads for per-point frontend loops; outputs are bit-identical for any count >= 2.
# 1 is rejected: omp_set_num_threads(MP_PROC_NUM) also drives Eigen's OpenMP GEMM
# (HTH = HT * h_x_, esekfom.hpp), which runs serially at 1 thread with a different depth
# blocking (kc) and would change IEKF results bitwise.
set(MA_SLAM_OMP_THREADS 6 CACHE STRING "OpenMP threads for MA-LIO per-point loops (>= 2)")
if(NOT MA_SLAM_OMP_THREADS MATCHES "^([2-9]|[1-9][0-9]+)$")
  message(FATAL_ERROR "MA_SLAM_OMP_THREADS must be an integer >= 2")
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL 1.12 REQUIRED COMPONENTS common filters features registration kdtree)
find_package(Threads REQUIRED)
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE)
  find_path(MA_SLAM_OMP_INCLUDE omp.h HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include REQUIRED)
  find_library(MA_SLAM_OMP_LIBRARY omp HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib REQUIRED)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
  set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${MA_SLAM_OMP_INCLUDE}"
    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
    INTERFACE_LINK_LIBRARIES "${MA_SLAM_OMP_LIBRARY}")
elseif(NOT OpenMP_CXX_FOUND)
  message(FATAL_ERROR "OpenMP is required for MA-LIO")
endif()

set(BUILD_SHARED_LIBS OFF)
foreach(option G2O_BUILD_APPS G2O_BUILD_EXAMPLES G2O_USE_OPENGL G2O_USE_OPENMP
    G2O_USE_CHOLMOD G2O_USE_CSPARSE G2O_BUILD_SLAM2D_TYPES G2O_BUILD_SLAM2D_ADDON_TYPES
    G2O_BUILD_SBA_TYPES G2O_BUILD_SIM3_TYPES G2O_BUILD_ICP_TYPES G2O_BUILD_SCLAM2D_TYPES)
  set(${option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory(third_party/g2o EXCLUDE_FROM_ALL)

add_library(ma_slam_common INTERFACE)
target_include_directories(ma_slam_common INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS})
target_link_libraries(ma_slam_common INTERFACE Eigen3::Eigen ${PCL_LIBRARIES} Threads::Threads)
target_compile_definitions(ma_slam_common INTERFACE MA_SLAM_LIDAR_COUNT=${MA_SLAM_LIDAR_COUNT})

add_library(ma_slam_frontend src/imuPreintegration.cpp
  third_party/ma_lio/src/BsplineSE3.cpp third_party/ma_lio/include/ikd-Tree/ikd_Tree.cpp)
target_include_directories(ma_slam_frontend PRIVATE third_party/ma_lio/include)
# Deterministic ikd-tree rebuild: identical results independent of thread scheduling (see ikd_Tree.cpp).
target_compile_definitions(ma_slam_frontend PRIVATE MP_EN MP_PROC_NUM=${MA_SLAM_OMP_THREADS}
  MA_SLAM_DETERMINISTIC_REBUILD)
target_link_libraries(ma_slam_frontend PUBLIC ma_slam_common PRIVATE OpenMP::OpenMP_CXX)

add_library(ma_slam_graph src/mapOptimization.cpp)
target_link_libraries(ma_slam_graph PUBLIC ma_slam_common
  PRIVATE g2o::core g2o::types_slam3d g2o::solver_eigen)
add_library(ma_slam_input src/imageProjection.cpp src/groundTruth.cpp)
target_link_libraries(ma_slam_input PUBLIC ma_slam_common)
add_library(ma_slam_storage src/mapStorage.cpp)
target_link_libraries(ma_slam_storage PUBLIC ma_slam_common)
add_library(ma_slam_pipeline src/pipeline.cpp src/parameters.cpp)
target_link_libraries(ma_slam_pipeline PUBLIC ma_slam_frontend ma_slam_graph ma_slam_input ma_slam_storage)

include(CTest)
if(BUILD_TESTING)
  foreach(test input frontend mapOptimization mapStorage pipeline)
    add_executable(test_${test} tests/test_${test}.cpp)
    target_link_libraries(test_${test} PRIVATE ma_slam_pipeline)
    target_compile_options(test_${test} PRIVATE -UNDEBUG)
    add_test(NAME ${test} COMMAND test_${test})
  endforeach()
endif()

if(MA_SLAM_WITH_ROS2)
  find_package(ament_cmake REQUIRED)
  foreach(package rclcpp rcl_interfaces sensor_msgs nav_msgs geometry_msgs std_srvs tf2_ros pcl_conversions visualization_msgs)
    find_package(${package} REQUIRED)
  endforeach()
  add_executable(ma_slam_node src/maSlamNode.cpp)
  target_link_libraries(ma_slam_node PRIVATE ma_slam_pipeline)
  ament_target_dependencies(ma_slam_node rclcpp rcl_interfaces sensor_msgs nav_msgs geometry_msgs
    std_srvs tf2_ros pcl_conversions visualization_msgs)
  if(BUILD_TESTING)
    add_test(NAME ros2_services COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ros2_services.sh
      $<TARGET_FILE:ma_slam_node> ${CMAKE_CURRENT_SOURCE_DIR}/config/params.yaml ${MA_SLAM_LIDAR_COUNT})
    set_tests_properties(ros2_services PROPERTIES TIMEOUT 60)
  endif()
  install(TARGETS ma_slam_node DESTINATION lib/${PROJECT_NAME})
  install(DIRECTORY config launch rviz DESTINATION share/${PROJECT_NAME})
  install(FILES README.md LICENSE NOTICE.txt DESTINATION share/${PROJECT_NAME})
  install(FILES third_party/ma_lio/PROVENANCE.txt third_party/ma_lio/LICENSE
    DESTINATION share/${PROJECT_NAME}/licenses/ma_lio)
  install(FILES third_party/g2o/doc/license-bsd.txt third_party/g2o/doc/license-gpl.txt
    third_party/g2o/doc/license-lgpl.txt DESTINATION share/${PROJECT_NAME}/licenses/g2o)
  ament_package()
endif()
