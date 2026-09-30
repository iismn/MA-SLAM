#!/usr/bin/env bash
# Run on a sourced ROS2 Linux environment. No sensor bag or map export is needed.
set -euo pipefail

test_node=${1:?Pass the ma_slam_node executable}
test_params=${2:?Pass config/params.yaml}
test_count=${3:-3}
case "$test_count" in
    2) test_strides='[1, 1]' ;;
    3) test_strides='[1, 1, 1]' ;;
    *) printf '%s\n' 'LiDAR count must be 2 or 3' >&2; exit 2 ;;
esac
test_dir=$(mktemp -d)
test_namespace="/ma_slam_smoke_$$"
test_pid=

cleanup() {
    local status=$?
    if [[ -n "$test_pid" ]]; then
        kill -INT "$test_pid" 2>/dev/null || true
        wait "$test_pid" || true
    fi
    if [[ $status -ne 0 ]]; then
        cat "$test_dir/node.log" "$test_dir/finish.log" "$test_dir/save.log" 2>/dev/null || true
    fi
    rm -rf -- "$test_dir"
    exit "$status"
}
trap cleanup EXIT

command -v ros2 >/dev/null
command -v timeout >/dev/null
"$test_node" --ros-args --params-file "$test_params" \
    -r "__ns:=$test_namespace" \
    -p "sensors.count:=$test_count" -p "estimator.point_filter_num:=$test_strides" \
    -p gt.enabled:=false -p gt.align_initial:=false -p gt.evaluate:=false \
    -p use_sim_time:=false -p output.save_on_shutdown:=false \
    -p output.overwrite:=false -p "output.directory:=$test_dir/output" \
    >"$test_dir/node.log" 2>&1 &
test_pid=$!

# ros2 service call waits for discovery; timeout prevents a silent launch failure
# or a broken worker/promise handoff from leaving the test blocked indefinitely.
timeout 30s ros2 service call "$test_namespace/ma_slam/finish" std_srvs/srv/Trigger '{}' \
    >"$test_dir/finish.log" 2>&1
grep -Eq 'success[=:][[:space:]]*(True|true)' "$test_dir/finish.log"

timeout 30s ros2 service call "$test_namespace/ma_slam/save_map" std_srvs/srv/Trigger '{}' \
    >"$test_dir/save.log" 2>&1
grep -Eq 'success[=:][[:space:]]*(False|false)' "$test_dir/save.log"
grep -q 'No registered scans' "$test_dir/save.log"
test ! -e "$test_dir/output"

printf '%s\n' 'PASS: ROS2 parameter startup, finish service, save rejection without data, and output protection.'
