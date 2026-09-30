#pragma once
#include "ma_slam/types.hpp"
#include <filesystem>
#include <fstream>
#include <string>

namespace ma_slam {

struct OutputConfig {
    std::string directory;
    std::string prefix = "MA_SLAM_map";
    bool save_full = true, save_voxel = true, save_trajectory = true;
    bool overwrite = false, save_on_shutdown = false;
    double voxel_size = .05, tile_size = 20.;
};

struct SavedMap {
    std::uint64_t full_points = 0, voxel_points = 0;
    std::vector<std::filesystem::path> files;
};

// Owns only a private temporary scan spool. Destruction removes it on success/error.
class MapStorage {
public:
    MapStorage();
    ~MapStorage();
    MapStorage(const MapStorage&) = delete;
    MapStorage& operator=(const MapStorage&) = delete;
    void append(const Cloud& cloud);
    SavedMap save(const OutputConfig& config, const std::vector<std::int64_t>& times_ns,
                 const std::vector<Matrix4>& poses);
    std::size_t frames() const { return counts_.size(); }
private:
    std::filesystem::path folder_;
    std::ofstream stream_;
    std::vector<std::uint64_t> counts_;
};

std::vector<Matrix4> interpolateCorrections(const std::vector<Matrix4>& raw,
                                          const std::vector<Matrix4>& graph,
                                          const std::vector<std::size_t>& anchors);

}  // namespace ma_slam
