#include "ma_slam/mapStorage.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace ma_slam {
namespace {
struct Record { float x, y, z, intensity; };
static_assert(sizeof(Record) == 16);

struct TemporaryDirectory {
    std::filesystem::path path;
    explicit TemporaryDirectory(const std::filesystem::path& parent) {
        auto pattern = (parent / "ma_slam-XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("Cannot create private temporary directory.");
        path = pattern;
    }
    ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

void header(std::ostream& out, std::uint64_t points) {
    out << "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        << "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH " << points
        << "\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS " << points << "\nDATA binary\n";
}

struct Voxel {
    std::int64_t x, y, z;
    bool operator==(const Voxel& other) const { return x == other.x && y == other.y && z == other.z; }
};
struct VoxelHash {
    std::size_t operator()(const Voxel& p) const {
        auto h = std::hash<std::int64_t>{}(p.x);
        for (const auto v : {p.y, p.z}) h ^= std::hash<std::int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
struct Accumulator { double x = 0, y = 0, z = 0, intensity = 0; std::uint64_t n = 0; };

bool rigid(const Matrix4& pose) {
    return pose.allFinite() && pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-8) &&
        (pose.topLeftCorner<3, 3>().transpose() * pose.topLeftCorner<3, 3>())
            .isApprox(Eigen::Matrix3d::Identity(), 1e-5) &&
        pose.topLeftCorner<3, 3>().determinant() > 0.0;
}

std::int64_t voxelIndex(float coordinate, double size) {
    const double index = std::floor(double(coordinate) / size);
    if (!std::isfinite(index) || index < double(std::numeric_limits<std::int64_t>::min()) ||
        index >= double(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Output point exceeds the voxel index range.");
    return static_cast<std::int64_t>(index);
}

std::int64_t tileIndex(std::int64_t voxel, std::int64_t width) {
    // - Integer floor division keeps every global voxel in one tile, including x/y < 0.
    return voxel / width - (voxel % width < 0 ? 1 : 0);
}
}

MapStorage::MapStorage() {
    // - Stage exact XYZI records outside the dataset, with bounded RAM.
    auto pattern = (std::filesystem::temp_directory_path() / "ma_slam-scans-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("Cannot create scan spool.");
    folder_ = pattern;
    stream_.exceptions(std::ios::failbit | std::ios::badbit);
    try { stream_.open(folder_ / "scans.xyzi", std::ios::binary); }
    catch (...) { std::filesystem::remove_all(folder_); throw; }
}

MapStorage::~MapStorage() {
    stream_.exceptions(std::ios::goodbit);
    stream_.close();
    std::error_code error;
    std::filesystem::remove_all(folder_, error);
}

void MapStorage::append(const Cloud& cloud) {
    // - Preserve measured intensity; no export-time denoising or point dropping.
    std::vector<Record> points;
    points.reserve(cloud.size());
    for (const auto& p : cloud) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.intensity))
            throw std::invalid_argument("Non-finite deskewed XYZI point.");
        points.push_back({p.x, p.y, p.z, p.intensity});
    }
    stream_.write(reinterpret_cast<const char*>(points.data()), points.size() * sizeof(Record));
    counts_.push_back(points.size());
}

SavedMap MapStorage::save(const OutputConfig& config, const std::vector<std::int64_t>& times,
                          const std::vector<Matrix4>& poses) {
    // - Transform each original scan by its final optimized pose.
    // - Write full XYZI and globally voxelized XYZI (centroid/mean intensity).
    // - Tile voxel accumulation to avoid a whole-city unordered_map in RAM.
    // - Publish files only after every staged payload has been completed.
    if (poses.size() != counts_.size() || times.size() != poses.size() || poses.empty())
        throw std::invalid_argument("Map poses/timestamps do not match stored scans.");
    if (config.directory.empty() || config.prefix.empty() ||
        std::filesystem::path(config.prefix).filename() != config.prefix ||
        !std::isfinite(config.voxel_size) || config.voxel_size <= 0 ||
        !std::isfinite(config.tile_size) || config.tile_size <= 0)
        throw std::invalid_argument("Invalid output directory/prefix/voxel settings.");
    if (config.save_scans && (config.scans_directory.empty() || config.scans_directory == "." || config.scans_directory == ".." ||
        std::filesystem::path(config.scans_directory).filename() != config.scans_directory))
        throw std::invalid_argument("output.scans_directory must be a plain folder name.");
    const double tile_voxels = std::max(1., std::ceil(config.tile_size / config.voxel_size));
    if (!std::isfinite(tile_voxels) || tile_voxels >= double(std::numeric_limits<std::int64_t>::max()))
        throw std::invalid_argument("Output tile contains too many voxel indices.");
    for (std::size_t i = 0; i < poses.size(); ++i) {
        if (!rigid(poses[i])) throw std::invalid_argument("Output pose is not a finite SE(3) transform.");
        if (i > 0 && times[i] <= times[i - 1]) throw std::invalid_argument("Output times must increase.");
    }
    const std::filesystem::path output(config.directory);
    std::filesystem::create_directories(output);
    SavedMap result;
    if (config.save_full) result.files.push_back(output / (config.prefix + "_Full.pcd"));
    if (config.save_voxel) result.files.push_back(output / (config.prefix + ".pcd"));
    if (config.save_trajectory) result.files.push_back(output / (config.prefix + "_trajectory.tum"));
    if (config.save_scans) result.files.push_back(output / config.scans_directory / "scans.xyzi");
    for (const auto& file : result.files)
        if (!config.overwrite && std::filesystem::symlink_status(file).type() != std::filesystem::file_type::not_found)
            throw std::runtime_error("Output exists; set output.overwrite explicitly: " + file.string());
    if (config.save_scans) std::filesystem::create_directories(output / config.scans_directory);
    TemporaryDirectory stage(output);
    stream_.flush();
    std::ifstream source(folder_ / "scans.xyzi", std::ios::binary);
    source.exceptions(std::ios::failbit | std::ios::badbit);
    const auto total = std::accumulate(counts_.begin(), counts_.end(), std::uint64_t{0});
    std::ofstream full;
    if (config.save_full) {
        full.exceptions(std::ios::failbit | std::ios::badbit);
        full.open(stage.path / (config.prefix + "_Full.pcd"), std::ios::binary);
        header(full, total);
    }
    using Tile = std::pair<std::int64_t, std::int64_t>;
    std::map<Tile, std::filesystem::path> tiles;
    const auto tile_width = static_cast<std::int64_t>(tile_voxels);
    for (std::size_t frame = 0; frame < poses.size(); ++frame) {
        std::vector<Record> points(counts_[frame]);
        source.read(reinterpret_cast<char*>(points.data()), points.size() * sizeof(Record));
        std::map<Tile, std::vector<Record>> chunks;
        for (auto& p : points) {
            const Eigen::Vector3d world = poses[frame].topLeftCorner<3, 3>() * Eigen::Vector3d(p.x, p.y, p.z)
                                        + poses[frame].topRightCorner<3, 1>();
            p.x = world.x(); p.y = world.y(); p.z = world.z();
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                throw std::invalid_argument("Output point exceeds the PCD float32 coordinate range.");
            if (config.save_voxel) {
                const Tile tile{tileIndex(voxelIndex(p.x, config.voxel_size), tile_width),
                                tileIndex(voxelIndex(p.y, config.voxel_size), tile_width)};
                chunks[tile].push_back(p);
            }
        }
        if (config.save_full) full.write(reinterpret_cast<const char*>(points.data()), points.size() * sizeof(Record));
        for (const auto& [tile, records] : chunks) {
            const auto path = stage.path / ("tile_" + std::to_string(tile.first) + "_" + std::to_string(tile.second));
            tiles[tile] = path;
            std::ofstream stream(path, std::ios::binary | std::ios::app);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
            stream.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(Record));
        }
    }
    if (config.save_scans) {
        // - The spool already holds the exact scans: move it (no extra disk copy) once it has been read.
        // - Different filesystems cannot rename; then copy and let the destructor drop the original.
        source.close();
        const auto spool = folder_ / "scans.xyzi";
        const auto staged = stage.path / "scans.xyzi";
        try { std::filesystem::rename(spool, staged); }
        catch (const std::filesystem::filesystem_error&) { std::filesystem::copy_file(spool, staged); }
        if (std::filesystem::file_size(staged) != total * sizeof(Record))
            throw std::runtime_error("Scan archive size does not match the stored scan counts.");
        result.scan_counts = counts_;
    }
    if (config.save_full) { full.close(); result.full_points = total; }
    if (config.save_voxel) {
        const auto payload_path = stage.path / "voxel_payload";
        std::ofstream payload(payload_path, std::ios::binary);
        payload.exceptions(std::ios::failbit | std::ios::badbit);
        for (const auto& [tile, path] : tiles) {
            std::unordered_map<Voxel, Accumulator, VoxelHash> voxels;
            std::ifstream tile_stream(path, std::ios::binary);
            Record p;
            while (tile_stream.read(reinterpret_cast<char*>(&p), sizeof(p))) {
                const Voxel key{voxelIndex(p.x, config.voxel_size), voxelIndex(p.y, config.voxel_size),
                                voxelIndex(p.z, config.voxel_size)};
                auto& value = voxels[key];
                value.x += p.x; value.y += p.y; value.z += p.z; value.intensity += p.intensity; ++value.n;
            }
            if (!tile_stream.eof() || tile_stream.gcount() != 0) throw std::runtime_error("Truncated voxel tile.");
            for (const auto& [key, v] : voxels) {
                const Record centroid{float(v.x / v.n), float(v.y / v.n), float(v.z / v.n), float(v.intensity / v.n)};
                payload.write(reinterpret_cast<const char*>(&centroid), sizeof(centroid));
            }
            result.voxel_points += voxels.size();
            std::filesystem::remove(path);
        }
        payload.close();
        std::ifstream payload_source(payload_path, std::ios::binary);
        std::ofstream voxel(stage.path / (config.prefix + ".pcd"), std::ios::binary);
        voxel.exceptions(std::ios::failbit | std::ios::badbit);
        header(voxel, result.voxel_points);
        if (result.voxel_points) voxel << payload_source.rdbuf();
        voxel.close();
    }
    if (config.save_trajectory) {
        std::ofstream tum(stage.path / (config.prefix + "_trajectory.tum"));
        tum.exceptions(std::ios::failbit | std::ios::badbit);
        tum << "# timestamp_s x y z qx qy qz qw; map frame, center LiDAR origin\n" << std::fixed << std::setprecision(9);
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const Eigen::Quaterniond q(poses[i].topLeftCorner<3, 3>());
            // - Preserve epoch nanoseconds exactly rather than rounding a ~1e9-second double.
            if (times[i] < 0) tum << '-';
            tum << std::abs(times[i] / 1000000000LL) << '.' << std::setfill('0') << std::setw(9)
                << std::abs(times[i] % 1000000000LL) << std::setfill(' ')
                << ' ' << poses[i](0, 3) << ' ' << poses[i](1, 3) << ' ' << poses[i](2, 3)
                << ' ' << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << '\n';
        }
        tum.close();
    }
    std::vector<std::filesystem::path> published;
    try {
        for (const auto& file : result.files) {
            const auto staged = stage.path / file.filename();
            if (config.overwrite) std::filesystem::rename(staged, file);
            else {
                // - Staging is on the same filesystem. A hard link publishes atomically
                //   and fails if any output appeared after the preflight check.
                std::filesystem::create_hard_link(staged, file);
                published.push_back(file);
            }
        }
    } catch (...) {
        for (const auto& file : published) {
            std::error_code error;
            const auto staged = stage.path / file.filename();
            if (std::filesystem::equivalent(staged, file, error) && !error)
                std::filesystem::remove(file, error);
        }
        throw;
    }
    return result;
}

std::vector<Matrix4> interpolateCorrections(const std::vector<Matrix4>& raw,
                                          const std::vector<Matrix4>& graph,
                                          const std::vector<std::size_t>& anchors) {
    // - Spread graph corrections continuously over traveled distance.
    // - Retain local odometry detail and preserve every optimized anchor exactly.
    if (raw.size() != graph.size() || anchors.empty() || anchors.front() != 0 || anchors.back() >= raw.size())
        throw std::invalid_argument("Invalid trajectory correction anchors.");
    for (std::size_t i = 0; i < raw.size(); ++i)
        if (!rigid(raw[i]) || !rigid(graph[i])) throw std::invalid_argument("Invalid SE(3) correction pose.");
    auto result = graph;
    for (std::size_t k = 1; k < anchors.size(); ++k) {
        const auto first = anchors[k - 1], last = anchors[k];
        if (last <= first) throw std::invalid_argument("Submap anchors must increase.");
        const Matrix4 inverse = raw[first].inverse();
        const Matrix4 local_end = inverse * raw[last];
        const Matrix4 correction = graph[first].inverse() * graph[last] * local_end.inverse();
        const Eigen::Quaterniond rotation(correction.topLeftCorner<3, 3>());
        std::vector<double> distance(last - first + 1, 0.);
        for (std::size_t i = first + 1; i <= last; ++i)
            distance[i - first] = distance[i - first - 1] + (raw[i].topRightCorner<3, 1>() - raw[i - 1].topRightCorner<3, 1>()).norm();
        for (std::size_t i = first; i <= last; ++i) {
            const double alpha = distance.back() > 1e-6 ? distance[i - first] / distance.back() : double(i - first) / (last - first);
            Matrix4 blend = Matrix4::Identity();
            blend.topLeftCorner<3, 3>() = Eigen::Quaterniond::Identity().slerp(alpha, rotation).toRotationMatrix();
            blend.topRightCorner<3, 1>() = alpha * correction.topRightCorner<3, 1>();
            result[i] = graph[first] * blend * inverse * raw[i];
        }
    }
    for (const auto index : anchors) result[index] = graph[index];
    return result;
}

}  // namespace ma_slam
