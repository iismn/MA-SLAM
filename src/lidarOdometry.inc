// Original MA-LIO planar residual, uncertainty and local ikd-Tree map.
// Safety patches are documented in third_party/ma_lio/PROVENANCE.txt.
#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

struct ValueIndexPair
{
    double value;
    int index;
    bool operator<(const ValueIndexPair &other) const
    {
        return value < other.value;
    }
};

mutex mtx_buffer;
condition_variable sig_buffer;

float res_last[100000] = {0.0};
bool point_selected_surf[100000] = {0};
const float MOV_THRESHOLD = 1.5f;
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

std::vector<double> cov_plane;
std::vector<BoxPointType> cub_needrm;
std::vector<vector<int>> pointSearchInd_surf;
std::vector<PointVector> Nearest_Points;
std::vector<PointCloudXYZI::Ptr> feats_undistort_vec;
std::vector<PointCloudXYZI::Ptr> feats_down_vec;
std::vector<vector<Pose>> pose_unc;

std::vector<Eigen::Quaterniond> ext_q;
std::vector<V3D> ext_t;
std::vector<Pose> extrinsic;
std::vector<V3D> extrinsic_trans;
std::vector<Eigen::Quaterniond> extrinsic_quat;
std::vector<int> last_indices;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0, lidar_end_time = 0, first_lidar_time = 0.0;
deque<double> time_start_buffer;
deque<double> time_buffer;
deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<ma_lio_core::Imu::ConstPtr> imu_buffer;
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
int effct_feat_num = 0;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);

MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
V3D prev_pos;
vect3 pos_lid;


void pointBodyToWorld(const Matrix<float, 3, 1> &pi, Matrix<float, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (extrinsic_quat[0] * p_body + extrinsic_trans[0]) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void pointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global;
    int lid_idx = pi->intensity;
    if (lid_idx == 0)
        p_global = state_point.rot * (extrinsic_quat[lid_idx] * p_body + extrinsic_trans[lid_idx]) + state_point.pos;
    else
        p_global = state_point.rot * (kf.temporal_comp[lid_idx - 1].q_ * (extrinsic_quat[lid_idx] * p_body + extrinsic_trans[lid_idx]) + kf.temporal_comp[lid_idx - 1].t_) + state_point.pos;
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global;
    int lid_idx = pi->intensity;
    if (lid_idx == 0)
        p_global = state_point.rot * (extrinsic_quat[lid_idx] * p_body + extrinsic_trans[lid_idx]) + state_point.pos;
    else
        p_global = state_point.rot * (kf.temporal_comp[lid_idx - 1].q_ * (extrinsic_quat[lid_idx] * p_body + extrinsic_trans[lid_idx]) + kf.temporal_comp[lid_idx - 1].t_) + state_point.pos;

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->normal_y;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
}

void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}


void extrinsic_update()
{
    extrinsic.clear();
    extrinsic_trans.clear();
    extrinsic_quat.clear();

    extrinsic.resize(lid_num);
    extrinsic_trans.resize(lid_num);
    extrinsic_quat.resize(lid_num);
    for (int num = 0; num < lid_num; num++)
    {
        V3D extrinsic_t = *(static_cast<MTK::vect<3, double> *>(state_point.vect_state_ptr[1 + num]));
        Eigen::Quaterniond extrinsic_q = *(static_cast<MTK::SO3<double> *>(state_point.SO3_state_ptr[1 + num]));
        extrinsic_trans[num] = extrinsic_t;
        extrinsic_quat[num] = extrinsic_q;
        PoseInitial(extrinsic[num], extrinsic_trans[num], extrinsic_quat[num], kf.getExtrinsicUncertainty(num));
    }
}


void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        if ((feats_down_body->points[i].normal_y > cov_threshold))
            continue;
        
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));

        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
            PointToAdd.push_back(feats_down_world->points[i]);
    }

    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}


void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    laserCloudOri->resize(feats_down_size);
    corr_normvect->resize(feats_down_size);
    cov_plane.resize(feats_down_size);
    extrinsic_update();
    /*** Utilize the local planarity ***/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
    // Iterations are independent (per-index writes only): dynamic chunks balance mixed
    // performance/efficiency cores without changing any result.
    #pragma omp parallel for schedule(dynamic, 64)
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /*** transform to world frame ***/
        V3D p_body(point_body.x, point_body.y, point_body.z);
        int lid_idx = point_body.intensity;
        if (lid_idx != 0)
            p_body = extrinsic_quat[0].conjugate() * ((kf.temporal_comp[lid_idx - 1].q_ * (extrinsic_quat[lid_idx] * p_body + extrinsic_trans[lid_idx]) + kf.temporal_comp[lid_idx - 1].t_) - extrinsic_trans[0]);

        V3D p_global(s.rot * (extrinsic_quat[0] * p_body + extrinsic_trans[0]) + s.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points[i];
        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > max_neighbor_distance * max_neighbor_distance ? false : true;
        }

        if (!point_selected_surf[i])
            continue;

        VF(4) pabcd;
        double unit_cov;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, plane_th, unit_cov, cov_threshold))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.1 && std::abs(pd2) <= max_plane_residual)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                cov_plane[i] = unit_cov;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;
    double max_unit_cov = 0;
    double min_unit_cov = 1000;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            cov_plane[effct_feat_num] = cov_plane[i];
            if (cov_plane[effct_feat_num] > max_unit_cov)
                max_unit_cov = cov_plane[effct_feat_num];
            if (cov_plane[effct_feat_num] < min_unit_cov)
                min_unit_cov = cov_plane[effct_feat_num];

            effct_feat_num++;
        }
    }

    // return when no effective points
    if (effct_feat_num < 3) {
        ekfom_data.valid = false;
        std::fprintf(stderr, "MA-LIO: no effective points.\n");
        return;
    }

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, (1 + lid_num) * 6);
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R = MatrixXd::Zero(effct_feat_num, 1);

    double max_cov = 0;
    double min_cov = 9999;

    for (int i = 0; i < effct_feat_num; i++)
    {
        if (cov_plane[i] == 0)
            cov_plane[i] = 1;
        else if (max_unit_cov == min_unit_cov)
            cov_plane[i] = (plane_cov_max + plane_cov_min) / 2;
        else
            cov_plane[i] = 1 / ((plane_cov_max - plane_cov_min) * (cov_plane[i] - min_unit_cov) / (max_unit_cov - min_unit_cov) + plane_cov_min);

        M3D cov;
        PointType &laser_p = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        int lid_idx = laser_p.intensity;
        if (lid_idx != 0)
            point_this_be = extrinsic_quat[0].conjugate() * ((kf.temporal_comp[lid_idx - 1].q_ * (extrinsic_quat[lid_idx] * point_this_be + extrinsic_trans[lid_idx]) + kf.temporal_comp[lid_idx - 1].t_) - extrinsic_trans[0]);

        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = extrinsic_quat[0] * point_this_be + extrinsic_trans[0];
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);

        /*** get the normal vector ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        V3D B;
        ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A);

        if (extrinsic_est_en)
        {
            if (lid_idx == 0)
                B = point_be_crossmat * extrinsic_quat[0].conjugate() * C; 
            else
            {
                V3D point_ori(laser_p.x, laser_p.y, laser_p.z);
                point_be_crossmat << SKEW_SYM_MATRX(point_ori);
                C = kf.temporal_comp[lid_idx - 1].q_.conjugate() * C;
                B = point_be_crossmat * extrinsic_quat[lid_idx].conjugate() * C;
            }
            ekfom_data.h_x.block<1, 3>(i, 6 + 3 * lid_idx) << VEC_FROM_ARRAY(B);
            ekfom_data.h_x.block<1, 3>(i, 6 + 3 * (lid_num + lid_idx)) << VEC_FROM_ARRAY(C);
        }
        {
            int uncertain = int(laser_p.normal_x);
            if (uncertain >= pose_unc[lid_idx].size())
                uncertain = pose_unc[lid_idx].size() - 2;
            evalPointUncertainty(laser_p, cov, pose_unc[lid_idx][uncertain]);
            ekfom_data.R(i, 0) = cov(0, 0) + cov(1, 1) + cov(2, 2);
            laser_p.normal_y = cov(0, 0) + cov(1, 1) + cov(2, 2);
            if (max_cov < ekfom_data.R(i, 0))
                max_cov = ekfom_data.R(i, 0);
            if (min_cov > ekfom_data.R(i, 0))
                min_cov = ekfom_data.R(i, 0);
        }

        /*** Measuremnt residual value ***/
        ekfom_data.h(i) = (-1) * norm_p.intensity;
    }

    /*** Uncertainty Calculation using FIC ***/
    double cov_diff = max_cov - min_cov;
    for (int i = 0; i < effct_feat_num; i++)
    {
        ekfom_data.h_x.block<1, Eigen::Dynamic>(i, 0, 1, 6 * (lid_num + 1)) = ekfom_data.h_x.block<1, Eigen::Dynamic>(i, 0, 1, 6 * (lid_num + 1)) * cov_plane[i];
        ekfom_data.h(i) = ekfom_data.h(i) * cov_plane[i];
        if (max_cov - min_cov < 1e-12)
            ekfom_data.R(i, 0) = (point_cov_min + point_cov_max) / 2;
        else if (ekfom_data.R(i, 0) < min_cov + (max_cov - min_cov) * range_min)
            ekfom_data.R(i, 0) = point_cov_min;
        else if (ekfom_data.R(i, 0) > min_cov + (max_cov - min_cov) * range_max)
            ekfom_data.R(i, 0) = point_cov_max;
        else
            ekfom_data.R(i, 0) = (point_cov_max - point_cov_min) * (ekfom_data.R(i, 0) - (min_cov + (max_cov - min_cov) * range_min)) / ((range_max - range_min) * (max_cov - min_cov)) + point_cov_min;
    }

    /*** Uncertainty given to non-planar point ***/
    int k = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            feats_down_body->points[i].normal_y = laserCloudOri->points[k].normal_y;
            k++;
        }
        else
        {
            M3D cov;
            float which_lidar = feats_down_body->points[i].intensity;
            int imu_idx = int(feats_down_body->points[i].normal_x);
            if (imu_idx >= pose_unc[which_lidar].size() - 1)
                imu_idx = pose_unc[which_lidar].size() - 2;
            evalPointUncertainty(feats_down_body->points[i], cov, pose_unc[which_lidar][imu_idx]);
            feats_down_body->points[i].normal_y = cov(0, 0) + cov(1, 1) + cov(2, 2);
        }
    }
    /*** Localization weight calculation ***/
    Eigen::MatrixXd svd_mat = ekfom_data.h_x.topLeftCorner(effct_feat_num, 3);
    Eigen::JacobiSVD<MatrixXd> svd(svd_mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    MatrixXd singular_value = svd.singularValues();

    double weight = singular_value(2, 0) / singular_value(0, 0);

    if (weight > localize_thresh_max)
        weight = localize_cov_max;
    else if (weight < localize_thresh_min)
        weight = localize_cov_min;
    else
        weight = (localize_cov_max - localize_cov_min) * (weight - localize_thresh_min) / (localize_thresh_max - localize_thresh_min) + localize_cov_min;

    ekfom_data.h_x = ekfom_data.h_x * weight;
    ekfom_data.h = ekfom_data.h * weight;
}

