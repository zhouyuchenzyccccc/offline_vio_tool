#include <System.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

struct ImageItem {
    double t;
    std::string rgb;
    std::string depth;
};

struct ImuItem {
    double t;
    float ax;
    float ay;
    float az;
    float gx;
    float gy;
    float gz;
};

static bool readImageList(const std::string &path, std::vector<ImageItem> &out) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Cannot open image association file: " << path << std::endl;
        return false;
    }

    out.clear();
    while (!fin.eof()) {
        ImageItem item;
        fin >> item.t >> item.rgb >> item.depth;
        if (fin.fail()) {
            break;
        }
        out.push_back(item);
    }
    std::sort(out.begin(), out.end(), [](const ImageItem &a, const ImageItem &b) { return a.t < b.t; });
    return !out.empty();
}

static bool readImu(const std::string &path, std::vector<ImuItem> &out) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Cannot open imu file: " << path << std::endl;
        return false;
    }

    out.clear();
    while (!fin.eof()) {
        ImuItem it;
        fin >> it.t >> it.ax >> it.ay >> it.az >> it.gx >> it.gy >> it.gz;
        if (fin.fail()) {
            break;
        }
        out.push_back(it);
    }
    std::sort(out.begin(), out.end(), [](const ImuItem &a, const ImuItem &b) { return a.t < b.t; });
    return !out.empty();
}

int main(int argc, char **argv) {
    if (argc < 6) {
        std::cerr << "Usage: rgbd_inertial_offline <vocab> <settings> <assoc_txt> <imu_txt> <traj_out> [--no-viewer] [--depth-scale 1000]" << std::endl;
        return 1;
    }

    std::string vocab = argv[1];
    std::string settings = argv[2];
    std::string assoc_txt = argv[3];
    std::string imu_txt = argv[4];
    std::string traj_out = argv[5];

    bool use_viewer = true;
    double depth_scale = 1000.0;

    for (int i = 6; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-viewer") {
            use_viewer = false;
        } else if (a == "--depth-scale") {
            if (i + 1 >= argc) {
                std::cerr << "--depth-scale requires a value" << std::endl;
                return 1;
            }
            depth_scale = std::stod(argv[++i]);
            if (depth_scale <= 0.0) {
                std::cerr << "depth_scale must be positive" << std::endl;
                return 1;
            }
        }
    }

    std::vector<ImageItem> images;
    std::vector<ImuItem> imu;

    if (!readImageList(assoc_txt, images)) {
        return 2;
    }
    if (!readImu(imu_txt, imu)) {
        return 3;
    }

    ORB_SLAM3::System slam(vocab, settings, ORB_SLAM3::System::IMU_RGBD, use_viewer);

    size_t imu_idx = 0;
    double last_t = -1.0;

    for (size_t i = 0; i < images.size(); ++i) {
        const ImageItem &frm = images[i];

        cv::Mat rgb = cv::imread(frm.rgb, cv::IMREAD_UNCHANGED);
        cv::Mat depth_raw = cv::imread(frm.depth, cv::IMREAD_UNCHANGED);

        if (rgb.empty() || depth_raw.empty()) {
            std::cerr << "Skip frame due to invalid image: " << frm.rgb << " or " << frm.depth << std::endl;
            continue;
        }

        if (rgb.channels() == 4) {
            cv::cvtColor(rgb, rgb, cv::COLOR_BGRA2BGR);
        }

        cv::Mat depth;
        if (depth_raw.type() == CV_16U) {
            depth_raw.convertTo(depth, CV_32F, 1.0 / depth_scale);
        } else if (depth_raw.type() == CV_32F) {
            depth = depth_raw;
        } else {
            depth_raw.convertTo(depth, CV_32F);
        }

        std::vector<ORB_SLAM3::IMU::Point> vImu;
        while (imu_idx < imu.size() && imu[imu_idx].t <= frm.t) {
            if (imu[imu_idx].t > last_t) {
                const ImuItem &m = imu[imu_idx];
                vImu.emplace_back(m.ax, m.ay, m.az, m.gx, m.gy, m.gz, m.t);
            }
            ++imu_idx;
        }

        slam.TrackRGBD(rgb, depth, frm.t, vImu);
        last_t = frm.t;

        if ((i + 1) % 100 == 0) {
            std::cout << "Processed frames: " << (i + 1) << "/" << images.size() << std::endl;
        }
    }

    slam.Shutdown();
    slam.SaveTrajectoryTUM(traj_out);

    std::cout << "Trajectory saved: " << traj_out << std::endl;
    return 0;
}
