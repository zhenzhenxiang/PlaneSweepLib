#include <string>
#include <boost/program_options.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <psl_base/exception.h>
#include <opencv2/opencv.hpp>
#include <psl_cudaBase/cudaFishEyeImageProcessor.h>
#include <psl_stereo/cudaFishEyePlaneSweep.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <psl_base/common.h>
#include <opencv2/ccalib/omnidir.hpp>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

using namespace std;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    cout << "Usage: ./pointCloudProjectionFromPly projectTestPly.txt" << endl;
    return 1;
  }

  // load data
  ifstream dataFile(argv[1]);
  if (!dataFile.is_open())
  {
    cout << "Cannot open file: " << argv[1] << endl;
    return 1;
  }

  string descpt;
  string imageFile, pointCloudFile;
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  double xi;
  double k1, k2, p1, p2;
  double roll, pitch, yaw, t_x, t_y, t_z;

  dataFile >> descpt >> imageFile;

  dataFile >> descpt >> K(0, 0) >> K(0, 1) >> K(0, 2) >> K(1, 1) >> K(1, 2);
  dataFile >> xi;
  dataFile >> k1 >> k2 >> p1 >> p2;

  dataFile >> descpt >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;

  dataFile >> descpt >> pointCloudFile;

  // -- set image
  cv::Mat image = cv::imread(imageFile);

  // -- set camera
  Eigen::Matrix3d R_cam2lidar;
  R_cam2lidar = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t_cam2lidar(t_x, t_y, t_z);

  Eigen::Matrix3d R_lidar2cam = R_cam2lidar.transpose();
  Eigen::Vector3d t_lidar2cam = -R_cam2lidar.transpose() * t_cam2lidar;

  PSL::FishEyeCameraMatrix<double> cam(K, R_lidar2cam, t_lidar2cam, xi);

  // -- load point cloud
  pcl::PLYReader pclReader;
  PointCloud inCloud;
  pclReader.read(pointCloudFile, inCloud);

  if (inCloud.size() == 0)
  {
    cout << "No points loaded!" << endl;
    return 1;
  }
  else
    cout << "Data loading finished. Total valid lidar points: "
         << inCloud.size() << endl;

  // undistort image
  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  cv::cvtColor(image, image, CV_BGR2GRAY);
  devImg.allocatePitchedAndUpload(image);
  cFEIP.setInputImg(devImg, cam);

  std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>> undistRes =
      cFEIP.undistort(1.0, 1.0, k1, k2, p1, p2);

  cv::Mat undistImage;
  undistRes.first.download(undistImage);

  cv::cvtColor(undistImage, undistImage, CV_GRAY2BGR);

  // project lidar points
  double minDist = 3.0;
  double maxDist = 100.0;
  for (int i = 0; i < inCloud.size(); i++)
  {
    Eigen::Vector3d p3Dlidar;
    p3Dlidar << inCloud.points[i].x, inCloud.points[i].y, inCloud.points[i].z;

    Eigen::Vector3d p3Dcam;
    p3Dcam = R_lidar2cam * p3Dlidar + t_lidar2cam;

    // ignore points with negative depth in camera frame
    if (p3Dcam(2) < 0)
      continue;

    // ignore points too close or too far
    double dist = sqrt(p3Dlidar(0) * p3Dlidar(0) + p3Dlidar(1) * p3Dlidar(1) +
                       p3Dlidar(2) * p3Dlidar(2));
    if (dist < minDist || dist > maxDist)
      continue;

    Eigen::Vector2d p2D;
    p2D = cam.projectPoint(p3Dlidar(0), p3Dlidar(1), p3Dlidar(2));

    // draw on image
    double minZ = 3.0;
    double maxZ = 60.0;
    double value = (p3Dcam(2) - minZ) / (maxZ - minZ) * 255;
    int color_r = value > 128 ? (value - 128) * 2 : 0;
    int color_g = value < 128 ? 2 * value : 255 - ((value - 128) * 2);
    int color_b = value < 128 ? 255 - (2 * value) : 0;
    cv::circle(undistImage, cv::Point(p2D(0), p2D(1)), 1,
               cv::Scalar(color_b, color_g, color_r), -1);
  }

  // show projection image
  cv::imshow("result", undistImage);
  cv::waitKey(0);

  return 0;
}
