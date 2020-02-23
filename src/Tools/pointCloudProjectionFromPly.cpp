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
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace std;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;

void getDepthImage(PointCloud::Ptr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage);
void displayDepthImage(const cv::Mat_<double>& depthImage,
                       cv::Mat& undistImage);

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
  PointCloud::Ptr inCloud(new PointCloud());
  pclReader.read(pointCloudFile, *inCloud);

  if (inCloud->points.size() == 0)
  {
    cout << "No points loaded!" << endl;
    return 1;
  }
  else
    cout << "Data loading finished. Total valid lidar points: "
         << inCloud->points.size() << endl;

  // plane segmentation
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr planeInliners(new pcl::PointIndices);
  // Create the segmentation object
  pcl::SACSegmentation<Point> seg;
  // Optional
  seg.setOptimizeCoefficients(true);
  // Mandatory
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(0.3);

  seg.setInputCloud(inCloud);
  seg.segment(*planeInliners, *coefficients);

  if (planeInliners->indices.size() == 0)
  {
    PCL_ERROR("Could not estimate a planar model for the given dataset.");
    return (-1);
  }

  cout << "Plane coefficients: " << coefficients->values[0] << " "
       << coefficients->values[1] << " " << coefficients->values[2] << " "
       << coefficients->values[3] << endl;

  cout << "Plane inliers: " << planeInliners->indices.size() << endl;

  PointCloud::Ptr planeCloud(new PointCloud());
  for (int i = 0; i < planeInliners->indices.size(); i++)
  {
    planeCloud->push_back(inCloud->points[planeInliners->indices[i]]);
  }

  // undistort image
  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  cv::cvtColor(image, image, CV_BGR2GRAY);
  devImg.allocatePitchedAndUpload(image);
  cFEIP.setInputImg(devImg, cam);

  pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>> undistRes =
      cFEIP.undistort(1.0, 1.0, k1, k2, p1, p2);

  cv::Mat undistImage;
  undistRes.first.download(undistImage);

  cv::cvtColor(undistImage, undistImage, CV_GRAY2BGR);

  // get depth image
  double minDepth = 3.0;
  double maxDepth = 150.0;
  cv::Mat_<double> depthImage(image.rows, image.cols, -1.0);
  cv::Mat_<double> depthPlaneImage(image.rows, image.cols, -1.0);

  getDepthImage(inCloud, minDepth, maxDepth, cam, depthImage);
  getDepthImage(planeCloud, minDepth, maxDepth, cam, depthPlaneImage);

  // visualize depth image
  cv::Mat allOnImage = undistImage.clone();
  displayDepthImage(depthImage, allOnImage);

  cv::Mat planeOnImage = undistImage.clone();
  displayDepthImage(depthPlaneImage, planeOnImage);

  return 0;
}

void getDepthImage(PointCloud::Ptr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage)
{
  for (int i = 0; i < cloud->points.size(); i++)
  {
    Eigen::Vector3d p3Dlidar;
    p3Dlidar << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z;

    // ignore points with negative depth
    Eigen::Matrix3d R_lidar2cam = cam.getR();
    Eigen::Vector3d t_lidar2cam = cam.getT();
    Eigen::Vector3d p3Dcam;
    p3Dcam = R_lidar2cam * p3Dlidar + t_lidar2cam;
    if (p3Dcam(2) < 0.0)
      continue;

    // ignore points too close or too far
    double curDepth = p3Dcam.norm();
    if (curDepth < minDepth || curDepth > maxDepth)
      continue;

    // get corresponding 2D point index on the image
    Eigen::Vector2d p2D;
    p2D = cam.projectPoint(p3Dlidar(0), p3Dlidar(1), p3Dlidar(2));

    int row = (int)p2D(1);
    int col = (int)p2D(0);

    if (row < 0 || row >= depthImage.rows || col < 0 || col >= depthImage.cols)
      continue;

    // replace if it is closer
    double& preDepth = depthImage.at<double>(row, col);

    if (curDepth < preDepth || preDepth < 0.0)
      preDepth = curDepth;
  }
}

void displayDepthImage(const cv::Mat_<double>& depthImage, cv::Mat& undistImage)
{
  for (int i = 0; i < depthImage.rows; i++)
    for (int j = 0; j < depthImage.cols; j++)
    {
      double depth = depthImage.at<double>(i, j);

      if (depth < 0.0)
        continue;

      // draw on image
      double minZ = 3.0;
      double maxZ = 100.0;
      double value = (depth - minZ) / (maxZ - minZ) * 255;
      int color_r = value > 128 ? (value - 128) * 2 : 0;
      int color_g = value < 128 ? 2 * value : 255 - ((value - 128) * 2);
      int color_b = value < 128 ? 255 - (2 * value) : 0;
      cv::circle(undistImage, cv::Point(j, i), 1,
                 cv::Scalar(color_b, color_g, color_r), -1);
    }

  // show projection image
  cv::imshow("result", undistImage);
  cv::waitKey(0);
}
