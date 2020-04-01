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
#include <omp.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace std;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;

void getDepthImage(PointCloud::ConstPtr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage);
void displayDepthImage(const cv::Mat_<double>& depthImage,
                       cv::Mat& undistImage);
void getFreespaceCloud(PointCloud::ConstPtr cloud, double minDepth,
                       double maxDepth, PSL::FishEyeCameraMatrix<double>& cam,
                       const cv::Mat& freespaceImage,
                       PointCloud::Ptr freespaceCloud);

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
  string imageFile, pointCloudFile, freespaceFile;
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  double xi;
  double k1, k2, p1, p2;
  double roll, pitch, yaw, t_x, t_y, t_z;
  double roll_cam2lidar, pitch_cam2lidar, yaw_cam2lidar, t_x_cam2lidar,
      t_y_cam2lidar, t_z_cam2lidar;

  dataFile >> descpt >> imageFile;

  dataFile >> descpt >> K(0, 0) >> K(0, 1) >> K(0, 2) >> K(1, 1) >> K(1, 2);
  dataFile >> xi;
  dataFile >> k1 >> k2 >> p1 >> p2;

  dataFile >> descpt >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;

  dataFile >> descpt >> roll_cam2lidar >> pitch_cam2lidar >> yaw_cam2lidar >>
      t_x_cam2lidar >> t_y_cam2lidar >> t_z_cam2lidar;

  dataFile >> descpt >> pointCloudFile;

  dataFile >> descpt >> freespaceFile;

  // -- set image
  cv::Mat image = cv::imread(imageFile);

  // -- set camera
  Eigen::Matrix3d R_cam2vehicle;
  R_cam2vehicle = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                  Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                  Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t_cam2vehicle(t_x, t_y, t_z);

  Eigen::Matrix4d T_cam2vehicle = Eigen::Matrix4d::Identity();
  T_cam2vehicle.topLeftCorner(3, 3) = R_cam2vehicle;
  T_cam2vehicle.topRightCorner(3, 1) = t_cam2vehicle;

  Eigen::Matrix3d R_vehicle2cam = R_cam2vehicle.transpose();
  Eigen::Vector3d t_vehicle2cam = -R_cam2vehicle.transpose() * t_cam2vehicle;

  PSL::FishEyeCameraMatrix<double> cam(K, R_vehicle2cam, t_vehicle2cam, xi);

  // -- set lidar
  Eigen::Matrix3d R_cam2lidar;
  R_cam2lidar = Eigen::AngleAxisd(yaw_cam2lidar, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_cam2lidar, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_cam2lidar, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t_cam2lidar(t_x_cam2lidar, t_y_cam2lidar, t_z_cam2lidar);

  Eigen::Matrix4d T_cam2lidar = Eigen::Matrix4d::Identity();
  T_cam2lidar.topLeftCorner(3, 3) = R_cam2lidar;
  T_cam2lidar.topRightCorner(3, 1) = t_cam2lidar;

  Eigen::Matrix4d T_lidar2vehicle = T_cam2vehicle * T_cam2lidar.inverse();

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

  // -- transform to vehicle frame
  PointCloud::Ptr inCloudVehicle(new PointCloud());
  pcl::transformPointCloud(*inCloud, *inCloudVehicle,
                           T_lidar2vehicle.cast<float>());

  // -- load freespace mask
  cv::Mat freespaceImage = cv::imread(freespaceFile, cv::IMREAD_GRAYSCALE);

  // get point cloud in freespace
  double minDepth = 3.0;
  double maxDepth = 150.0;
  PointCloud::Ptr freespaceCloud(new PointCloud());
  getFreespaceCloud(inCloudVehicle, minDepth, maxDepth, cam, freespaceImage,
                    freespaceCloud);

  cout << "Freespace cloud points: " << freespaceCloud->points.size() << endl;

  // save to ply file
  pcl::PLYWriter writer;
  string freespaceCloudFileName = pointCloudFile.replace(
      pointCloudFile.size() - 4, pointCloudFile.size(), "_freespace.ply");
  writer.write(freespaceCloudFileName, *freespaceCloud, true);

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

  seg.setInputCloud(freespaceCloud);
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
    planeCloud->push_back(freespaceCloud->points[planeInliners->indices[i]]);
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
  cv::Mat_<double> depthImage(image.rows, image.cols, -1.0);
  cv::Mat_<double> depthPlaneImage(image.rows, image.cols, -1.0);

  getDepthImage(inCloudVehicle, minDepth, maxDepth, cam, depthImage);
  getDepthImage(planeCloud, minDepth, maxDepth, cam, depthPlaneImage);

  // visualize depth image
  cv::Mat allOnImage = undistImage.clone();
  displayDepthImage(depthImage, allOnImage);

  cv::Mat planeOnImage = undistImage.clone();
  displayDepthImage(depthPlaneImage, planeOnImage);

  return 0;
}

void getDepthImage(PointCloud::ConstPtr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage)
{
#pragma omp parallel for
  for (int i = 0; i < cloud->points.size(); i++)
  {
    Eigen::Vector3d p3Dlidar;
    p3Dlidar << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z;

    // ignore points too close or too far
    double curDepth = p3Dlidar(0);

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
#pragma omp parallel for collapse(2)
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

void getFreespaceCloud(PointCloud::ConstPtr cloud, double minDepth,
                       double maxDepth, PSL::FishEyeCameraMatrix<double>& cam,
                       const cv::Mat& freespaceImage,
                       PointCloud::Ptr freespaceCloud)
{
// ref:
// https://stackoverflow.com/questions/18669296/c-openmp-parallel-for-loop-alternatives-to-stdvector/18671256#18671256
#pragma omp declare reduction(                                                 \
    merge : std::vector <                                                      \
    Point > : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

  std::vector<Point> points;

#pragma omp parallel for reduction(merge : points)
  for (int i = 0; i < cloud->points.size(); i++)
  {
    Eigen::Vector3d p3Dlidar;
    p3Dlidar << cloud->points[i].x, cloud->points[i].y, cloud->points[i].z;

    // ignore points too close or too far
    Eigen::Matrix3d R_lidar2cam = cam.getR();
    Eigen::Vector3d t_lidar2cam = cam.getT();
    Eigen::Vector3d p3Dcam;
    p3Dcam = R_lidar2cam * p3Dlidar + t_lidar2cam;
    double curDepth = p3Dcam(2);

    if (curDepth < minDepth || curDepth > maxDepth)
      continue;

    // get corresponding 2D point index on the image
    Eigen::Vector2d p2D;
    p2D = cam.projectPoint(p3Dlidar(0), p3Dlidar(1), p3Dlidar(2));

    int row = (int)p2D(1);
    int col = (int)p2D(0);

    if (row < 0 || row >= freespaceImage.rows || col < 0 ||
        col >= freespaceImage.cols)
      continue;

    // add free point
    uchar value = freespaceImage.at<uchar>(row, col);
    if (value > 250)
      points.push_back(cloud->points[i]);
  }

  for (auto& p : points)
    freespaceCloud->points.push_back(p);
}
