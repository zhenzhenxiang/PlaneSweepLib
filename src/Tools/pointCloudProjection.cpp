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

using namespace std;

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    cout << "Usage: ./pointCloudProjection projectTest.txt" << endl;
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
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t(t_x, t_y, t_z);

  PSL::FishEyeCameraMatrix<double> cam(K, R, t, xi);

  // -- load point cloud
  vector<Eigen::Vector3d> pointCloud;
  vector<double> intensities;
  ifstream pclDataFile(pointCloudFile.c_str());

  string line;
  while (getline(pclDataFile, line))
  {
    if (line[0] == 'n')
      continue;

    stringstream lineStr(line);
    string tmpStr;
    vector<double> point;
    for (int i = 0; i < 4; i++)
    {
      getline(lineStr, tmpStr, ',');
      point.push_back(atof(tmpStr.c_str()));
    }

    Eigen::Vector3d pos;
    pos << point[0], point[1], point[2];

    pointCloud.push_back(pos);
    intensities.push_back(point[3]);
  }

  cout << "Data loading finished. Total valid lidar points: "
       << pointCloud.size() << endl;

  // undistort image
  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  cv::cvtColor(image, image, CV_BGR2GRAY);
  devImg.allocatePitchedAndUpload(image);
  cFEIP.setInputImg(devImg, cam);

  std::pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double> > undistRes =
      cFEIP.undistort(1.0, 1.0, k1, k2, p1, p2);

  cv::Mat undistImage;
  undistRes.first.download(undistImage);

  cv::cvtColor(undistImage, undistImage, CV_GRAY2BGR);

  // project lidar points
  double minDist = 5.0;
  double maxDist = 100.0;
  for (int i = 0; i < pointCloud.size(); i++)
  {
    Eigen::Vector3d p3Dworld = pointCloud[i];

    // ignore points too close or too far
    if (p3Dworld.norm() < minDist || p3Dworld.norm() > maxDist)
      continue;

    // ignore points with negative depth in camera frame
    Eigen::Vector3d p3Dcam;
    p3Dcam = cam.getR() * p3Dworld + cam.getT();
    if (p3Dcam[2] < 0)
      continue;

    Eigen::Vector2d p2D;
    p2D = cam.projectPoint(p3Dworld(0), p3Dworld(1), p3Dworld(2));

    // draw on image
    cv::circle(undistImage, cv::Point(p2D(0), p2D(1)), 1, cv::Scalar(0, 0, 255),
               -1);
  }

  // show projection image
  cv::imshow("result", undistImage);
  cv::waitKey(0);

  // test opencv omnicam model
  cv::Mat K_cv = cv::Mat::eye(3, 3, CV_64F);
  K_cv.at<double>(0, 0) = K(0, 0);
  K_cv.at<double>(1, 1) = K(1, 1);
  K_cv.at<double>(0, 2) = K(0, 2);
  K_cv.at<double>(1, 2) = K(1, 2);

  cv::Mat D_cv = cv::Mat::zeros(4, 1, CV_64F);
  D_cv.at<double>(0) = k1;
  D_cv.at<double>(1) = k2;
  D_cv.at<double>(2) = p1;
  D_cv.at<double>(3) = p2;

  cv::Mat R_cv = cv::Mat::zeros(3, 3, CV_64F);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
      R_cv.at<double>(i, j) = R(i, j);
    }

  cv::Mat rvec;
  Rodrigues(R_cv, rvec);

  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  tvec.at<double>(0) = t(0);
  tvec.at<double>(1) = t(1);
  tvec.at<double>(2) = t(2);

  cv::Mat K_new_cv = cv::Mat::eye(3, 3, CV_64F);
  K_new_cv.at<double>(0, 0) = image.cols / 3.1415;
  K_new_cv.at<double>(1, 1) = image.rows / 3.1415;
  cv::Mat undistortImageCV;
  cv::omnidir::undistortImage(image, undistortImageCV, K_cv, D_cv, xi,
                              cv::omnidir::RECTIFY_LONGLATI, K_new_cv);

  vector<cv::Point3d> points3D;
  for (int i = 0; i < pointCloud.size(); i++)
  {
    Eigen::Vector3d p3Dworld = pointCloud[i];

    // ignore points too close or too far
    if (p3Dworld.norm() < minDist || p3Dworld.norm() > maxDist)
      continue;

    // ignore points with negative depth in camera frame
    Eigen::Vector3d p3Dcam;
    p3Dcam = cam.getR() * p3Dworld + cam.getT();
    if (p3Dcam[2] < 0)
      continue;

    cv::Point3d p;
    p.x = p3Dworld(0);
    p.y = p3Dworld(1);
    p.z = p3Dworld(2);

    points3D.push_back(p);
  }

  vector<cv::Point2d> points2D;
  cv::omnidir::projectPoints(points3D, points2D, rvec, tvec, K_cv, xi, D_cv);

  image = cv::imread(imageFile);
  for (int i = 0; i < points2D.size(); i++)
  {
    cv::circle(image, points2D[i], 1, cv::Scalar(0, 0, 255), -1);
  }

  cv::imshow("result_cv", image);
  cv::waitKey(0);

  return 0;
}
