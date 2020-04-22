#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <Eigen/Dense>

#include <psl_base/common.h>
#include <psl_base/exception.h>
#include <psl_cudaBase/cudaFishEyeImageProcessor.h>
#include <psl_stereo/cudaFishEyePlaneSweep.h>

#include <opencv2/opencv.hpp>
#include <opencv2/ccalib/omnidir.hpp>

#include <omp.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#define GetCurrentDir getcwd

using namespace std;

std::string GetCurrentWorkingDir(void)
{
  char buff[FILENAME_MAX];
  GetCurrentDir(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

bool isDataFileValid(ifstream& dataFile)
{
  return dataFile.is_open() &&
         dataFile.peek() != std::ifstream::traits_type::eof();
}

struct Distortion
{
  double k1;
  double k2;
  double p1;
  double p2;
};

struct CalibInfo
{
  uint64_t timestamp;
  cv::Mat calibImage;
  PSL::FishEyeCameraMatrix<double> cam;
  Distortion distCoefficients;
  Eigen::Matrix4d T_Lidar2Cam;
  int camTimeShift_MS;
  PointCloud::Ptr calibCloud;
};

struct Pose
{
  uint64_t timestamp;
  float x;
  float y;
  float z;
  float roll;
  float pitch;
  float yaw;
};

struct Frame
{
  uint64_t timestamp;
  string imagePath;
  cv::Mat image;
};

bool readData(string filePath, CalibInfo& calibInfo, vector<Pose>& vehiclePoses,
              vector<Frame>& camFrames, vector<string>& cloudFiles,
              string& outputFolder);

void findPoseAtTimestampUS(const uint64_t timestamp, const vector<Pose>& poses,
                           const int startInd, Pose& outputPose,
                           int& outputInd);
void findPoseAtTimestampNS(const uint64_t timestamp, const vector<Pose>& poses,
                           const int startInd, Pose& outputPose,
                           int& outputInd);

Eigen::Matrix4d convertPoseToEigenMatrix(const Pose& pose);

void getDepthImage(PointCloud::ConstPtr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage);
void displayDepthImage(const cv::Mat_<double>& depthImage,
                       cv::Mat& undistImage);

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    cout << "Usage: ./syncCameraLidar syncCameraLidarData.txt" << endl;
    return 1;
  }

  cout << "Current working directory: " << GetCurrentWorkingDir() << endl;

  // read data
  string dataFilePath = argv[1];

  CalibInfo calibInfo;
  vector<Pose> vehiclePoses;
  vector<Frame> camFrames;
  vector<string> cloudFiles;
  string outputFolder;

  if (!readData(dataFilePath, calibInfo, vehiclePoses, camFrames, cloudFiles,
                outputFolder))
  {
    cerr << "Failed to read data from: " << dataFilePath << endl;
    return 1;
  }
  else
    cout << "Finished loading data!" << endl;

  string cameraFramePosesPath = outputFolder + "/cameraFramePoses.txt";
  string syncPointCloudsFolder = outputFolder + "/syncPointClouds";
  string syncPointCloudsPath = syncPointCloudsFolder + "/syncPointClouds.txt";
  ofstream cameraFramePose(cameraFramePosesPath.c_str());
  ofstream syncPointClouds(syncPointCloudsPath.c_str());

  // output description
  cout << "Output camera poses path: " << cameraFramePosesPath << endl;
  cameraFramePose << "# timestamp x y z roll pitch yaw" << endl;

  cout << "Output lidar point clouds path: " << syncPointCloudsPath << endl;

  // calculate corrected transformation from CAM to LiDAR
  // -- find the pose matching to the calib timestamp
  Pose calibPose;
  int calibInd;
  findPoseAtTimestampUS(calibInfo.timestamp, vehiclePoses, 0, calibPose,
                        calibInd);

  if (calibInd >= 0)
    cout << "Find the matching pose at #" << calibInd
         << " with actual timestamp: " << calibPose.timestamp << endl;
  else
  {
    cerr << "Can not find the matching pose given the timestamp: "
         << calibInfo.timestamp << endl;
    return 1;
  }

  // -- find the pose matching to the shifted calib timestamp
  uint64_t shiftedTimestampCalib =
      calibInfo.timestamp + calibInfo.camTimeShift_MS * 1e3;

  cout << "Input estimated time shift (ms): " << calibInfo.camTimeShift_MS
       << endl;

  Pose shiftedPose;
  int shiftedInd;
  findPoseAtTimestampUS(shiftedTimestampCalib, vehiclePoses, 0, shiftedPose,
                        shiftedInd);

  if (shiftedInd >= 0)
    cout << "Find the matching pose at #" << shiftedInd
         << " with actual timestamp: " << shiftedPose.timestamp << endl;
  else
  {
    cerr << "Can not find the matching pose given the timestamp: "
         << shiftedTimestampCalib << endl;
    return 1;
  }

  cout << "Actual time shift (ms): "
       << ((double)shiftedPose.timestamp - (double)calibPose.timestamp) / 1e3
       << endl;

  // -- compute the corrected transformation
  //       (shifted)     (calib)
  //      (cam time)   (lidar time)
  //         lidar         lidar'
  //          ^           . ^
  //          |    (calib)  |
  //          |   .         |
  //         cam           cam'
  //          ^             ^
  //          |             |
  //        vehicle ----> vehicle

  Eigen::Matrix4d T_Calib = convertPoseToEigenMatrix(calibPose);
  Eigen::Matrix4d T_ShiftedCalib = convertPoseToEigenMatrix(shiftedPose);

  Eigen::Matrix4d T_Vehicle2Cam = Eigen::Matrix4d::Identity();
  T_Vehicle2Cam.topLeftCorner(3, 3) = calibInfo.cam.getR();
  T_Vehicle2Cam.topRightCorner(3, 1) = calibInfo.cam.getT();

  Eigen::Matrix4d T_Cam2Vehicle = T_Vehicle2Cam.inverse();

  Eigen::Matrix4d T_Shfited2Calib = T_Calib.inverse() * T_ShiftedCalib;

  Eigen::Matrix4d T_Lidar2Cam_sync = T_Cam2Vehicle.inverse() * T_Shfited2Calib *
                                     T_Cam2Vehicle * calibInfo.T_Lidar2Cam;

  Eigen::Matrix4d T_Cam2Lidar = T_Lidar2Cam_sync.inverse();
  Eigen::Matrix3d R_Cam2Lidar = T_Cam2Lidar.topLeftCorner(3, 3);
  Eigen::Vector3d euler_Cam2Lidar = R_Cam2Lidar.eulerAngles(2, 1, 0);

  cout << "Previous T_Lidar2Cam: \n" << calibInfo.T_Lidar2Cam << endl;
  cout << "Sync T_Lidar2Cam: \n" << T_Lidar2Cam_sync << endl;
  cout << "Sync T_Cam2Lidar: \n" << T_Cam2Lidar << endl;
  cout << "Sync euler_Cam2Lidar: \n" << euler_Cam2Lidar << endl;

  // test corrected calibration transformation by projecting cloud to image
  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;
  size_t startCloudInd = 0;
  size_t startPoseInd = 0;

  for (size_t i = 0; i < camFrames.size(); i++)
  {
    Frame& curFrame = camFrames[i];

    // -- load frame image
    curFrame.image = cv::imread(curFrame.imagePath);

    // -- undistort frame image
    cv::Mat imGray;
    cv::cvtColor(curFrame.image, imGray, CV_BGR2GRAY);
    devImg.allocatePitchedAndUpload(imGray);
    cFEIP.setInputImg(devImg, calibInfo.cam);

    double k1 = calibInfo.distCoefficients.k1;
    double k2 = calibInfo.distCoefficients.k2;
    double p1 = calibInfo.distCoefficients.p1;
    double p2 = calibInfo.distCoefficients.p2;
    pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>> undistRes =
        cFEIP.undistort(1.0, 1.0, k1, k2, p1, p2);

    cv::Mat undistImage;
    undistRes.first.download(undistImage);

    cv::cvtColor(undistImage, undistImage, CV_GRAY2BGR);

    // -- find the corresponding cloud to current frame
    PointCloud::Ptr curCloud(new PointCloud());
    bool foundCloud = false;

    for (size_t cloudInd = startCloudInd; cloudInd < cloudFiles.size();
         cloudInd++)
    {
      // ---- get timestamp of the cloud
      string cloudFile = cloudFiles[cloudInd];
      uint64_t tCloud = atoll(cloudFile.substr(cloudFile.size() - 23,
                                               cloudFile.size() - 4).c_str());

      if (curFrame.timestamp == tCloud)
      {
        pcl::PLYReader pclReader;
        pclReader.read(cloudFile, *curCloud);

        foundCloud = true;
        startCloudInd = cloudInd;
        break;
      }
    }

    if (!foundCloud)
    {
      cout << "Can not find corresponding cloud for timestamp: "
           << curFrame.timestamp << endl;
      continue;
    }

    if (curCloud->points.size() == 0)
    {
      cout << "No points loaded from: " << cloudFiles[startCloudInd] << endl;
      continue;
    }

    // -- get pose transformation corresponding to the time shift
    Pose curPose;
    int curInd;
    findPoseAtTimestampNS(curFrame.timestamp, vehiclePoses, startPoseInd,
                          curPose, curInd);

    uint64_t shiftedTimestamp =
        curFrame.timestamp + calibInfo.camTimeShift_MS * 1e6;
    Pose shiftedPose;
    int shiftedInd;
    findPoseAtTimestampNS(shiftedTimestamp, vehiclePoses, startPoseInd,
                          shiftedPose, shiftedInd);

    startPoseInd = shiftedInd < curInd ? shiftedInd : curInd;

    Eigen::Matrix4d T_Cur = convertPoseToEigenMatrix(curPose);
    Eigen::Matrix4d T_Shifted = convertPoseToEigenMatrix(shiftedPose);
    Eigen::Matrix4d T_Cur2Shifted = T_Shifted.inverse() * T_Cur;

    // -- transform the cloud to the shifted timestamp of vehicle pose
    PointCloud::Ptr shiftedCloudVehicle(new PointCloud());

    Eigen::Matrix4d T_Lidar2ShiftedVehicle =
        T_Cur2Shifted * T_Cam2Vehicle * T_Lidar2Cam_sync;

    pcl::transformPointCloud(*curCloud, *shiftedCloudVehicle,
                             T_Lidar2ShiftedVehicle.cast<float>());

    // -- project the cloud to the frame of camera
    double minDepth = 10.0;
    double maxDepth = 200.0;
    cv::Mat_<double> depthImage(undistImage.rows, undistImage.cols, -1.0);
    getDepthImage(shiftedCloudVehicle, minDepth, maxDepth, calibInfo.cam,
                  depthImage);

    cv::Mat dispImage = undistImage.clone();
    displayDepthImage(depthImage, dispImage);
  }

  cout << "Finished!" << endl;

  return 0;
}

bool readData(string filePath, CalibInfo& calibInfo, vector<Pose>& vehiclePoses,
              vector<Frame>& camFrames, vector<string>& cloudFiles,
              string& outputFolder)
{
  ifstream dataFile(filePath.c_str());

  // check data file
  if (!isDataFileValid(dataFile))
  {
    cerr << "Can not open txt file or it's empty: " << filePath << endl;
    return false;
  }

  // read data
  string descpt;
  dataFile >> descpt >> calibInfo.timestamp;

  string rawImageFile;
  dataFile >> descpt >> rawImageFile;

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  double xi;
  double k1, k2, p1, p2;
  double roll, pitch, yaw, t_x, t_y, t_z;
  double roll_cam2lidar, pitch_cam2lidar, yaw_cam2lidar, t_x_cam2lidar,
      t_y_cam2lidar, t_z_cam2lidar;

  dataFile >> descpt >> K(0, 0) >> K(0, 1) >> K(0, 2) >> K(1, 1) >> K(1, 2);
  dataFile >> xi;
  dataFile >> k1 >> k2 >> p1 >> p2;

  dataFile >> descpt >> roll >> pitch >> yaw >> t_x >> t_y >> t_z;

  dataFile >> descpt >> roll_cam2lidar >> pitch_cam2lidar >> yaw_cam2lidar >>
      t_x_cam2lidar >> t_y_cam2lidar >> t_z_cam2lidar;

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

  calibInfo.cam =
      PSL::FishEyeCameraMatrix<double>(K, R_vehicle2cam, t_vehicle2cam, xi);
  calibInfo.distCoefficients = {k1, k2, p1, p2};

  // -- undistort image
  PSL_CUDA::DeviceImage devImg;
  PSL_CUDA::CudaFishEyeImageProcessor cFEIP;

  cv::Mat rawImage = cv::imread(rawImageFile);
  cv::cvtColor(rawImage, rawImage, CV_BGR2GRAY);
  devImg.allocatePitchedAndUpload(rawImage);
  cFEIP.setInputImg(devImg, calibInfo.cam);

  pair<PSL_CUDA::DeviceImage, PSL::FishEyeCameraMatrix<double>> undistRes =
      cFEIP.undistort(1.0, 1.0, k1, k2, p1, p2);

  cv::Mat undistImage;
  undistRes.first.download(undistImage);

  cv::cvtColor(undistImage, undistImage, CV_GRAY2BGR);

  calibInfo.calibImage = undistImage;

  // -- set lidar
  Eigen::Matrix3d R_cam2lidar;
  R_cam2lidar = Eigen::AngleAxisd(yaw_cam2lidar, Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(pitch_cam2lidar, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(roll_cam2lidar, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t_cam2lidar(t_x_cam2lidar, t_y_cam2lidar, t_z_cam2lidar);

  Eigen::Matrix4d T_cam2lidar = Eigen::Matrix4d::Identity();
  T_cam2lidar.topLeftCorner(3, 3) = R_cam2lidar;
  T_cam2lidar.topRightCorner(3, 1) = t_cam2lidar;

  calibInfo.T_Lidar2Cam = T_cam2lidar.inverse();

  // -- load time shift
  dataFile >> descpt >> calibInfo.camTimeShift_MS;

  // -- load point cloud
  string calibCloudFile;
  dataFile >> descpt >> calibCloudFile;

  pcl::PLYReader pclReader;
  PointCloud::Ptr calibCloud(new PointCloud());
  pclReader.read(calibCloudFile, *calibCloud);

  if (calibCloud->points.size() == 0)
  {
    cout << "No calib points loaded from: " << calibCloudFile << endl;
    return false;
  }
  else
    cout << "Valid calib lidar points loaded: " << calibCloud->points.size()
         << endl;

  calibInfo.calibCloud = calibCloud;

  // -- load vehicle poses
  string poseDataFilePath;
  dataFile >> descpt >> poseDataFilePath;
  ifstream poseDataFile(poseDataFilePath.c_str());
  if (!isDataFileValid(poseDataFile))
  {
    cerr << "Can not open txt file or it's empty: " << poseDataFilePath << endl;
    return false;
  }

  string line;
  while (getline(poseDataFile, line))
  {
    if (line[0] == '#')
      continue;

    Pose p;
    std::stringstream lineStr(line);
    lineStr >> p.timestamp >> p.x >> p.y >> p.z >> p.roll >> p.pitch >> p.yaw;

    vehiclePoses.push_back(p);
  }

  cout << "Loaded vehicle poses: " << vehiclePoses.size() << endl;

  // -- load camera frames
  string cameraFramesPath;
  dataFile >> descpt >> cameraFramesPath;
  ifstream cameraFramesFile(cameraFramesPath.c_str());
  if (!isDataFileValid(cameraFramesFile))
  {
    cerr << "Can not open txt file or it's empty: " << cameraFramesPath << endl;
    return false;
  }

  while (getline(cameraFramesFile, line))
  {
    Frame f;
    f.imagePath = line;
    f.timestamp = atoll(line.substr(line.size() - 23, line.size() - 4).c_str());

    camFrames.push_back(f);
  }

  cout << "Loaded camera timestamps: " << camFrames.size() << endl;

  // -- load cloud paths
  string cloudsPath;
  dataFile >> descpt >> cloudsPath;
  ifstream cloudsFile(cloudsPath.c_str());
  if (!isDataFileValid(cloudsFile))
  {
    cerr << "Can not open txt file or it's empty: " << cloudsPath << endl;
    return false;
  }

  while (cloudsFile >> line)
  {
    cloudFiles.push_back(line);
  }

  cout << "Loaded cloud files: " << cloudFiles.size() << endl;

  // -- load output folder
  dataFile >> descpt >> outputFolder;
  cout << "Output folder: " << outputFolder << endl;

  return true;
}

void findPoseAtTimestampUS(const uint64_t timestamp, const vector<Pose>& poses,
                           const int startInd, Pose& outputPose, int& outputInd)
{
  // Note: The timestamps of poses should be in ascending order.

  // find the closest timestamp ahead
  int newStartInd = startInd > 0 ? startInd : 0;
  auto iter = find_if(poses.begin() + newStartInd, poses.end(),
                      [&timestamp](const Pose& p)
                      {
                        return timestamp <= p.timestamp;
                      });

  if (iter == poses.end())
  {
    cerr << "Can not find the valid timestamp in poses vector." << endl;

    outputInd = -1;
    return;
  }

  // check if the timestamp behind is closer
  if (iter - 1 >= poses.begin() &&
      timestamp - (*(iter - 1)).timestamp < (*iter).timestamp - timestamp)
  {
    outputInd = iter - poses.begin() - 1;
    outputPose = poses[outputInd];
  }
  else
  {
    outputInd = iter - poses.begin();
    outputPose = poses[outputInd];
  }
}

void findPoseAtTimestampNS(const uint64_t timestamp, const vector<Pose>& poses,
                           const int startInd, Pose& outputPose, int& outputInd)
{
  uint64_t newTimestamp = timestamp / 1e3;
  findPoseAtTimestampUS(newTimestamp, poses, startInd, outputPose, outputInd);
}

Eigen::Matrix4d convertPoseToEigenMatrix(const Pose& pose)
{
  Eigen::Matrix3d R;
  R = Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX());

  Eigen::Vector3d t(pose.x, pose.y, pose.z);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.topLeftCorner(3, 3) = R;
  T.topRightCorner(3, 1) = t;

  return T;
}

void getDepthImage(PointCloud::ConstPtr cloud, double minDepth, double maxDepth,
                   PSL::FishEyeCameraMatrix<double>& cam,
                   cv::Mat_<double>& depthImage)
{
#pragma omp parallel for
  for (size_t i = 0; i < cloud->points.size(); i++)
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
